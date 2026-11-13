"""15-min / 0-4 h final comparison under the rolling protocol: Ours (frozen Chronos-2 + M1 + M2 + M3) vs strong baselines
(PatchTST, iTransformer, TiDE, DLinear, LightGBM-K, Chronos-2 full fine-tuning / LoRA in the M1 representation,
probabilistic smart persistence, geometry-blind Chronos-2, smart persistence). Every probabilistic competitor gets its own
M3 with step sizes chosen by online replay on ITS OWN calibration window (same recipe as Ours, delay 16).
Grid-code metrics: daytime accuracy 1 - RMSE/Cap (all cells, and with curtailment-bound cells exempted, as in the
Shandong assessment), per lead hour, per day. Paired day-block bootstrap CIs of Ours vs each competitor."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PVFM_NO_TORCH"] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from pvfm.common import STATIONS, LEVELS
from pvfm.protocol import FOLDS, REGIME
from pvfm.preds import ROOT, MID, regime
from pvfm.task15 import Station15, splits15, fm15_all, H15
from pvfm.radapt15 import usable
from pvfm.calib import groups, fit_static, online_hier_loc, metrics as cmetrics

FM = {'Chronos-2 geometry-blind': 'P_c96_pastfut', 'M1 only': 'Kc_c672_pastfut'}
FOLD = {'Ours': 'runs/radapt15/Kc_c672_pastfut/pooled_fmgeo', 'Full fine-tune c672': 'runs/rbase15/FT_full_c672',
        'LoRA c672': 'runs/rbase15/FT_lora_c672', 'PatchTST': 'runs/rbase15/DL_PatchTST', 'iTransformer': 'runs/rbase15/DL_iTransformer',
        'TiDE': 'runs/rbase15/DL_TiDE', 'DLinear': 'runs/rbase15/DL_DLinear', 'LightGBM-K': 'runs/rbase15/LGBM_K'}


def bind15(S):
    cap = S.cap; agc = np.nan_to_num(S.x['agc'], nan=cap); av = np.nan_to_num(S.x['avail'], nan=cap)
    return (agc < 0.97 * cap) & (S.P >= agc - 0.03 * cap) & (av > 0.05 * cap)


def psp_fit(S, o):
    fi = o[:, None] + np.arange(1, H15 + 1)[None]; r = (S.P[fi] - S.smart_persistence(o)) / S.Ceff[fi]; g = regime(S.Ceff[fi], S.cap)
    tab = np.zeros((H15, 3, len(LEVELS)))
    for h in range(H15):
        for k in range(3):
            m = g[:, h] == k
            tab[h, k] = np.quantile(r[m, h], LEVELS) if m.sum() > 30 else np.quantile(r[:, h], LEVELS)
    tab[..., MID] = 0.0
    return tab


def psp(S, o, tab):
    fi = o[:, None] + np.arange(1, H15 + 1)[None]; g = regime(S.Ceff[fi], S.cap)
    return np.sort(S.smart_persistence(o)[..., None] + tab[np.arange(H15)[None], g] * S.Ceff[fi][..., None], -1)


def load(S, st, proto, part, tab):
    o = splits15(S, proto)[part]; raw = {k: fm15_all(v, st) for k, v in FM.items()}
    for k, v in FOLD.items():
        f = ROOT / v / proto / f'{st}.npz'
        if f.exists():
            z = np.load(f); raw[k] = (z[f'o_{part}'], z[f'q_{part}'])
    for oo, _ in raw.values(): o = np.intersect1d(o, oo)
    fi = o[:, None] + np.arange(1, H15 + 1)[None]; cap = S.cap
    d = dict(o=o, cap=cap, H=H15, delay=H15, y=S.P[fi].astype(np.float32), Ceff=S.Ceff[fi].astype(np.float32), cz=S.cz[fi])
    d['reg'] = regime(d['Ceff'], cap)
    d['q'] = {k: np.clip(q[np.searchsorted(oo, o)], -0.1 * cap, 1.1 * cap) for k, (oo, q) in raw.items()}
    d['q']['Prob. smart persistence'] = np.clip(psp(S, o, tab), -0.1 * cap, 1.1 * cap)
    d['sp'] = S.smart_persistence(o); d['bind'] = bind15(S)[fi]
    return d


def m3(cal, te, k):
    qc = cal['q'][k]
    thg = fit_static(cal, qc, groups(cal, 'glob'), 'K', 1)[0]; thr = fit_static(cal, qc, groups(cal, 'reg'), 'K', 3) - thg[None]
    best = None
    for g in (0.0001, 0.0003, 0.001, 0.002):
        for gl in (0.0, 0.0001, 0.0003, 0.001):
            pv = cmetrics(cal, online_hier_loc(cal, qc, groups(cal, 'reg'), 'K', thg, thr, g, 1.0, gl))['pinball']
            if best is None or pv < best[0]: best = (pv, g, gl)
    return online_hier_loc(te, te['q'][k], groups(te, 'reg'), 'K', thg, thr, best[1], 1.0, best[2])


def one(args):
    st, proto = args
    if not (ROOT / FOLD['Ours'] / proto / f'{st}.npz').exists() or not usable(st, proto): return None
    S = Station15(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1])
    tab = psp_fit(S, splits15(S, proto)['train'])
    cal = load(S, st, proto, 'cal', tab); te = load(S, st, proto, 'test', tab)
    sysq = {k: m3(cal, te, k) for k in te['q']}
    cap = te['cap']; y = te['y']; day = te['Ceff'] > 0.05 * cap; dex = day & ~te['bind']
    pts = {k: q[..., MID] for k, q in sysq.items()}; pts['Smart persistence'] = te['sp']
    rows, pw, daily = [], {}, []; dkey = te['o'] // 96
    for k, med in pts.items():
        e = y - med; q = sysq.get(k)
        m = cmetrics(te, q) if q is not None else dict(mae=float(np.abs(e).mean()))
        m['acc_day'] = 1 - np.sqrt((e[day] ** 2).mean()) / cap; m['acc_day_exempt'] = 1 - np.sqrt((e[dex] ** 2).mean()) / cap
        for h in range(4):
            dd = day[:, 4 * h:4 * h + 4]; m[f'acc_h{h + 1}'] = 1 - np.sqrt((e[:, 4 * h:4 * h + 4][dd] ** 2).mean()) / cap
        m['frac_bind_day'] = float(te['bind'][day].mean())
        rows.append(dict(station=st, proto=proto, regime=REGIME.get(proto, 'main'), system=k, **m))
        pl = np.full(len(e), np.nan)
        if q is not None:
            ee = y[..., None] - q; pl = np.maximum(LEVELS * ee, (LEVELS - 1) * ee).mean((-1, -2))
        pw[k] = (np.abs(e).mean(-1), pl)
        for dd_ in np.unique(dkey):
            w = dkey == dd_; ma = day[w]; m4 = day[w][:, 12:16]; mx = dex[w]
            if ma.sum() < 20: continue
            daily.append(dict(station=st, proto=proto, regime=REGIME.get(proto, 'main'), system=k, day=int(dd_),
                              acc=1 - np.sqrt((e[w][ma] ** 2).mean()) / cap,
                              acc_h4=1 - np.sqrt((e[w][:, 12:16][m4] ** 2).mean()) / cap if m4.sum() else np.nan,
                              acc_exempt=1 - np.sqrt((e[w][mx] ** 2).mean()) / cap if mx.sum() >= 20 else np.nan))
    return dict(rows=rows, pw=pw, day=dkey, proto=proto, daily=daily)


def boot(res, a, b, k, B=2000):
    rng = np.random.default_rng(0); tot = np.zeros(B); pts = []; base = []
    for r in res:
        if a not in r['pw'] or b not in r['pw']: continue
        da = r['pw'][a][k] - r['pw'][b][k]
        if np.isnan(da).all(): return None
        ud, inv = np.unique(r['day'], return_inverse=True); sums = np.bincount(inv, da); cnt = np.bincount(inv)
        pick = rng.integers(0, len(ud), (B, len(ud))); tot += sums[pick].sum(1) / cnt[pick].sum(1); pts.append(da.mean()); base.append(r['pw'][b][k].mean())
    if not pts: return None
    tot /= len(pts); d = float(np.mean(pts))
    return d, 100 * d / float(np.mean(base)), float(np.quantile(tot, .025)), float(np.quantile(tot, .975)), len(pts)


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    with Pool(12) as p:
        res = [r for r in p.map(one, [(st, pr) for st in STATIONS for pr in ['main'] + list(FOLDS)]) if r]
    R = pd.DataFrame(sum([r['rows'] for r in res], [])); R.to_csv(out / 'rows.csv', index=False)
    pd.DataFrame(sum([r['daily'] for r in res], [])).to_csv(out / 'daily.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    cols = ['mae', 'pinball', 'qce', 'cov80', 'acc_day', 'acc_day_exempt', 'acc_h1', 'acc_h4']
    T = R[R.proto != 'main'].groupby('system')[cols].mean().sort_values('mae'); print(T.round(4).to_string()); T.to_csv(out / 'table_roll.csv')
    print(R.groupby(['regime', 'system'])[cols].mean().round(4).to_string())
    ci = []
    for reg in ('ALL_ROLL', 'free', 'transition', 'dispatch', 'main'):
        rr = [r for r in res if (reg == 'ALL_ROLL' and r['proto'] != 'main') or REGIME.get(r['proto'], 'main') == reg]
        for b in sorted({k for r in rr for k in r['pw']} - {'Ours'}):
            for k, met in ((0, 'mae'), (1, 'pinball')):
                x = boot(rr, 'Ours', b, k)
                if x: ci.append(dict(regime=reg, ref=b, metric=met, diff=x[0], rel_pct=x[1], lo=x[2], hi=x[3], n=x[4], sig=bool(x[3] < 0 or x[2] > 0)))
    C = pd.DataFrame(ci); C.to_csv(out / 'ci.csv', index=False)
    C['cell'] = [f"{p:+.1f}%{'*' if s else ''}" for p, s in zip(C.rel_pct, C.sig)]
    print(C.pivot_table(index='ref', columns=['metric', 'regime'], values='cell', aggfunc='first').to_string())
