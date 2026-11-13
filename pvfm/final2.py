"""Final comparison (rolling 5 folds + main). Ours = Chronos-2 + M1(+spatial) + M2 + M3. Every probabilistic competitor
is also given the same M3 (their calibrated best). Paired day-block bootstrap CIs of Ours vs each, per regime and pooled."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PVFM_NO_TORCH"] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from pvfm.common import STATIONS, LEVELS
from pvfm.protocol import FOLDS, REGIME, available, splits
from pvfm.rpreds import load, station, MID, ROOT
from pvfm.calib import groups, fit_static, online_hier_loc, metrics as cmetrics
from pvfm.probbase import chpeen, psp_fit, psp

SYS = {'Ours': 'fold:radapt/Kc_c8192_pastfutsp/pooled_fmgeo', 'Ours_noSpatial': 'fold:radapt/Kc_c8192_pastfut/pooled_fmgeo',
       'TimesFM+modules': 'fold:radapt/timesfm_Kc_c8192_none/pooled_fmgeo',
       'FT_full_c2880': 'fold:rbase/FT_full_ALL_c2880', 'FT_full_c8192': 'fold:rbase/FT_full_ALL_c8192', 'LoRA_c2880': 'fold:rbase/FT_lora_ALL_c2880',
       'PatchTST': 'fold:rbase/DL_PatchTST', 'iTransformer': 'fold:rbase/DL_iTransformer', 'TiDE': 'fold:rbase/DL_TiDE', 'DLinear': 'fold:rbase/DL_DLinear',
       'LightGBM-K': 'fold:rbase/LGBM_K_ALL', 'Chronos-2 geometry-blind': 'fm:P_c96_past', 'TimesFM geometry-blind': 'fm:timesfm_P_c96_none',
       'FT geometry-blind c96': 'fold:rbase/FT_full_P_c96', 'FT geometry-blind c2880': 'fold:rbase/FT_full_P_c2880'}


def present(src, st, proto):
    kind, rest = src.split(':', 1)
    if kind == 'fm': return (ROOT / 'runs/stageA' / rest / st / 'all.npz').exists()
    return (ROOT / 'runs' / rest / proto / f'{st}.npz').exists()


def m3(dc, dt, name):
    qc = dc['q'][name]
    thg = fit_static(dc, qc, groups(dc, 'glob'), 'K', 1)[0]; thr = fit_static(dc, qc, groups(dc, 'reg'), 'K', 3) - thg[None]
    return online_hier_loc(dt, dt['q'][name], groups(dt, 'reg'), 'K', thg, thr, 0.002, 1.0, 0.002)


def one(args):
    st, proto = args; S = station(st)
    if proto != 'main' and not available(S, proto): return None
    src = {k: v for k, v in SYS.items() if present(v, st, proto)}
    dc = load(st, src, 'cal', proto); dt = load(st, src, 'test', proto)
    for d in (dc, dt):  # loose physical-range guard [-0.1, 1.1] x cap (removes non-physical FM quantile blow-ups without creating ties at 0), applied identically to every forecaster
        for k in d['q']: d['q'][k] = np.clip(d['q'][k], -0.1 * S.cap, 1.1 * S.cap)
    sp = splits(S, proto); tr_o = sp['train']; tr_o = tr_o[tr_o > 1440 * 30]; tab = psp_fit(S, tr_o)
    for d in (dc, dt):
        d['q']['Prob. smart persistence'] = np.clip(psp(S, d['o'], tab), -0.1 * S.cap, 1.1 * S.cap).astype(np.float32); d['q']['CH-PeEn'] = np.clip(chpeen(S, d['o']), -0.1 * S.cap, 1.1 * S.cap).astype(np.float32)
    rows, pw = [], {}
    fi = dt['o'][:, None] + np.arange(1, 97)[None]
    sp_point = np.nan_to_num(S.x['P'])[dt['o']][:, None] * S.Ceff[fi] / S.Ceff[dt['o']][:, None]
    e = np.abs(dt['y'] - sp_point); rows.append(dict(station=st, proto=proto, regime=REGIME.get(proto, 'main'), system='Smart persistence', calib='point', mae=e.mean(), nmae=e.mean() / S.cap))
    pw[('Smart persistence', 'point')] = (e.mean(-1), np.full(len(e), np.nan))
    for name in dt['q']:
        for cal in ('raw', 'M3'):
            q = dt['q'][name] if cal == 'raw' else m3(dc, dt, name)
            m = cmetrics(dt, q); rows.append(dict(station=st, proto=proto, regime=REGIME.get(proto, 'main'), system=name, calib=cal, **m))
            ee = dt['y'][..., None] - q
            pw[(name, cal)] = (np.abs(dt['y'] - q[..., MID]).mean(-1), np.maximum(LEVELS * ee, (LEVELS - 1) * ee).mean((-1, -2)))
    return dict(rows=rows, pw=pw, day=dt['o'] // 1440, proto=proto)


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
    ci = []
    for reg in ('ALL_ROLL', 'free', 'transition', 'dispatch', 'main'):
        rr = [r for r in res if (reg == 'ALL_ROLL' and r['proto'] != 'main') or REGIME.get(r['proto'], 'main') == reg]
        for b in sorted({k[0] for r in rr for k in r['pw']}):
            if b == 'Ours': continue
            for k, met in ((0, 'mae'), (1, 'pinball')):
                bb = (b, 'point') if b == 'Smart persistence' else (b, 'M3')
                x = boot(rr, ('Ours', 'M3'), bb, k)
                if x: ci.append(dict(regime=reg, ref=b, metric=met, diff=x[0], rel_pct=x[1], lo=x[2], hi=x[3], n=x[4], sig=bool(x[3] < 0 or x[2] > 0)))
    C = pd.DataFrame(ci); C.to_csv(out / 'ci.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    T = R[(R.proto != 'main') & (R.calib.isin(['M3', 'point']))].groupby(['system'])[['mae', 'nmae', 'pinball', 'crps_trap', 'qce', 'cov80', 'worst_reg_ce80']].mean().sort_values('mae')
    print('== rolling folds, calibrated (M3) or point'); print(T.round(4).to_string()); T.to_csv(out / 'table_roll.csv')
    C['cell'] = [f"{p:+.1f}%{'*' if s else ''}" for p, s in zip(C.rel_pct, C.sig)]
    print(C.pivot_table(index='ref', columns=['metric', 'regime'], values='cell', aggfunc='first').to_string())
