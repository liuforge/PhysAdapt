"""Direct comparison with PARA-PV (authors' code, all branches, one model per station and fold as in their design) on the
15-min / 0-4 h task, rolling folds. PARA-PV emits the levels 0.05, 0.1, 0.5, 0.9, 0.95, so all methods are scored on these
five levels: MAE of the median, pinball loss averaged over the five levels, coverage of the 80% and 90% central intervals,
daytime accuracy 1-RMSE/cap, and the 2026 grid-code penalty (as in gridcode15, submitted forecast = median).
Ours and the references are shown with M3 (as in the paper) and Ours also without M3. Paired day-block bootstrap."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PVFM_NO_TORCH"] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from pvfm import t15_full as T
from pvfm.gridcode15 import reference, penalty, RULES
from pvfm.common import STATIONS, LEVELS
from pvfm.protocol import FOLDS, REGIME
from pvfm.preds import ROOT
from pvfm.task15 import Station15, splits15, H15
from pvfm.radapt15 import usable

L5 = np.array([0.05, 0.1, 0.5, 0.9, 0.95]); IDX = [int(np.argmin(np.abs(LEVELS - t))) for t in L5]
REFS = ['Full fine-tune c672', 'PatchTST', 'LightGBM-K', 'Prob. smart persistence']


def one(args):
    st, proto = args
    f = ROOT / 'runs/rbase15' / os.environ.get('PARAPV_TAG', 'PARA_PV') / proto / f'{st}.npz'
    if not f.exists() or not usable(st, proto): return None
    S = Station15(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1]); tab = T.psp_fit(S, splits15(S, proto)['train'])
    cal = T.load(S, st, proto, 'cal', tab); te = T.load(S, st, proto, 'test', tab)
    z = np.load(f); o_pp = z['o_test']; q_pp = np.clip(z['q_test'], -0.1 * S.cap, 1.1 * S.cap)
    o = np.intersect1d(te['o'], o_pp); it = np.searchsorted(te['o'], o); ip = np.searchsorted(o_pp, o)
    Q = {'PARA-PV': q_pp[ip], 'Ours (no M3)': te['q']['Ours'][it][..., IDX]}
    for k in ['Ours'] + REFS:
        if k in te['q']: Q[k] = T.m3(cal, te, k)[it][..., IDX]
    y = te['y'][it]; cap = S.cap; day = te['Ceff'][it] > 0.05 * cap
    Rf, rf = reference(S, st); fi = o[:, None] + np.arange(1, H15 + 1)[None]; R = Rf[fi]; restr = rf[fi]; P = S.P[fi]
    rows, pw = [], {}
    for k, q in Q.items():
        med = q[..., 2]; e = y - med; ee = y[..., None] - q; pin = np.maximum(L5 * ee, (L5 - 1) * ee).mean(-1)
        pen = penalty(R, med, restr, cap, RULES['2026rev'])
        rows.append(dict(station=st, proto=proto, regime=REGIME.get(proto, 'main'), system=k, n=len(o), mae=float(np.abs(e).mean()), pinball5=float(pin.mean()),
                         cov80=float(((y >= q[..., 1]) & (y <= q[..., 3])).mean()), cov90=float(((y >= q[..., 0]) & (y <= q[..., 4])).mean()),
                         width90=float((q[..., 4] - q[..., 0]).mean()), acc_day=float(1 - np.sqrt((e[day] ** 2).mean()) / cap),
                         pen=float(pen.sum()), energy=float(np.clip(P, 0, None).sum() * 0.25)))
        pw[k] = (np.abs(e).mean(-1), pin.mean(-1), pen.mean(-1))
    return dict(rows=rows, pw=pw, day=o // 96, proto=proto)


def boot(res, a, b, k, B=2000):
    rng = np.random.default_rng(0); tot = np.zeros(B); num = []; den = []
    for r in res:
        if a not in r['pw'] or b not in r['pw']: continue
        da = r['pw'][a][k] - r['pw'][b][k]; ud, inv = np.unique(r['day'], return_inverse=True)
        s = np.bincount(inv, da); c = np.bincount(inv); pick = rng.integers(0, len(ud), (B, len(ud)))
        tot += s[pick].sum(1) / c[pick].sum(1); num.append(da.mean()); den.append(r['pw'][b][k].mean())
    if not num: return None
    tot /= len(num); base = np.mean(den)
    return 100 * np.mean(num) / base, 100 * np.quantile(tot, .025) / base, 100 * np.quantile(tot, .975) / base, len(num)


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    with Pool(12) as p:
        res = [r for r in p.map(one, [(st, pr) for st in STATIONS for pr in ['main'] + list(FOLDS)]) if r]
    D = pd.DataFrame(sum([r['rows'] for r in res], [])); D.to_csv(out / 'rows.csv', index=False)
    pd.set_option('display.width', 250)
    roll = D[D.proto != 'main']; print('station-folds:', roll[['station', 'proto']].drop_duplicates().shape[0])
    g = roll.groupby('system')[['mae', 'pinball5', 'cov80', 'cov90', 'width90', 'acc_day']].mean()
    pe = roll.groupby('system')[['pen', 'energy']].sum(); g['penalty_pct'] = 100 * pe.pen / pe.energy
    print(g.sort_values('mae').round(4).to_string()); g.to_csv(out / 'table_roll.csv')
    print(roll.groupby(['system', 'regime']).mae.mean().unstack().round(4).to_string())
    ci = []
    for reg in ('ALL_ROLL', 'free', 'transition', 'dispatch', 'main'):
        rr = [r for r in res if (reg == 'ALL_ROLL' and r['proto'] != 'main') or REGIME.get(r['proto'], 'main') == reg]
        for a in ('Ours', 'Ours (no M3)'):
            for k, met in ((0, 'mae'), (1, 'pinball5'), (2, 'penalty')):
                x = boot(rr, a, 'PARA-PV', k)
                if x: ci.append(dict(regime=reg, system=a, metric=met, rel_pct=x[0], lo=x[1], hi=x[2], n=x[3], sig=bool(x[2] < 0 or x[1] > 0)))
    C = pd.DataFrame(ci); C.to_csv(out / 'ci.csv', index=False)
    C['cell'] = [f"{p:+.1f}%{'*' if s else ''}" for p, s in zip(C.rel_pct, C.sig)]
    print('vs PARA-PV (negative: ours better)'); print(C.pivot_table(index=['system', 'metric'], columns='regime', values='cell', aggfunc='first').to_string())
