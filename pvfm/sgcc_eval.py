"""Public-data comparison (State Grid competition sites 1 and 6, PARA-PV protocol: 8:1:1 split, 15-min, H=16).
Run with PVFM_ROOT=/mnt/user/pvfm_sgcc PVFM_DATASET=sgcc. All methods on the same test windows; probabilistic methods with
their own M3 (as in the main paper), PARA-PV with its own five levels. Reports MW metrics, 13-level pinball/QCE, a
five-level comparison including PARA-PV, and MAE/MSE on PARA-PV's standardised scale (z-score with the training mean/std
of power, all 16 leads, day and night). Paired day-block bootstrap of Ours vs each method."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PVFM_NO_TORCH"] = "1"
import sys
from pathlib import Path
import numpy as np, pandas as pd
from pvfm import t15_full as T
from pvfm.common import STATIONS, LEVELS
from pvfm.preds import ROOT, MID
from pvfm.task15 import Station15, splits15, H15
from pvfm.calib import metrics as cmetrics

L5 = np.array([0.05, 0.1, 0.5, 0.9, 0.95]); IDX = [int(np.argmin(np.abs(LEVELS - t))) for t in L5]
PPTAG = os.environ.get('PARAPV_TAG', 'PARA_PV_paper')


def one(st):
    S = Station15(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1]); cap = S.cap
    tab = T.psp_fit(S, splits15(S, 'main')['train'])
    cal = T.load(S, st, 'main', 'cal', tab); te = T.load(S, st, 'main', 'test', tab)
    Q = {k: T.m3(cal, te, k) for k in te['q']}; Q['Ours (no M3)'] = te['q']['Ours']
    o = te['o']; y = te['y']; tr = splits15(S, 'main')['train']
    Ptr = S.P[: int(tr.max()) + 1]; mu, sd = float(Ptr.mean()), float(Ptr.std())
    f = ROOT / 'runs/rbase15' / PPTAG / 'main' / f'{st}.npz'; pp = None
    if f.exists():
        z = np.load(f); oo = np.intersect1d(o, z['o_test']); keep = np.isin(o, oo)
        pp = (keep, np.clip(z['q_test'][np.searchsorted(z['o_test'], oo)], -0.1 * cap, 1.1 * cap))
    rows, pw = [], {}; day = te['Ceff'] > 0.05 * cap
    meds = {k: q[..., MID] for k, q in Q.items()}; meds['Smart persistence'] = te['sp']
    for k, med in meds.items():
        e = y - med; q = Q.get(k)
        m = cmetrics(te, q) if q is not None else dict(mae=float(np.abs(e).mean()))
        m.update(acc_day=float(1 - np.sqrt((e[day] ** 2).mean()) / cap), mae_std=float(np.abs(e).mean() / sd), mse_std=float((e ** 2).mean() / sd ** 2))
        if q is not None:
            q5 = q[..., IDX]; ee = y[..., None] - q5; m['pinball5'] = float(np.maximum(L5 * ee, (L5 - 1) * ee).mean())
            m['picp80'] = float(((y >= q5[..., 1]) & (y <= q5[..., 3])).mean()); m['picp90'] = float(((y >= q5[..., 0]) & (y <= q5[..., 4])).mean())
            m['pinaw90'] = float((q5[..., 4] - q5[..., 0]).mean() / (S.P.max() - S.P.min()))
        rows.append(dict(station=st, system=k, n=len(o), **m)); ee = y[..., None] - (q if q is not None else med[..., None])
        lv = LEVELS if q is not None else np.array([0.5])
        pw[k] = (np.abs(e).mean(-1), np.maximum(lv * ee, (lv - 1) * ee).mean((-1, -2)))
    if pp is not None:
        keep, q5 = pp; yk = y[keep]; e = yk - q5[..., 2]; ee = yk[..., None] - q5
        rows.append(dict(station=st, system='PARA-PV', n=int(keep.sum()), mae=float(np.abs(e).mean()), acc_day=float(1 - np.sqrt((e[day[keep]] ** 2).mean()) / cap),
                         mae_std=float(np.abs(e).mean() / sd), mse_std=float((e ** 2).mean() / sd ** 2), pinball5=float(np.maximum(L5 * ee, (L5 - 1) * ee).mean()),
                         picp80=float(((yk >= q5[..., 1]) & (yk <= q5[..., 3])).mean()), picp90=float(((yk >= q5[..., 0]) & (yk <= q5[..., 4])).mean()),
                         pinaw90=float((q5[..., 4] - q5[..., 0]).mean() / (S.P.max() - S.P.min()))))
        qo = Q['Ours'][keep][..., IDX]; eo = yk[..., None] - qo
        pw['PARA-PV|5'] = (np.abs(e).mean(-1), np.maximum(L5 * ee, (L5 - 1) * ee).mean((-1, -2)))
        pw['Ours|5'] = (np.abs(yk - qo[..., 2]).mean(-1), np.maximum(L5 * eo, (L5 - 1) * eo).mean((-1, -2)))
        pw['_day5'] = o[keep] // 96
    return dict(rows=rows, pw=pw, day=o // 96, st=st)


def boot(res, a, b, k, dayk='day', B=2000):
    rng = np.random.default_rng(0); tot = np.zeros(B); num = []; den = []
    for r in res:
        if a not in r['pw'] or b not in r['pw']: continue
        dd = r['pw']['_day5'] if dayk == '_day5' else r['day']
        da = r['pw'][a][k] - r['pw'][b][k]; ud, inv = np.unique(dd, return_inverse=True)
        s = np.bincount(inv, da); c = np.bincount(inv); pick = rng.integers(0, len(ud), (B, len(ud)))
        tot += s[pick].sum(1) / c[pick].sum(1); num.append(da.mean()); den.append(r['pw'][b][k].mean())
    if not num: return None
    tot /= len(num); base = np.mean(den)
    return 100 * np.mean(num) / base, 100 * np.quantile(tot, .025) / base, 100 * np.quantile(tot, .975) / base


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    res = [one(st) for st in STATIONS]
    R = pd.DataFrame(sum([r['rows'] for r in res], [])); R.to_csv(out / 'rows.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    cols = [c for c in ['mae', 'mae_std', 'mse_std', 'pinball', 'pinball5', 'qce', 'picp80', 'picp90', 'pinaw90', 'acc_day'] if c in R]
    print(R.pivot_table(index='system', columns='station', values='mae').round(3).to_string())
    G = R.groupby('system')[cols].mean().sort_values('mae'); print(G.round(4).to_string()); G.to_csv(out / 'table.csv')
    ci = []
    for b in sorted({k for r in res for k in r['pw']} - {'Ours', 'PARA-PV|5', 'Ours|5', '_day5'}):
        for k, met in ((0, 'mae'), (1, 'pinball')):
            x = boot(res, 'Ours', b, k)
            if x: ci.append(dict(ref=b, metric=met, rel_pct=x[0], lo=x[1], hi=x[2], sig=bool(x[2] < 0 or x[1] > 0)))
    for k, met in ((0, 'mae'), (1, 'pinball5')):
        x = boot(res, 'Ours|5', 'PARA-PV|5', k, '_day5')
        if x: ci.append(dict(ref='PARA-PV', metric=met, rel_pct=x[0], lo=x[1], hi=x[2], sig=bool(x[2] < 0 or x[1] > 0)))
    C = pd.DataFrame(ci); C.to_csv(out / 'ci.csv', index=False)
    C['cell'] = [f"{p:+.1f}%{'*' if s else ''}" for p, s in zip(C.rel_pct, C.sig)]
    print('Ours vs ref (negative: Ours better)'); print(C.pivot_table(index='ref', columns='metric', values='cell', aggfunc='first').to_string())
