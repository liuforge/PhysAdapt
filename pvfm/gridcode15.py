"""Reviewer MC3: ultra-short-term assessment of the Shandong grid code (2026 revision, Sec. 4 'ultra-short-term forecast'):
  penalty energy = int alpha * (|P - PP| - dP)^+ dt,
  PP forecast; dP = allowed deviation = frac * P, min 1 MW, frac = 10% (unrestricted) / 13% (restricted) for PV;
  P = actual power, or available power PA when the plan differs from the forecast and |P - PS| <= max(6% PS, 1 MW);
  alpha by relative deviation |P - PP| / max(P, 5% cap): 0 | 0.03 (<=20%) | 0.07 (<=30%) | 0.1 (<=50%) | 0.3 (<=70%) | 0.7 (<=90%) | 0.8;
  monthly penalty capped at 7% of on-grid energy.
Proxies: restricted = curtailment-bound interval (bind15); plan PS = AGC command; PA = reported available power, used only at
the four stations whose available-power channel is valid (elsewhere the reference is P). Submitted PP = calibrated median.
Leads: 1 (first point), 16 (4th hour), all 16 points. Variant 'rev-thr' uses 15%/18% allowed deviation (thresholds quoted by
the reviewer for the Dec-2025 version; same alpha tiers). Ka = M2 trained on the available-power clear-sky index (4-station pool),
calibrated online against the rule reference."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PVFM_NO_TORCH"] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from pvfm import t15_full as T
from pvfm.common import STATIONS
from pvfm.protocol import FOLDS, REGIME
from pvfm.preds import ROOT, MID
from pvfm.task15 import Station15, splits15, H15
from pvfm.radapt15 import usable

GOOD = ('feicheng_pv', 'heze_pv', 'huimin_pv', 'yiyuan_pv')
T.FOLD.update({'Ours (4-station pool)': 'runs/radapt15/Kc_c672_pastfut_good4/pooled_fmgeo', 'Ka (4-station pool)': 'runs/radapt15/Ka_c672_pastfut/pooled_fmgeo'})
TIERS = np.array([0.2, 0.3, 0.5, 0.7, 0.9]); ALPHA = np.array([0.03, 0.07, 0.1, 0.3, 0.7, 0.8])
RULES = {'2026rev': (0.10, 0.13), 'rev-thr': (0.15, 0.18)}


def reference(S, st):
    cap = S.cap; P = S.P; restr = T.bind15(S)
    if st not in GOOD: return P.copy(), restr
    agc = np.nan_to_num(S.x['agc'], nan=cap); PA = np.nan_to_num(S.x['avail'])
    follow = restr & (np.abs(P - agc) <= np.maximum(0.06 * agc, 1.0))
    return np.where(follow, np.maximum(PA, P), P).astype(np.float32), restr


def penalty(R, PP, restr, cap, fr):
    dev = np.abs(R - PP); dP = np.maximum(np.where(restr, fr[1], fr[0]) * np.clip(R, 0, None), 1.0)
    rel = dev / np.maximum(R, 0.05 * cap); a = ALPHA[np.searchsorted(TIERS, rel, side='left')]
    return np.where(dev > dP, a * (dev - dP), 0.0) * 0.25  # MWh per 15-min interval


def one(args):
    st, proto = args
    if not (ROOT / T.FOLD['Ours'] / proto / f'{st}.npz').exists() or not usable(st, proto): return None
    S = Station15(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1]); Rf, restr_f = reference(S, st)
    tab = T.psp_fit(S, splits15(S, proto)['train'])
    cal = T.load(S, st, proto, 'cal', tab); te = T.load(S, st, proto, 'test', tab)
    med = {}
    for k in te['q']:
        if k.startswith('Ka'):
            c2, t2 = dict(cal), dict(te)
            for d, d0 in ((c2, cal), (t2, te)): d['y'] = Rf[d0['o'][:, None] + np.arange(1, H15 + 1)[None]]
            med[k] = T.m3(c2, t2, k)[..., MID]
        else: med[k] = T.m3(cal, te, k)[..., MID]
    med['Smart persistence'] = te['sp']
    fi = te['o'][:, None] + np.arange(1, H15 + 1)[None]; R = Rf[fi]; restr = restr_f[fi]; P = S.P[fi]
    month = pd.to_datetime(S.minutes[fi[:, 0]], unit='m').to_period('M').astype(str).values
    rows, pw = [], {}
    for k, pp in med.items():
        for rn, fr in RULES.items():
            pen = penalty(R, pp, restr, S.cap, fr)
            for ln, sl in (('h1', slice(0, 1)), ('h16', slice(15, 16)), ('all', slice(0, 16))):
                pe = pen[:, sl].mean(1); en = np.clip(P[:, sl], 0, None).mean(1) * 0.25
                for mo in np.unique(month):
                    w = month == mo
                    rows.append(dict(station=st, proto=proto, regime=REGIME.get(proto, 'main'), system=k, rule=rn, lead=ln, month=mo,
                                     n=int(w.sum()), pen=float(pe[w].sum()), energy=float(en[w].sum()), restr_frac=float(restr[w][:, sl].mean()),
                                     pa_used=float((R[w][:, sl] > P[w][:, sl] + 1e-6).mean())))
                if rn == '2026rev': pw[(k, ln)] = pe
    return dict(rows=rows, pw=pw, day=te['o'] // 96, proto=proto, st=st)


def boot(res, a, b, B=2000):
    rng = np.random.default_rng(0); tot = np.zeros(B); num = []; den = []
    for r in res:
        if a not in r['pw'] or b not in r['pw']: continue
        da = r['pw'][a] - r['pw'][b]; ud, inv = np.unique(r['day'], return_inverse=True)
        s = np.bincount(inv, da); pick = rng.integers(0, len(ud), (B, len(ud))); tot += s[pick].sum(1); num.append(da.sum()); den.append(r['pw'][b].sum())
    if not num: return None
    return 100 * sum(num) / sum(den), 100 * np.quantile(tot, .025) / sum(den), 100 * np.quantile(tot, .975) / sum(den)


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    with Pool(12) as p:
        res = [r for r in p.map(one, [(st, pr) for st in STATIONS for pr in ['main'] + list(FOLDS)]) if r]
    D = pd.DataFrame(sum([r['rows'] for r in res], [])); D.to_csv(out / 'rows.csv', index=False)
    D['pen_capped'] = np.minimum(D.pen, 0.07 * D.energy)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    roll = D[D.proto != 'main']
    for sub, nm in ((roll, 'all 7 stations'), (roll[roll.station.isin(GOOD)], '4 stations with valid available power')):
        g = sub.groupby(['rule', 'lead', 'system'])[['pen', 'pen_capped', 'energy']].sum()
        g['pen_pct'] = 100 * g.pen / g.energy; g['capped_pct'] = 100 * g.pen_capped / g.energy
        print(f'== rolling folds, {nm}: penalty energy as % of on-grid energy'); print(g[['pen_pct', 'capped_pct']].unstack(['rule', 'lead']).round(3).to_string())
    print('restricted fraction / PA-reference fraction (4 stations):', roll[roll.station.isin(GOOD) & (roll.system == 'Ours')].groupby('lead')[['restr_frac', 'pa_used']].mean().round(3).to_string())
    ci = []
    for grp, sts in (('all7', set(STATIONS)), ('good4', set(GOOD))):
        rr = [r for r in res if r['proto'] != 'main' and r['st'] in sts]
        for ln in ('h1', 'h16', 'all'):
            for b in sorted({k[0] for r in rr for k in r['pw']} - {'Ours'}):
                x = boot(rr, ('Ours', ln), (b, ln))
                if x: ci.append(dict(stations=grp, lead=ln, ref=b, rel_pct=x[0], lo=x[1], hi=x[2], sig=bool(x[2] < 0 or x[1] > 0)))
            if grp == 'good4':
                x = boot(rr, ('Ka (4-station pool)', ln), ('Ours (4-station pool)', ln))
                if x: ci.append(dict(stations=grp, lead=ln, ref='Ka vs Kc (4-station pool)', rel_pct=x[0], lo=x[1], hi=x[2], sig=bool(x[2] < 0 or x[1] > 0)))
    C = pd.DataFrame(ci); C.to_csv(out / 'ci.csv', index=False)
    C['cell'] = [f"{p:+.1f}%{'*' if s else ''}" for p, s in zip(C.rel_pct, C.sig)]
    print('Ours vs ref, penalty energy (2026 rev), negative = Ours lower'); print(C.pivot_table(index='ref', columns=['stations', 'lead'], values='cell', aggfunc='first').to_string())
