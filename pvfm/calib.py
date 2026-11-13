"""Module 3 research: median-locked calibrators (static / Mondrian / online quantile tracking) in P- or K-space.

Protocol per station: fit static offsets on validation 2nd half; choose online step size gamma on validation
(online pass over the full validation period); then run on test in time order. Online updates for window i use only
labels of windows < i (labels of a window are complete before the next non-overlapping window is issued)."""
import json, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from pvfm.common import LEVELS, score
from pvfm.preds import load, MID

TAU = LEVELS
NONMID = [j for j in range(len(TAU)) if j != MID]


def fix(q):
    lower = np.minimum.accumulate(q[..., :MID + 1][..., ::-1], -1)[..., ::-1]
    upper = np.maximum.accumulate(q[..., MID:], -1)
    return np.concatenate([lower[..., :MID], q[..., MID:MID + 1], upper[..., 1:]], -1)


def groups(d, mode):
    n, H = d['y'].shape
    if mode == 'glob': return np.zeros((n, H), int)
    if mode == 'reg': return d['reg'].astype(int)
    if mode == 'regh':
        hb = np.minimum(np.arange(H) * 4 // H, 3)[None].repeat(n, 0)
        return d['reg'].astype(int) * 4 + hb
    raise ValueError(mode)


def resid(d, q, space):
    r = d['y'][..., None] - q
    return r / d['Ceff'][..., None] if space == 'K' else r


def apply(d, q, theta, g, space):
    sc = d['Ceff'][..., None] if space == 'K' else 1.0
    adj = np.zeros_like(q)
    for j in NONMID:
        adj[..., j] = theta[g, j]
    return fix(q + adj * sc)


def fit_static(d, q, g, space, G):
    r = resid(d, q, space); th = np.zeros((G, len(TAU)))
    for k in range(G):
        m = g == k
        if m.sum() < 30: continue
        for j in NONMID:
            th[k, j] = np.quantile(r[m][:, j], TAU[j])
    return th


def online(d, q, g, space, G, theta0, gamma):
    """Quantile tracking with feedback delay d['delay'] (windows): window i uses labels of windows <= i - delay."""
    th = theta0.copy(); out = np.empty_like(q); delay = d.get('delay', 1)
    r = resid(d, q, space)
    for i in range(len(q)):
        j = i - delay
        if j >= 0:
            thj = hist_th[j]
            for k in np.unique(g[j]):
                m = g[j] == k
                hit = (r[j][m][:, NONMID] <= thj[k, NONMID]).mean(0)
                th[k, NONMID] += gamma * (TAU[NONMID] - hit)
        if i == 0: hist_th = np.empty((len(q),) + th.shape)
        hist_th[i] = th
        out[i] = apply({'Ceff': d['Ceff'][i:i + 1]}, q[i:i + 1], th, g[i:i + 1], space)[0]
    return out


def online_hier(d, q, reg, space, theta_g0, theta_r0, gamma, rho):
    """Hierarchical tracking (global + regime deviation, shrinkage via smaller step) with feedback delay."""
    tg, tr = theta_g0.copy(), theta_r0.copy(); out = np.empty_like(q); r = resid(d, q, space); delay = d.get('delay', 1)
    hist = []
    for i in range(len(q)):
        j = i - delay
        if j >= 0:
            thj = hist[j]; rj = r[j]
            hit_all = (rj[:, NONMID] <= thj[reg[j]][:, NONMID]).mean(0)
            tg[NONMID] += gamma * (TAU[NONMID] - hit_all)
            for k in np.unique(reg[j]):
                m = reg[j] == k
                hit = (rj[m][:, NONMID] <= thj[k, NONMID]).mean(0)
                tr[k, NONMID] += rho * gamma * (TAU[NONMID] - hit) * m.mean()
        th = tr + tg[None]; hist.append(th.copy())
        out[i] = apply({'Ceff': d['Ceff'][i:i + 1]}, q[i:i + 1], th, reg[i:i + 1], space)[0]
    return out


def metrics(d, q):
    s = score(d['y'], q, TAU, d['cap'])
    cov = [(d['y'] <= q[..., j]).mean() for j in range(len(TAU))]
    s['qce'] = float(np.mean([abs(cov[j] - TAU[j]) for j in NONMID]))
    w = []
    for k in (0, 1, 2):
        m = d['reg'] == k
        if m.sum(): w.append(abs(((d['y'][m] >= q[m][:, 2]) & (d['y'][m] <= q[m][:, 10])).mean() - 0.8))
    s['worst_reg_ce80'] = float(max(w))
    return s


EXPERTS = {
    '1min': {'fm': 'fm:Kc_c8192_pastfut', 'fm2880': 'fm:K_c2880_pastfut', 'ad': 'ad:stageB2/Kc_c8192/pooled:pooled_fmgeo'},
}


def run_station(args):
    st, task, name = args
    src = {name: EXPERTS[task][name]}
    va = load(st, src, 'validation', task); te = load(st, src, 'test', task)
    n = len(va['o']); h = n // 2
    cal = {k: (v[h:] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v) for k, v in va.items()}
    cal['q'] = {name: va['q'][name][h:]}
    rows = []
    for space in ('P', 'K'):
        for mode in ('glob', 'reg', 'regh'):
            G = {'glob': 1, 'reg': 3, 'regh': 12}[mode]
            gc, gt, gv = groups(cal, mode), groups(te, mode), groups(va, mode)
            th = fit_static(cal, cal['q'][name], gc, space, G)
            qs = apply(te, te['q'][name], th, gt, space)
            rows.append(dict(station=st, expert=name, method=f'static_{space}_{mode}', **metrics(te, qs)))
            # online: choose gamma on validation (online pass from zero offsets), then run on test from static init
            best = None
            for gam in ((0.0005, 0.001, 0.002, 0.005, 0.01) if space == 'K' else (0.005, 0.01, 0.02, 0.05, 0.1)):
                qv = online(va, va['q'][name], gv, space, G, np.zeros((G, len(TAU))), gam)
                pv = metrics(va, qv)['pinball']
                if best is None or pv < best[0]: best = (pv, gam)
            qo = online(te, te['q'][name], gt, space, G, th, best[1])
            rows.append(dict(station=st, expert=name, method=f'online_{space}_{mode}', gamma=best[1], **metrics(te, qo)))
    # hierarchical online (K space): init global from static_glob, regime deviations from static_reg - glob
    gc, gt, gv = groups(cal, 'reg'), groups(te, 'reg'), groups(va, 'reg')
    thg = fit_static(cal, cal['q'][name], groups(cal, 'glob'), 'K', 1)[0]
    thr = fit_static(cal, cal['q'][name], gc, 'K', 3) - thg[None]
    best = None
    for gam in (0.0005, 0.001, 0.002, 0.005):
        for rho in (0.25, 1.0, 4.0):
            qv = online_hier(va, va['q'][name], gv, 'K', np.zeros(len(TAU)), np.zeros((3, len(TAU))), gam, rho)
            m = metrics(va, qv); obj = m['qce'] + m['worst_reg_ce80']
            if best is None or obj < best[0]: best = (obj, gam, rho)
    qh = online_hier(te, te['q'][name], gt, 'K', thg, thr, best[1], best[2])
    rows.append(dict(station=st, expert=name, method='online_K_hier', gamma=best[1], rho=best[2], **metrics(te, qh)))
    rows.append(dict(station=st, expert=name, method='none', **metrics(te, te['q'][name])))
    return rows


if __name__ == '__main__':
    task, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:  # name=src,name=src
        EXPERTS[task] = dict(kv.split('=', 1) for kv in sys.argv[3].split(','))
    from pvfm.common import STATIONS
    jobs = [(st, task, e) for e in EXPERTS[task] for st in STATIONS]
    with Pool(len(jobs)) as p:
        rows = sum(p.map(run_station, jobs), [])
    R = pd.DataFrame(rows); R.to_csv(out, index=False)
    cols = ['mae', 'pinball', 'crps_trap', 'qce', 'cov40', 'cov80', 'cov90', 'width80', 'worst_reg_ce80']
    pd.set_option('display.width', 220)
    print(R.groupby(['expert', 'method'])[cols].mean().round(4).to_string())


def online_hier_loc(d, q, reg, space, theta_g0, theta_r0, gamma, rho, gamma_loc):
    """M3 extended to online drift correction: as online_hier for non-median levels, plus a location offset that is
    added to ALL levels (including the median) and tracked as the 0.5-quantile of K-space residuals with step gamma_loc
    (global + regime deviation). Same feedback delay; preserves monotonicity (common shift)."""
    tg, tr = theta_g0.copy(), theta_r0.copy(); lg = 0.0; lr = np.zeros(tr.shape[0])
    out = np.empty_like(q); r = resid(d, q, space); delay = d.get('delay', 1); hist = []
    for i in range(len(q)):
        j = i - delay
        if j >= 0:
            thj, lgj, lrj = hist[j]; rj = r[j]
            loc_j = lgj + lrj[reg[j]]  # [H]
            rr = rj - loc_j[:, None]  # residual after location shift
            hit_all = (rr[:, NONMID] <= thj[reg[j]][:, NONMID]).mean(0)
            tg[NONMID] += gamma * (TAU[NONMID] - hit_all)
            hit_med = (rr[:, MID] <= 0).mean(); lg += gamma_loc * (0.5 - hit_med)
            for k in np.unique(reg[j]):
                m = reg[j] == k
                hit = (rr[m][:, NONMID] <= thj[k, NONMID]).mean(0)
                tr[k, NONMID] += rho * gamma * (TAU[NONMID] - hit) * m.mean()
                lr[k] += rho * gamma_loc * (0.5 - (rr[m][:, MID] <= 0).mean()) * m.mean()
        th = tr + tg[None]; hist.append((th.copy(), lg, lr.copy()))
        loc = lg + lr[reg[i]]  # [H]
        sc = d['Ceff'][i][:, None] if space == 'K' else 1.0
        base = apply({'Ceff': d['Ceff'][i:i + 1]}, q[i:i + 1], th, reg[i:i + 1], space)[0]
        out[i] = base + loc[:, None] * sc
    return out
