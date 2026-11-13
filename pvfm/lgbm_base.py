"""Fair strong baseline: LightGBM quantile regression in clear-sky-index space with the SAME information as the FM
(recent history, same-time values of the previous 5 days ~ the 8192-min context, future geometry, dispatch state).
Loss weighted by C_eff(t+h) (Proposition 1: equals power-space pinball). Outputs P-space quantiles in the same
format as fm_zero so calibration / gating code can use it as an expert."""
from pvfm.common import PVROOT
import json, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station
from pvfm.fm_zero import ffill

ROOT = Path(PVROOT)
H = 96
HSUB = np.arange(0, H, 3)  # training horizons (every 3rd); prediction uses all 96


def station_feats(S):
    cap = S.cap
    K = np.clip(S.K, -0.5, 1.5)
    ghi = np.clip(ffill(S.x['ghi']), 0, None)
    gcs = np.where(S.cz > 0.05, 1098 * np.clip(S.cz, 0.05, None) * np.exp(-0.057 / np.clip(S.cz, 0.05, None)), np.nan)
    kt = np.nan_to_num(np.clip(ghi / gcs, 0, 1.5))
    cs = np.cumsum(np.concatenate([[0], K])); cs2 = np.cumsum(np.concatenate([[0], K ** 2]))
    mean = lambda o, w: (cs[o + 1] - cs[o + 1 - w]) / w
    std = lambda o, w: np.sqrt(np.maximum((cs2[o + 1] - cs2[o + 1 - w]) / w - mean(o, w) ** 2, 0))
    avail, agc = ffill(S.x['avail']), ffill(S.x['agc'])
    return dict(K=K, kt=kt, mean=mean, std=std, avail=avail, agc=agc, temp=ffill(S.x['temp']), wind=ffill(S.x['wind']))


def design(S, F, o, hs):
    """rows for every (window, horizon in hs). returns X [n*len(hs), d], target K, weight Ceff, fi index."""
    cap = S.cap
    oo = np.repeat(o, len(hs)); hh = np.tile(hs, len(o)); t = oo + hh + 1
    tod = (S.minutes[t] % 1440) / 1440
    lag_same = np.stack([F['K'][t - 1440 * k] for k in range(1, 6)], -1)
    X = np.column_stack([
        F['K'][oo], F['mean'](oo, 15), F['mean'](oo, 60), F['mean'](oo, 180), F['std'](oo, 60), F['kt'][oo],
        lag_same, np.nanmean(lag_same, -1), np.nanstd(lag_same, -1),
        S.cz[oo], S.cz[t], S.Ceff[t] / cap, (hh + 1) / H, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        (F['agc'][oo] < 0.97 * F['avail'][oo]).astype(float), F['agc'][oo] / cap, np.clip(F['avail'][oo] / S.Ceff[oo], 0, 2),
        F['temp'][oo] / 40, F['wind'][oo] / 10]).astype(np.float32)
    return X, F['K'][t], S.Ceff[t], t


def main(job, out, state):
    scope = job['scope']; names = list(STATIONS) if scope == 'ALL' else [scope]
    if job.get('proto') and job['proto'] != 'main':
        from pvfm.protocol import available
        names = [s for s in names if available(Station(ROOT / 'data' / f'{s}.npz', s, STATIONS[s][1]), job['proto'])]
    data = {}
    for s in names:
        S = Station(ROOT / 'data' / f'{s}.npz', s, STATIONS[s][1]); F = station_feats(S)
        if job.get('proto'):
            from pvfm.protocol import splits
            o = {k: np.sort(v) for k, v in splits(S, job['proto']).items()}
        else:
            o = {sp: np.sort(S.origins[sp]) for sp in ('train', 'validation', 'test')}
            nv = len(o['validation']); o['es'] = o['validation'][:nv // 2]
        o = {k: v[v >= 1440 * 5 + 200] for k, v in o.items()}
        data[s] = (S, F, o)
    def stack(split, hs):
        Xs, ys, ws = [], [], []
        for i, s in enumerate(names):
            S, F, o = data[s]; X, y, w, _ = design(S, F, o[split], hs)
            if scope == 'ALL': X = np.column_stack([X, np.full(len(X), i, np.float32)])
            Xs.append(X); ys.append(y); ws.append(w)
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    Xtr, ytr, wtr = stack('train', HSUB); Xes, yes, wes = stack('es', HSUB)
    t = time.time(); models = []
    for tau in LEVELS:
        m = lgb.LGBMRegressor(objective='quantile', alpha=float(tau), n_estimators=2000, learning_rate=0.05, num_leaves=63,
                              min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=job.get('threads', 24), verbose=-1)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xes, yes)], eval_sample_weight=[wes],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
    fit_s = time.time() - t
    res = {}
    if job.get('proto'):
        for i, s in enumerate(names):
            S, F, o = data[s]; outd = {}
            for part in ('es', 'cal', 'test'):
                X, _, Ce, fi = design(S, F, o[part], np.arange(H))
                if scope == 'ALL': X = np.column_stack([X, np.full(len(X), i, np.float32)])
                q = np.stack([m.predict(X) for m in models], -1).reshape(len(o[part]), H, len(LEVELS))
                outd[f'o_{part}'] = o[part]; outd[f'q_{part}'] = (np.sort(q, -1) * S.Ceff[fi].reshape(len(o[part]), H)[..., None]).astype(np.float32)
            od = ROOT / 'runs/rbase' / job['tag'] / job['proto']; od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f'{s}.npz', **outd); res[s] = len(o['test'])
        return dict(fit_seconds=fit_s, best_iters=[int(m.best_iteration_ or 0) for m in models], n_train_rows=int(len(Xtr)), predicted=res)
    for i, s in enumerate(names):
        S, F, o = data[s]
        od = ROOT / 'runs/stageA' / job['tag'] / s; od.mkdir(parents=True, exist_ok=True)
        for split in ('validation', 'test'):
            X, _, Ce, fi = design(S, F, o[split], np.arange(H))
            if scope == 'ALL': X = np.column_stack([X, np.full(len(X), i, np.float32)])
            q = np.stack([m.predict(X) for m in models], -1).reshape(len(o[split]), H, len(LEVELS))
            q = np.sort(q, -1) * S.Ceff[fi].reshape(len(o[split]), H)[..., None]
            np.savez_compressed(od / f'{split}.npz', q=q.astype(np.float32), origins=o[split], seconds=0.0)
        res[s] = len(o['test'])
    return dict(fit_seconds=fit_s, best_iters=[int(m.best_iteration_ or 0) for m in models], n_train_rows=int(len(Xtr)), predicted=res)
