"""Stage A: zero-shot Chronos-2 variants. spec = {space}_c{ctx}_{cov}; space in {P,K}; cov in {none,past,pastfut}."""
import json, time
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station

COVS = ['avail', 'agc', 'ghi', 'dni', 'dhi', 'poa', 'temp', 'wind', 'pressure', 'tpanel']


def ffill(a):
    a = a.copy(); m = np.isnan(a)
    if m.all(): return np.zeros_like(a)
    idx = np.where(~m, np.arange(len(a)), 0); np.maximum.accumulate(idx, out=idx); a = a[idx]
    a[np.isnan(a)] = 0; return a


def run(spec, station, data_dir, out_dir, pipe, splits=('validation', 'test'), H=96, cov_subset=None, origins_map=None):
    space, c, cov = spec.split('_'); ctx = int(c[1:])
    st = Station(Path(data_dir) / f'{station}.npz', station, STATIONS[station][1])
    if space == 'Kt':  # measured-irradiance clear-sky index (weather; unaffected by curtailment)
        g = np.clip(ffill(st.x['ghi']), 0, None)
        gcs = np.where(st.cz > 0.05, 1098 * np.clip(st.cz, 0.05, None) * np.exp(-0.057 / np.clip(st.cz, 0.05, None)), 0)
        tgt = np.where(gcs > 0, np.clip(g / np.maximum(gcs, 1), 0, 1.5), 0).astype(np.float32)
    elif space == 'Agc':  # dispatch set-point as fraction of capacity (NaN -> capacity, i.e. no limit)
        a = st.x['agc']; tgt = np.clip(np.where(np.isfinite(a), a, st.cap) / st.cap, 0, 1.2).astype(np.float32)
    elif space == 'Kpc':  # potential (uncurtailed) clear-sky index
        from pvfm.physics import potential
        Pp, _, _ = potential(st)
        tgt = np.clip(Pp / st.Ceff, -0.5, 1.5)
    else:
        tgt = {'K': st.K, 'Kc': np.clip(st.K, -0.5, 1.5)}.get(space, None)
    if tgt is None: tgt = np.nan_to_num(st.x['P'])
    covs = {k: ffill(st.x[k]) for k in COVS if k in st.x and (cov_subset is None or k in cov_subset)}
    if cov.endswith('nd'):  # 'no dispatch': drop AGC set-point and available power from the past covariates
        covs = {k: v for k, v in covs.items() if k not in ('agc', 'avail')}; cov = cov[:-2]
    if cov.endswith('sp'):  # spatial: clipped clear-sky index of the other stations, aligned on absolute minute (past only)
        for other, (_, ocap) in STATIONS.items():
            if other == station: continue
            So = Station(Path(data_dir) / f'{other}.npz', other, ocap)
            a = np.full(st.n, np.nan, np.float32)
            lo = max(st.t0, So.t0); hi = min(st.t0 + st.n, So.t0 + So.n)
            if hi > lo: a[lo - st.t0:hi - st.t0] = np.clip(So.K[lo - So.t0:hi - So.t0], -0.5, 1.5)
            covs[f'K_{other}'] = ffill(a)
        cov = cov[:-2]
    out = {}
    for split in splits:
        if split == 'all' or (origins_map is not None and split in origins_map):
            if origins_map is not None and split in origins_map:
                o = np.sort(origins_map[split])
            else:
                from pvfm.protocol import all_origins
                o = all_origins(st)
            o = o[o - ctx + 1 >= 0]
            ci = o[:, None] + np.arange(-ctx + 1, 1)[None]; fi = o[:, None] + np.arange(1, H + 1)[None]
        else:
            o, ci, fi = st.windows(split, ctx, H)
        inputs = []
        for j in range(len(o)):
            d = {'target': tgt[ci[j]]}
            if cov in ('past', 'pastfut'):
                d['past_covariates'] = {k: v[ci[j]] for k, v in covs.items()}
            if cov == 'pastfut':
                d.setdefault('past_covariates', {})
                d['past_covariates'].update(cz=st.cz[ci[j]], cs=st.Ceff[ci[j]])
                d['future_covariates'] = {'cz': st.cz[fi[j]], 'cs': st.Ceff[fi[j]]}
            inputs.append(d)
        bs = 256 if ctx <= 360 else (96 if ctx <= 1440 else (48 if ctx <= 2880 else 16))
        t = time.time()
        q, _ = pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=LEVELS.tolist(), batch_size=bs)
        el = time.time() - t
        q = np.stack([x[0].float().cpu().numpy() for x in q])  # (n, H, Q)
        if space in ('K', 'Kc', 'Kpc'):
            q = q * st.Ceff[fi][..., None]
        elif space == 'Agc':
            q = q * st.cap
        q = np.maximum.accumulate(q, axis=-1).astype(np.float32)
        np.savez_compressed(Path(out_dir) / f'{split}.npz', q=q, origins=o, seconds=el)
        out[split] = dict(n=len(o), seconds=round(el, 2))
    return out
