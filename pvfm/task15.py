"""15-min resolution, 0-4 h (16-step) task, issued every 15 min (grid-code style ultra-short-term).
Split boundaries are the same calendar boundaries as the 1-min protocol."""
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station
from pvfm.fm_zero import ffill, COVS

R = 15
H15 = 16
CTX_MAX = 672


class Station15:
    def __init__(self, path, name, cap):
        S = Station(path, name, cap)
        self.name, self.cap, self.base = name, cap, S
        n = S.n // R
        agg = lambda a: np.nanmean(a[:n * R].reshape(n, R), 1).astype(np.float32)
        self.P = agg(np.nan_to_num(S.x['P']))
        self.x = {k: agg(ffill(S.x[k])) for k in COVS if k in S.x}
        self.cz = agg(S.cz)
        self.C = agg(S.C)
        self.floor = S.floor
        self.Ceff = np.maximum(self.C, self.floor).astype(np.float32)
        self.K = (self.P / self.Ceff).astype(np.float32)
        self.minutes = S.minutes[:n * R].reshape(n, R)[:, -1]  # interval end
        # calendar split boundaries from the 1-min protocol (index of 15-min interval containing the boundary)
        v0 = S.origins['validation'].min() // R
        t0 = S.origins['test'].min() // R
        t1 = (S.origins['test'].max() + 96) // R
        idx = np.arange(CTX_MAX - 1, n - H15)
        self.origins = {'train': idx[idx + H15 < v0], 'validation': idx[(idx >= v0) & (idx + H15 < t0)],
                        'test': idx[(idx >= t0) & (idx + H15 <= min(t1, n - 1))]}

    def windows(self, split, ctx):
        o = self.origins[split]
        return o, o[:, None] + np.arange(-ctx + 1, 1)[None], o[:, None] + np.arange(1, H15 + 1)[None]

    def smart_persistence(self, o):
        f = o[:, None] + np.arange(1, H15 + 1)[None]
        return self.P[o][:, None] * self.Ceff[f] / self.Ceff[o][:, None]


def run15(spec, station, data_dir, out_dir, pipe, splits=('train', 'validation', 'test')):
    space, c, cov = spec.split('_'); ctx = int(c[1:])
    st = Station15(Path(data_dir) / f'{station}.npz', station, STATIONS[station][1])
    if space == 'Ka':  # available-power clear-sky index (target of the grid-code assessment in restricted periods)
        av = np.nan_to_num(st.x['avail']); tgt = np.clip(av / st.Ceff, -0.5, 1.5)
    else:
        tgt = np.clip(st.K, -0.5, 1.5) if space == 'Kc' else st.P
    out = {}
    import time
    for split in splits:
        o, ci, fi = st.windows(split, ctx)
        inputs = []
        for j in range(len(o)):
            d = {'target': tgt[ci[j]]}
            if cov in ('past', 'pastfut'):
                d['past_covariates'] = {k: v[ci[j]] for k, v in st.x.items()}
            if cov == 'pastfut':
                d['past_covariates'].update(cz=st.cz[ci[j]], cs=st.Ceff[ci[j]])
                d['future_covariates'] = {'cz': st.cz[fi[j]], 'cs': st.Ceff[fi[j]]}
            inputs.append(d)
        t = time.time()
        q, _ = pipe.predict_quantiles(inputs, prediction_length=H15, quantile_levels=LEVELS.tolist(), batch_size=256)
        el = time.time() - t
        q = np.stack([x[0].float().cpu().numpy() for x in q])
        if space in ('Kc', 'Ka'):
            q = q * st.Ceff[fi][..., None]
        q = np.maximum.accumulate(q, axis=-1).astype(np.float32)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(out_dir) / f'{split}.npz', q=q, origins=o, seconds=el)
        out[split] = dict(n=len(o), seconds=round(el, 2))
    return out


def splits15(S15, proto):
    """15-min origins per protocol part, using the 1-min fold calendar. Origin i (interval index) is issued at the end
    of interval i (minute 15(i+1)-1 from S.t0); its targets end at minute 15(i+17)-1."""
    from pvfm.protocol import FOLDS, _minute
    if proto == 'main':
        va = np.sort(S15.origins['validation']); h = len(va) // 2
        return dict(train=np.sort(S15.origins['train']), es=va[:h], cal=va[h:], test=np.sort(S15.origins['test']))
    t0, t1 = (_minute(x) - S15.base.t0 for x in FOLDS[proto])
    o = np.arange(CTX_MAX - 1, len(S15.P) - H15); m0 = 15 * (o + 1) - 1; m1 = 15 * (o + H15 + 1) - 1
    d14 = 14 * 1440
    return dict(train=o[m1 < t0 - 2 * d14], es=o[(m0 >= t0 - 2 * d14) & (m1 < t0 - d14)],
                cal=o[(m0 >= t0 - d14) & (m1 < t0)], test=o[(m0 >= t0) & (m1 < t1)])


def fm15_all(variant, st):
    """union of the train/validation/test caches of a 15-min zero-shot run (they tile all origins)."""
    from pvfm.preds import ROOT
    os_, qs = [], []
    for sp in ('train', 'validation', 'test'):
        f = ROOT / 'runs/t15A' / variant / st / f'{sp}.npz'
        if f.exists():
            z = np.load(f); os_.append(z['origins']); qs.append(z['q'])
    o = np.concatenate(os_); q = np.concatenate(qs); order = np.argsort(o)
    return o[order], q[order]
