"""Probabilistic reference baselines (no learning beyond empirical quantiles), 1-min task:
- CHPeEn: complete-history persistence ensemble (Yang 2019 style): members = clear-sky index at the same time of day on
  each of the previous N days, scaled by C_eff(t+h).
- PSP: probabilistic smart persistence: median = smart persistence; quantiles from training-period K-space residual
  quantiles per (horizon, regime).
Outputs P-space [n,96,13] in runs/stageA/<name>/<station>/<split>.npz."""
from pvfm.common import PVROOT
import sys
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station
from pvfm.preds import regime

ROOT = Path(PVROOT); H = 96; NDAYS = 30


def chpeen(S, o):
    K = np.clip(S.K, -0.5, 1.5)
    fi = o[:, None] + np.arange(1, H + 1)[None]
    mem = np.stack([K[np.maximum(fi - 1440 * d, 0)] for d in range(1, NDAYS + 1)], -1)  # [n,H,N]; fi-1440d < origin always
    qk = np.quantile(mem, LEVELS, axis=-1)  # [Q,n,H]
    return np.moveaxis(qk, 0, -1) * S.Ceff[fi][..., None]


def psp_fit(S, o):
    P = np.nan_to_num(S.x['P']); fi = o[:, None] + np.arange(1, H + 1)[None]
    sp = S.smart_persistence(o); r = (P[fi] - sp) / S.Ceff[fi]; g = regime(S.Ceff[fi], S.cap)
    tab = np.zeros((H, 3, len(LEVELS)))
    for h in range(H):
        for k in range(3):
            m = g[:, h] == k
            tab[h, k] = np.quantile(r[m, h], LEVELS) if m.sum() > 30 else np.quantile(r[:, h], LEVELS)
    tab[..., int(np.argmin(np.abs(LEVELS - 0.5)))] = 0.0  # median = smart persistence
    return tab


def psp(S, o, tab):
    fi = o[:, None] + np.arange(1, H + 1)[None]; sp = S.smart_persistence(o); g = regime(S.Ceff[fi], S.cap)
    off = tab[np.arange(H)[None], g]  # [n,H,Q]
    return np.sort(sp[..., None] + off * S.Ceff[fi][..., None], -1)


if __name__ == '__main__':
    for st, (_, cap) in STATIONS.items():
        S = Station(ROOT / 'data' / f'{st}.npz', st, cap)
        tr = np.sort(S.origins['train']); tr = tr[tr > 1440 * NDAYS]
        tab = psp_fit(S, tr)
        for name in ('CHPeEn', 'PSP'):
            od = ROOT / 'runs/stageA' / name / st; od.mkdir(parents=True, exist_ok=True)
            for split in ('validation', 'test'):
                o = np.sort(S.origins[split])
                q = chpeen(S, o) if name == 'CHPeEn' else psp(S, o, tab)
                np.savez_compressed(od / f'{split}.npz', q=q.astype(np.float32), origins=o, seconds=0.0)
        print('done', st, flush=True)
