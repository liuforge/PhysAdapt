"""Evaluation protocols. 'main' = original 70/10/20 split. 'rollK' = calendar rolling-origin folds (expanding window):
test block [t0, t1); calibration = 14 days before t0; early stopping = the 14 days before that; train = everything earlier.
All origins come from the non-overlapping 1-min window grid of the original caches (union of train/val/test origins)."""
import numpy as np
import pandas as pd

FOLDS = {  # name: (test start, test end), local time
    'roll1': ('2025-09-01', '2025-10-06'),
    'roll2': ('2025-10-20', '2025-11-24'),
    'roll3': ('2025-12-08', '2026-01-12'),
    'roll4': ('2026-02-01', '2026-03-08'),
    'roll5': ('2026-03-15', '2026-04-30'),
}
REGIME = {'roll1': 'free', 'roll2': 'free', 'roll3': 'transition', 'roll4': 'dispatch', 'roll5': 'dispatch'}
H = 96


def _minute(s):
    return int(np.datetime64(s, 'm').astype(np.int64))


def all_origins(S):
    return np.unique(np.concatenate([S.origins[k] for k in ('train', 'validation', 'test')]))


def splits(S, proto, min_ctx=8192):
    """returns dict train/es/cal/test of sorted origin indices for Station S (1-min)."""
    if proto == 'main':
        va = np.sort(S.origins['validation']); h = len(va) // 2
        out = dict(train=np.sort(S.origins['train']), es=va[:h], cal=va[h:], test=np.sort(S.origins['test']))
    else:
        t0, t1 = (_minute(x) - S.t0 for x in FOLDS[proto])
        o = all_origins(S); o = o[o >= min_ctx - 1]
        d14 = 14 * 1440
        out = dict(train=o[o + H < t0 - 2 * d14], es=o[(o >= t0 - 2 * d14) & (o + H < t0 - d14)],
                   cal=o[(o >= t0 - d14) & (o + H < t0)], test=o[(o >= t0) & (o + H < t1)])
    return out


def available(S, proto, min_train_windows=200):
    """a fold is usable for a station when it has a test block and enough history."""
    sp = splits(S, proto)
    return len(sp['test']) > 50 and len(sp['cal']) > 50 and len(sp['es']) > 50 and len(sp['train']) >= min_train_windows
