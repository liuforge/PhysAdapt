"""Clear-sky envelope C_t and smart persistence, fitted on the training period only."""
import numpy as np
from pvfm.common import cos_zenith, ghi_clearsky, PVROOT, DATASET

CITY = {'feicheng_pv': (36.18, 116.77), 'heze_pv': (35.23, 115.48), 'huimin_pv': (37.49, 117.51), 'jinxiang_pv': (35.07, 116.31),
        'laiwu_pv': (36.21, 117.68), 'xiaozhuang_pv': (36.50, 116.00), 'yiyuan_pv': (36.18, 118.15)}

if DATASET == 'sgcc':  # fitted from the irradiance data (prep_sgcc.py); exact coordinates are not published
    import json
    CITY = {k: tuple(v) for k, v in json.load(open(f'{PVROOT}/data/coords.json')).items()}

class Station:
    def __init__(self, path, name, cap):
        z = np.load(path)
        self.name, self.cap = name, cap
        self.t0 = int(z['t0']); self.n = len(z['observed'])
        self.minutes = self.t0 + np.arange(self.n)
        self.x = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith('x_')}
        self.origins = {s: z[f'origin_{s}'] for s in ('train', 'validation', 'test')}
        self.train_end = int(self.origins['validation'].min()) - 96  # last minute usable for fitting physics
        lat, lon = CITY[name]
        self.lat, self.lon = lat, lon
        P = np.nan_to_num(self.x['P'])
        cs0 = ghi_clearsky(cos_zenith(self.minutes, lat, lon))
        tr = slice(0, self.train_end)
        # timing offset: align P centroid with clear-sky centroid on training days (minutes, positive = P later)
        day = (self.minutes[tr] // 1440); tod = self.minutes[tr] % 1440
        Pp = np.clip(P[tr], 0, None)
        cp = np.bincount(day - day.min(), Pp * tod) / np.maximum(np.bincount(day - day.min(), Pp), 1e-9)
        cc = np.bincount(day - day.min(), cs0[tr] * tod) / np.maximum(np.bincount(day - day.min(), cs0[tr]), 1e-9)
        ok = np.bincount(day - day.min(), Pp) > 0.2 * cap * 60
        self.shift = float(np.clip(np.median((cp - cc)[ok]), -30, 30))
        self.cz = cos_zenith(self.minutes - self.shift, lat, lon).astype(np.float32)
        cs = ghi_clearsky(self.cz)
        sel = cs[tr] > 300
        self.kappa = float(np.quantile(P[tr][sel] / cs[tr][sel], 0.99))
        self.C = (self.kappa * cs).astype(np.float32)
        self.floor = 0.02 * cap
        self.Ceff = np.maximum(self.C, self.floor).astype(np.float32)
        self.K = (P / self.Ceff).astype(np.float32)

    def windows(self, split, ctx, H=96):
        o = self.origins[split]
        o = o[o - ctx + 1 >= 0]
        return o, o[:, None] + np.arange(-ctx + 1, 1)[None], o[:, None] + np.arange(1, H + 1)[None]

    def smart_persistence(self, o, H=96):
        P = np.nan_to_num(self.x['P']); f = o[:, None] + np.arange(1, H + 1)[None]
        return P[o][:, None] * self.Ceff[f] / self.Ceff[o][:, None]


def curtailment(S):
    """Robust binding-curtailment flag (does not use 'available power'): AGC set-point below 97% of rated capacity
    and active power pinned at the set-point (within 3% of capacity)."""
    P = np.nan_to_num(S.x['P']); agc = S.x['agc']
    a = np.where(np.isfinite(agc), agc, S.cap)
    return ((a < 0.97 * S.cap) & (P >= a - 0.03 * S.cap) & (S.C > 0.02 * S.cap)), a


def potential(S):
    """Reconstructed uncurtailed potential: P where not binding; during binding use max(P, r*avail) when the station's
    available-power channel is reliable on uncurtailed daylight training data (corr > 0.9), else P (censored at AGC).
    Returns (P_pot, reliable_flag, ratio)."""
    P = np.nan_to_num(S.x['P']); av = S.x['avail']; bind, a = curtailment(S)
    tr = np.zeros(S.n, bool); tr[:S.train_end] = True
    ok = tr & np.isfinite(av) & (S.C > 0.2 * S.cap) & (av > 0.1 * S.cap) & ~bind
    if ok.sum() < 1000:
        return P.copy(), False, np.nan
    r = float(np.median(P[ok] / av[ok])); c = float(np.corrcoef(P[ok], av[ok])[0, 1])
    rel = c > 0.9 and 0.85 < r < 1.15
    Pp = P.copy()
    if rel:
        m = bind & np.isfinite(av)
        Pp[m] = np.maximum(P[m], np.minimum(r * av[m], 1.1 * S.cap))
    return Pp, rel, r
