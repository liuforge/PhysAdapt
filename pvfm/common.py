"""Shared definitions for the pvfm (physics-consistent foundation-model adaptation) pipeline."""
import os
import numpy as np

PVROOT = os.environ.get('PVFM_ROOT', '/mnt/user/pvfm')
DATASET = os.environ.get('PVFM_DATASET', 'shandong')

STATIONS = {
    # name: (raw-file stem, capacity in MW). The Shandong SCADA data are not public; raw-file stems are anonymised.
    'feicheng_pv': ('feicheng_pv_raw', 20.0),
    'heze_pv': ('heze_pv_raw', 20.0),
    'huimin_pv': ('huimin_pv_raw', 20.0),
    'jinxiang_pv': ('jinxiang_pv_raw', 20.0),
    'laiwu_pv': ('laiwu_pv_raw', 40.0),
    'xiaozhuang_pv': ('xiaozhuang_pv_raw', 40.0),
    'yiyuan_pv': ('yiyuan_pv_raw', 20.0),
}
if DATASET == 'sgcc':  # public State Grid competition data (Chen et al., Sci. Data 2022), sites 1 and 6
    STATIONS = {'sgcc1_pv': ('Solar station site 1 (Nominal capacity-50MW)', 50.0), 'sgcc6_pv': ('Solar station site 6 (Nominal capacity-35MW)', 35.0)}
COLMAP = {'日期': 'ts', '有功功率': 'P', '可用功率': 'avail', 'AGC调度命令值': 'agc', '总辐射辐照度': 'ghi',
          '直接辐照度': 'dni', '散射辐照度': 'dhi', '倾斜辐射辐照度': 'poa', '气温': 'temp', '气象站风速': 'wind',
          '气压': 'pressure', '风向': 'wdir', '电池板温度': 'tpanel'}
LEVELS = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
TZ = 8.0


def cos_zenith(minutes_utc8, lat, lon):
    """minutes_utc8: int64 minutes since 1970-01-01 in local UTC+8 wall time. Spencer declination + equation of time."""
    m = np.asarray(minutes_utc8, dtype=np.int64)
    d = m.astype('datetime64[m]')
    doy = (d.astype('datetime64[D]') - d.astype('datetime64[Y]').astype('datetime64[D]')).astype(np.int64) + 1
    hour = (m % 1440) / 60.0
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    solar_time = hour * 60 + eot + 4 * lon - 60 * TZ  # minutes
    ha = np.radians(solar_time / 4 - 180)
    la = np.radians(lat)
    return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)


def ghi_clearsky(cz):
    """Haurwitz clear-sky GHI (W/m2)."""
    c = np.clip(cz, 1e-4, None)
    return np.where(cz > 0, 1098.0 * c * np.exp(-0.057 / c), 0.0)


def pinball(y, q, levels):
    e = y[..., None] - q
    return np.maximum(levels * e, (levels - 1) * e)


def score(y, q, levels, cap=None):
    """y: (...,), q: (..., K) monotone. Returns dict of point/probabilistic metrics (median = level 0.5)."""
    levels = np.asarray(levels)
    mi = int(np.argmin(np.abs(levels - 0.5)))
    e = y - q[..., mi]
    out = dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
               pinball=float(pinball(y, q, levels).mean()))
    # CRPS approx: 2 * mean over levels of pinball, trapezoid over the available level grid
    pl = pinball(y, q, levels).reshape(-1, len(levels)).mean(0)
    out['crps_trap'] = float(2 * np.trapezoid(pl, levels) / (levels[-1] - levels[0]) * (levels[-1] - levels[0]))
    for lo, hi, name in [(0.3, 0.7, 40), (0.2, 0.8, 60), (0.1, 0.9, 80), (0.05, 0.95, 90)]:
        if lo in levels and hi in levels:
            i, j = list(levels).index(lo), list(levels).index(hi)
            cov = ((y >= q[..., i]) & (y <= q[..., j])).mean()
            out[f'cov{name}'] = float(cov); out[f'width{name}'] = float((q[..., j] - q[..., i]).mean())
    if cap:
        out['nmae'] = out['mae'] / cap; out['nrmse'] = out['rmse'] / cap
    return out


def monotone(q):
    return np.maximum.accumulate(q, axis=-1)
