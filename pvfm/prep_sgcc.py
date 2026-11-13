"""Public data (Chen et al., Sci. Data 2022; State Grid forecasting competition, CC0): solar sites 1 (50 MW, 2019, as
'Dataset 1' of PARA-PV) and 6 (35 MW, 2019-2020, 'Dataset 2'). 15-min values are expanded to 1-min (each value held for
its 15 minutes) so that the 15-min pipeline reproduces them exactly; -99 is missing. Split 8:1:1 by rows as in PARA-PV.
Coordinates are not published: latitude/longitude are fitted to the irradiance curve on clear days (coarse-to-fine).
Usage: PVFM_ROOT=<root> python -m pvfm.prep_sgcc <xlsx_dir> <out_data_dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from pvfm.common import cos_zenith, ghi_clearsky

SITES = {'sgcc1_pv': ('Solar station site 1 (Nominal capacity-50MW).xlsx', 50.0, ('2019-01-01', '2019-12-31 23:45')),
         'sgcc6_pv': ('Solar station site 6 (Nominal capacity-35MW).xlsx', 35.0, ('2019-01-01', '2020-12-31 23:45'))}
COLS = {'Total solar irradiance (W/m2)': 'ghi', 'Direct normal irradiance (W/m2)': 'dni', 'Global horizontal irradiance (W/m2)': 'dhi',
        'Air temperature  (°C) ': 'temp', 'Atmosphere (hpa)': 'pressure', 'Power (MW)': 'P'}


def fit_coords(m, g, thr=10.0):
    """Coordinates from the daily light period (robust to tilted sensors): per day, first and last 15-min interval with
    irradiance > thr. Latitude: least squares of the day length against the sunrise equation (with a fitted threshold
    elevation h0). Longitude: the mid-point of the light period is solar noon; lon = 120 - (noon - 720 + EoT)/4."""
    d = m // 1440; tod = m % 1440; df = pd.DataFrame({'d': d, 'tod': tod, 'on': g > thr})
    on = df[df.on].groupby('d').tod.agg(['min', 'max']); on = on[(on['max'] - on['min']) > 300]
    doy = pd.to_datetime(on.index.to_numpy() * 1440, unit='m').dayofyear.to_numpy(); gam = 2 * np.pi * (doy - 1) / 365
    dec = (0.006918 - 0.399912 * np.cos(gam) + 0.070257 * np.sin(gam) - 0.006758 * np.cos(2 * gam) + 0.000907 * np.sin(2 * gam)
           - 0.002697 * np.cos(3 * gam) + 0.00148 * np.sin(3 * gam))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(gam) - 0.032077 * np.sin(gam) - 0.014615 * np.cos(2 * gam) - 0.040849 * np.sin(2 * gam))
    L = (on['max'] - on['min'] + 15).to_numpy().astype(float); mid = ((on['max'] + on['min'] + 15) / 2).to_numpy()
    best = None
    for lat in np.arange(18, 54.01, 0.05):
        for h0 in np.arange(-2, 12.01, 0.25):
            phi, h = np.radians(lat), np.radians(h0)
            c = np.clip((np.sin(h) - np.sin(phi) * np.sin(dec)) / (np.cos(phi) * np.cos(dec)), -1, 1)
            e = float(np.median(np.abs(L - 2 * np.degrees(np.arccos(c)) * 4)))
            if best is None or e < best[0]: best = (e, lat, h0)
    lon = float(np.median(120 - (mid - 720 + eot) / 4))
    return round(best[1], 2), round(lon, 2), len(L)


if __name__ == '__main__':
    src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); coords, meta = {}, {}
    for st, (fn, cap, (a, b)) in SITES.items():
        x = pd.read_excel(src / fn); x = x.rename(columns=COLS)
        t = pd.to_datetime(x['Time(year-month-day h:m:s)']); x = x[(t >= a) & (t <= b)].copy(); t = t[(t >= a) & (t <= b)]
        for c in COLS.values(): x[c] = pd.to_numeric(x[c], errors='coerce'); x.loc[x[c] <= -99, c] = np.nan
        m15 = t.to_numpy().astype('datetime64[m]').astype(np.int64); assert (np.diff(m15) == 15).all(), 'gaps in 15-min index'
        n15 = len(m15); t0 = int(m15[0]); n = 15 * n15
        rep = lambda v: np.repeat(np.asarray(v, np.float32), 15)
        arr = {c: rep(x[c].to_numpy()) for c in COLS.values()}
        arr['P'] = np.nan_to_num(arr['P']); arr['agc'] = np.full(n, cap, np.float32); arr['avail'] = np.full(n, cap, np.float32)
        arr['wind'] = np.full(n, np.nan, np.float32)
        g = x['ghi'].to_numpy(); ok = np.isfinite(g)
        lat, lon, nd = fit_coords(m15[ok], np.nan_to_num(g[ok])); coords[st] = [lat, lon]
        # 8:1:1 by rows (PARA-PV); 15-min origin k = last observed row, issued at 1-min index 15k+14
        n_tr = int(n15 * 0.8); n_te = int(n15 * 0.1); n_va = n15 - n_tr - n_te
        org = lambda k0, k1: 15 * np.arange(k0, k1) + 14
        origins = dict(train=org(671, n_tr - 16), validation=org(n_tr - 1, n_tr + n_va - 16), test=org(n_tr + n_va - 1, n15 - 16))
        np.savez_compressed(out / f'{st}.npz', t0=np.int64(t0), observed=np.ones(n, bool), **{f'x_{c}': v for c, v in arr.items()},
                            **{f'origin_{k}': v for k, v in origins.items()})
        meta[st] = dict(capacity=cap, lat=lat, lon=lon, coord_fit_days=nd, rows=n15, first=str(t.iloc[0]), last=str(t.iloc[-1]),
                        rows_train=n_tr, rows_val=n_va, rows_test=n_te, test_first=str(t.iloc[n_tr + n_va]),
                        missing={c: float(np.isnan(x[c].to_numpy()).mean()) for c in COLS.values()})
        print(st, json.dumps(meta[st]), flush=True)
    (out / 'coords.json').write_text(json.dumps(coords)); (out / 'meta.json').write_text(json.dumps(meta, indent=1))
