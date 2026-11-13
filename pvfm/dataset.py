"""Assemble GPU-ready tensors for one station: K-space targets, FM quantiles, history (2 scales), future geometry."""
from pvfm.common import PVROOT
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station
from pvfm.fm_zero import ffill

H = 96
HIST_F = ['K', 'Pn', 'availn', 'agcn', 'curt', 'kt', 'dfrac', 'temp', 'wind', 'cz']
COARSE_F = ['K', 'kt', 'agcn', 'cz']


def _features(S):
    cap = S.cap
    P = np.nan_to_num(S.x['P'])
    ghi = np.clip(ffill(S.x['ghi']), 0, None)
    cz = S.cz
    ghi_cs = np.where(cz > 0.05, 1098.0 * np.clip(cz, 0.05, None) * np.exp(-0.057 / np.clip(cz, 0.05, None)), np.nan)
    avail = ffill(S.x['avail']); agc = ffill(S.x['agc'])
    f = dict(
        K=np.clip(S.K, -1, 2),
        Pn=P / cap,
        availn=np.clip(avail / S.Ceff, 0, 2),
        agcn=np.clip(agc / cap, 0, 1.5),
        curt=((agc < 0.97 * avail) & (avail > 0.05 * cap)).astype(np.float32),
        kt=np.nan_to_num(np.clip(ghi / ghi_cs, 0, 1.5)),
        dfrac=np.clip(np.nan_to_num(ffill(S.x['dhi']) / np.maximum(ghi, 1)), 0, 1),
        temp=ffill(S.x['temp']) / 40.0,
        wind=ffill(S.x['wind']) / 10.0,
        cz=cz,
    )
    return {k: v.astype(np.float32) for k, v in f.items()}


def build(station, data_dir, fm_dir, splits=('train', 'validation', 'test')):
    cap = STATIONS[station][1]
    S = Station(Path(data_dir) / f'{station}.npz', station, cap)
    F = _features(S)
    out = {}
    for split in splits:
        o = np.sort(S.origins[split])
        o = o[o - 1440 + 1 >= 0]
        q = None
        if fm_dir is not None:
            z = np.load(Path(fm_dir) / station / f'{split}.npz')
            zo, order = z['origins'], np.argsort(z['origins'])
            o = o[np.isin(o, zo)]
            q = z['q'][order[np.searchsorted(zo[order], o)]]
        fi = o[:, None] + np.arange(1, H + 1)[None]
        ri = o[:, None] + np.arange(-95, 1)[None]
        ci = o[:, None] + np.arange(-1439, 1)[None]
        Ce = S.Ceff[fi]
        hist = np.stack([F[k][ri] for k in HIST_F], -1)
        coarse = np.stack([F[k][ci].reshape(len(o), 96, 15).mean(-1) for k in COARSE_F], -1)
        fut = np.stack([S.cz[fi], Ce / cap, np.broadcast_to(np.arange(H) / H, fi.shape)], -1)
        yP = np.nan_to_num(S.x['P'])[fi]
        d = dict(o=o, yP=yP.astype(np.float32), Ceff=Ce.astype(np.float32), hist=hist.astype(np.float32),
                 coarse=coarse.astype(np.float32), fut=fut.astype(np.float32), Klast=F['K'][o],
                 agc_last=np.nan_to_num(S.x['agc'][o], nan=cap).astype(np.float32),
                 sp=S.smart_persistence(o).astype(np.float32), cap=cap)
        if q is not None:
            d['qK'] = (q / Ce[..., None]).astype(np.float32)
        out[split] = d
    return out


def build15(station, data_dir, fm_dir, splits=('train', 'validation', 'test')):
    """15-min task: recent 8 h at 15 min (32 steps) + past 7 days hourly (168 steps); H=16."""
    from pvfm.task15 import Station15, H15
    cap = STATIONS[station][1]
    S = Station15(Path(data_dir) / f'{station}.npz', station, cap)
    B = S.base; F1 = _features(B)
    n = len(S.P)
    agg = lambda a, r=15: np.nanmean(a[:n * r].reshape(n, r), 1).astype(np.float32)
    F = {k: agg(v) for k, v in F1.items()}
    F['K'] = np.clip(S.K, -1, 2); F['Pn'] = S.P / cap
    out = {}
    for split in splits:
        o = np.sort(S.origins[split])
        z = np.load(Path(fm_dir) / station / f'{split}.npz'); zo, order = z['origins'], np.argsort(z['origins'])
        o = o[np.isin(o, zo)]; q = z['q'][order[np.searchsorted(zo[order], o)]]
        fi = o[:, None] + np.arange(1, H15 + 1)[None]
        ri = o[:, None] + np.arange(-31, 1)[None]
        ci = o[:, None] + np.arange(-671, 1)[None]
        Ce = S.Ceff[fi]
        hist = np.stack([F[k][ri] for k in HIST_F], -1)
        coarse = np.stack([F[k][ci].reshape(len(o), 168, 4).mean(-1) for k in COARSE_F], -1)
        fut = np.stack([S.cz[fi], Ce / cap, np.broadcast_to(np.arange(H15) / H15, fi.shape)], -1)
        out[split] = dict(o=o, yP=S.P[fi].astype(np.float32), Ceff=Ce.astype(np.float32), hist=hist.astype(np.float32),
                          coarse=coarse.astype(np.float32), fut=fut.astype(np.float32), Klast=F['K'][o].astype(np.float32),
                          agc_last=np.full(len(o), cap, np.float32), sp=S.smart_persistence(o).astype(np.float32), cap=cap,
                          qK=(q / Ce[..., None]).astype(np.float32))
    return out


def build_proto(station, data_dir, fm_variant, proto):
    """Same tensors as build(), but splits from a protocol (train / validation = es+cal / test) and FM quantiles
    from the all-origins cache. Also returns the es/cal boundary so split_val reproduces the protocol exactly."""
    from pvfm.protocol import splits
    from pvfm.rpreds import _fm_all
    cap = STATIONS[station][1]
    S = Station(Path(data_dir) / f'{station}.npz', station, cap)
    F = _features(S)
    sp = splits(S, proto)
    zo, zq = _fm_all(fm_variant, station); order = np.argsort(zo)
    out = {}
    for part, o in (('train', sp['train']), ('validation', np.concatenate([sp['es'], sp['cal']])), ('test', sp['test'])):
        o = np.sort(o); o = o[(o - 1440 + 1 >= 0) & np.isin(o, zo)]
        q = zq[order[np.searchsorted(zo[order], o)]]
        fi = o[:, None] + np.arange(1, H + 1)[None]; ri = o[:, None] + np.arange(-95, 1)[None]; ci = o[:, None] + np.arange(-1439, 1)[None]
        Ce = S.Ceff[fi]
        out[part] = dict(o=o, ot=(S.t0 + o).astype(np.float64), yP=np.nan_to_num(S.x['P'])[fi].astype(np.float32), Ceff=Ce.astype(np.float32),
                         hist=np.stack([F[k][ri] for k in HIST_F], -1).astype(np.float32),
                         coarse=np.stack([F[k][ci].reshape(len(o), 96, 15).mean(-1) for k in COARSE_F], -1).astype(np.float32),
                         fut=np.stack([S.cz[fi], Ce / cap, np.broadcast_to(np.arange(H) / H, fi.shape)], -1).astype(np.float32),
                         Klast=F['K'][o], agc_last=np.nan_to_num(S.x['agc'][o], nan=cap).astype(np.float32),
                         sp=S.smart_persistence(o).astype(np.float32), cap=cap,
                         # K-space input of M2; the clip is a no-op for Kc bases, guards power-space bases near sunrise/sunset
                         qK=np.clip(q / Ce[..., None], -1.0, 2.0).astype(np.float32))
    out['n_es'] = int(np.isin(out['validation']['o'], sp['es']).sum())
    return out


def fm_residual_series(station, fm_variant, data_dir=(PVROOT + '/data')):
    """1-min series R(t) = K(t) - K_fm_median(t), where K_fm_median(t) comes from the non-overlapping window that covers t
    (issued at origin o' < t <= o'+96). At any origin o, R over [o-95, o] only uses forecasts issued before o and
    observations up to o: no look-ahead. Undefined minutes -> 0, plus availability mask."""
    from pvfm.rpreds import _fm_all
    cap = STATIONS[station][1]
    S = Station(Path(data_dir) / f'{station}.npz', station, cap)
    zo, zq = _fm_all(fm_variant, station)
    R = np.zeros(S.n, np.float32); A = np.zeros(S.n, np.float32)
    K = np.clip(S.K, -0.5, 1.5)
    for o, q in zip(zo, zq):
        t = o + np.arange(1, H + 1); t = t[t < S.n]
        R[t] = K[t] - q[:len(t), 6] / S.Ceff[t]; A[t] = 1.0
    return S.t0, R, A


def build_proto_res(station, data_dir, fm_variant, proto, neighbours=True, _cache={}):
    """build_proto + residual-feedback channels appended to the recent-history tensor: own FM residual (last 96 min)
    and, optionally, the mean residual of the other stations at the same absolute minutes."""
    out = build_proto(station, data_dir, fm_variant, proto)
    key = fm_variant
    if key not in _cache:
        _cache[key] = {s: fm_residual_series(s, fm_variant, data_dir) for s in STATIONS}
    res = _cache[key]
    t0, R, A = res[station]
    if neighbours:
        tot = np.zeros_like(R); cnt = np.zeros_like(R)
        for s, (t0s, Rs, As) in res.items():
            if s == station: continue
            lo = max(t0, t0s); hi = min(t0 + len(R), t0s + len(Rs))
            if hi <= lo: continue
            tot[lo - t0:hi - t0] += Rs[lo - t0s:hi - t0s]; cnt[lo - t0:hi - t0] += As[lo - t0s:hi - t0s]
        Rn = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0).astype(np.float32)
    for part in ('train', 'validation', 'test'):
        o = out[part]['o']; ri = o[:, None] + np.arange(-95, 1)[None]
        extra = [R[ri], A[ri]] + ([Rn[ri]] if neighbours else [np.zeros_like(R[ri])])
        out[part]['hist'] = np.concatenate([out[part]['hist'], np.stack(extra, -1)], -1).astype(np.float32)
    return out


def sim_plan(a, fi, lag, sig, cap, seed):
    """Simulated dispatch plan for horizon indices fi: set-point curve shifted by `lag` minutes plus AR(1) noise."""
    A = a[np.clip(fi - lag, 0, len(a) - 1)].astype(float)
    if sig > 0:
        rng = np.random.default_rng(seed); eps = np.zeros(fi.shape); z = rng.normal(0, 1, fi.shape)
        for h in range(fi.shape[1]):
            eps[:, h] = (0.95 * eps[:, h - 1] if h else 0) + np.sqrt(1 - 0.95 ** 2) * z[:, h]
        A = np.clip(A + sig * cap * eps, 0, cap)
    return A


def build_proto_plan(station, data_dir, fm_variant, proto, lag, sig):
    """build_proto + a simulated dispatch plan (fraction of capacity) appended to the future-known covariates.
    Independent noise seeds per split so the adapter cannot memorise the noise."""
    from pvfm.physics import curtailment
    out = build_proto(station, data_dir, fm_variant, proto)
    S = Station(Path(data_dir) / f'{station}.npz', station, STATIONS[station][1]); _, a = curtailment(S)
    for i, part in enumerate(('train', 'validation', 'test')):
        o = out[part]['o']; fi = o[:, None] + np.arange(1, H + 1)[None]
        import zlib
        pl = sim_plan(a, fi, lag, sig, S.cap, seed=zlib.crc32(f'{station}|{proto}|{part}|{lag}|{sig}'.encode())) / S.cap
        out[part]['fut'] = np.concatenate([out[part]['fut'], pl[..., None].astype(np.float32)], -1)
    return out


def build15_proto(station, data_dir, fm_variant, proto, target='P'):
    """15-min analogue of build_proto (same adapter tensors as build15), splits from splits15."""
    from pvfm.task15 import Station15, H15, splits15, fm15_all
    cap = STATIONS[station][1]
    S = Station15(Path(data_dir) / f'{station}.npz', station, cap)
    F1 = _features(S.base); n = len(S.P)
    agg = lambda a, r=15: np.nanmean(a[:n * r].reshape(n, r), 1).astype(np.float32)
    F = {k: agg(v) for k, v in F1.items()}
    F['K'] = np.clip(S.K, -1, 2); F['Pn'] = S.P / cap
    sp = splits15(S, proto); zo, zq = fm15_all(fm_variant, station)
    out = {}
    for part, o in (('train', sp['train']), ('validation', np.concatenate([sp['es'], sp['cal']])), ('test', sp['test'])):
        o = np.sort(o); o = o[np.isin(o, zo)]; q = zq[np.searchsorted(zo, o)]
        fi = o[:, None] + np.arange(1, H15 + 1)[None]; ri = o[:, None] + np.arange(-31, 1)[None]; ci = o[:, None] + np.arange(-671, 1)[None]
        Ce = S.Ceff[fi]
        yv = (np.nan_to_num(S.x['avail']) if target == 'avail' else S.P)[fi]  # 'avail': available power (grid-code basis when restricted)
        out[part] = dict(o=o, ot=(15.0 * o), yP=yv.astype(np.float32), Ceff=Ce.astype(np.float32),
                         hist=np.stack([F[k][ri] for k in HIST_F], -1).astype(np.float32),
                         coarse=np.stack([F[k][ci].reshape(len(o), 168, 4).mean(-1) for k in COARSE_F], -1).astype(np.float32),
                         fut=np.stack([S.cz[fi], Ce / cap, np.broadcast_to(np.arange(H15) / H15, fi.shape)], -1).astype(np.float32),
                         Klast=F['K'][o].astype(np.float32), agc_last=np.full(len(o), cap, np.float32),
                         sp=S.smart_persistence(o).astype(np.float32), cap=cap, qK=(q / Ce[..., None]).astype(np.float32))
    out['n_es'] = int(np.isin(out['validation']['o'], sp['es']).sum())
    return out
