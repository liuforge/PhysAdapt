"""Load aligned forecasts (several experts) + targets for one station, 1-min or 15-min task, ordered in time."""
from pvfm.common import PVROOT
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station

ROOT = Path(PVROOT)
MID = int(np.argmin(np.abs(LEVELS - 0.5)))


def regime(Ceff, cap):
    r = Ceff / cap
    return np.where(r <= 0.0201, 0, np.where(r < 0.3, 1, 2)).astype(np.int8)


def _fm(variant, st, split, task):
    d = ROOT / 'runs' / ('stageA' if task == '1min' else 't15A') / variant / st / f'{split}.npz'
    z = np.load(d); return z['origins'], z['q']


def _adapter(spec, st, split):
    """spec = '<run dir under runs/>:<tag>' -> mean over all seeds found."""
    run, tag = spec.split(':')
    fs = sorted((ROOT / 'runs' / run).glob(f'pred_{tag}_{st}_*.npz'))
    assert fs, (run, tag, st)
    qs, o = [], None
    for f in fs:
        z = np.load(f)
        oo, q = (z['o_val'], z['q_val']) if split == 'validation' else (z['o'], z['q_uncal'])
        o = oo if o is None else o; assert np.array_equal(o, oo); qs.append(q)
    return o, np.mean(qs, 0)


def load(st, experts, split, task='1min'):
    """experts: dict name -> 'fm:<variant>' | 'ad:<run>:<tag>' | 'sp:<expert-name-for-spread>'"""
    cap = STATIONS[st][1]
    if task == '1min':
        S = Station(ROOT / 'data' / f'{st}.npz', st, cap); H = 96
        P, Ceff, cz = np.nan_to_num(S.x['P']), S.Ceff, S.cz
        aux = dict(ghi=S.x['ghi'], agc=S.x['agc'], avail=S.x['avail'], K=S.K)
    else:
        from pvfm.task15 import Station15
        S = Station15(ROOT / 'data' / f'{st}.npz', st, cap); H = 16
        P, Ceff, cz = S.P, S.Ceff, S.cz
        S.x = {**S.x}; S.x.setdefault('ghi', S.x.get('ghi'))
        aux = dict(ghi=S.x['ghi'], agc=S.x['agc'], avail=S.x['avail'], K=S.K)
    raw = {}
    for name, src in experts.items():
        kind, rest = src.split(':', 1)
        if kind == 'fm': raw[name] = _fm(rest, st, split, task)
        elif kind == 'ad': raw[name] = _adapter(rest, st, split)
        elif kind == 'run':
            z = np.load(ROOT / 'runs' / rest / st / f'{split}.npz'); raw[name] = (z['origins'], z['q'])
    common = None
    for o, _ in raw.values():
        common = o if common is None else np.intersect1d(common, o)
    common = np.sort(common)
    out = dict(o=common, cap=cap, H=H)
    # online feedback delay in windows: labels of window j are complete only once origin >= o_j + H
    stride = int(np.median(np.diff(common))) if len(common) > 1 else H
    out['delay'] = max(1, int(np.ceil(H / stride)))
    fi = common[:, None] + np.arange(1, H + 1)[None]
    out['y'] = P[fi].astype(np.float32); out['Ceff'] = Ceff[fi].astype(np.float32); out['cz'] = cz[fi]
    out['reg'] = regime(out['Ceff'], cap)
    q = {}
    for name, (o, qq) in raw.items():
        order = np.argsort(o); q[name] = qq[order[np.searchsorted(o[order], common)]]
    for name, src in experts.items():
        if src.startswith('sp:'):  # smart persistence median + spread borrowed from another expert
            ref = q[src[3:]]
            spm = P[common][:, None] * Ceff[fi] / Ceff[common][:, None]
            q[name] = ref - ref[..., MID:MID + 1] + spm[..., None]
    out['q'] = q
    # issue-time context features
    o = common
    ghi = np.nan_to_num(aux['ghi']); Kx = np.clip(aux['K'], -0.5, 1.5)
    look = 60 if task == '1min' else 4
    past = o[:, None] + np.arange(-look + 1, 1)[None]
    out['ctx'] = dict(cz0=cz[o], K0=Kx[o], Kstd=Kx[past].std(1), Kmean=Kx[past].mean(1),
                      curt=((np.nan_to_num(aux['agc'][o], nan=1e9) < 0.97 * np.nan_to_num(aux['avail'][o])) & (np.nan_to_num(aux['avail'][o]) > 0.05 * cap)).astype(np.float32))
    return out
