"""Protocol-aware loader: experts' quantile forecasts + targets for one station, one protocol ('main'|'rollK'), one part
(train/es/cal/test). Expert sources:
  fm:<variant>        zero-shot cache runs/stageA/<variant>/<st>/all.npz (fallback: union of split files)
  fold:<run>          fold-trained model outputs runs/<run>/<proto>/<st>.npz with o_<part>, q_<part>
  sp:<expert>         smart-persistence median + spread of <expert>
"""
from pvfm.common import PVROOT
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station, curtailment
from pvfm.protocol import splits
from pvfm.preds import regime

ROOT = Path(PVROOT)
MID = int(np.argmin(np.abs(LEVELS - 0.5)))
H = 96
_cache = {}


def station(st):
    if st not in _cache:
        _cache[st] = Station(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1])
    return _cache[st]


def _fm_all(variant, st):
    d = ROOT / 'runs/stageA' / variant / st
    if (d / 'all.npz').exists():
        z = np.load(d / 'all.npz'); return z['origins'], z['q']
    os_, qs = [], []
    for s in ('train', 'validation', 'test'):
        if (d / f'{s}.npz').exists():
            z = np.load(d / f'{s}.npz'); os_.append(z['origins']); qs.append(z['q'])
    o = np.concatenate(os_); q = np.concatenate(qs); u, i = np.unique(o, return_index=True)
    return u, q[i]


def _fold(run, st, proto, part):
    z = np.load(ROOT / 'runs' / run / proto / f'{st}.npz')
    return z[f'o_{part}'], z[f'q_{part}']


def load(st, experts, part, proto):
    S = station(st); cap = S.cap
    want = splits(S, proto)[part]
    raw = {}
    for name, src in experts.items():
        kind, rest = src.split(':', 1)
        if kind == 'fm': raw[name] = _fm_all(rest, st)
        elif kind == 'fold': raw[name] = _fold(rest, st, proto, part)
    common = want
    for o, _ in raw.values():
        common = np.intersect1d(common, o)
    common = np.sort(common)
    fi = common[:, None] + np.arange(1, H + 1)[None]
    P = np.nan_to_num(S.x['P'])
    out = dict(o=common, cap=cap, H=H, delay=1, y=P[fi].astype(np.float32), Ceff=S.Ceff[fi].astype(np.float32), cz=S.cz[fi])
    out['reg'] = regime(out['Ceff'], cap)
    b, a = curtailment(S); out['bind'] = b[fi]; out['agc_o'] = a[common]
    q = {}
    for name, (o, qq) in raw.items():
        # loose physical-range guard: removes non-physical FM quantile blow-ups (seen in power-space Chronos-2) without ties at 0
        order = np.argsort(o); q[name] = np.clip(qq[order[np.searchsorted(o[order], common)]], -0.1 * cap, 1.1 * cap)
    for name, src in experts.items():
        if src.startswith('sp:'):
            ref = q[src[3:]]; spm = P[common][:, None] * S.Ceff[fi] / S.Ceff[common][:, None]
            q[name] = ref - ref[..., MID:MID + 1] + spm[..., None]
    out['q'] = q
    Kx = np.clip(S.K, -0.5, 1.5); past = common[:, None] + np.arange(-59, 1)[None]
    out['ctx'] = dict(cz0=S.cz[common], K0=Kx[common], Kstd=Kx[past].std(1), Kmean=Kx[past].mean(1),
                      curt=b[common].astype(np.float32), agc0=a[common] / cap)
    return out
