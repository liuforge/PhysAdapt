"""Rolling-protocol training of the pooled K-space adapter on a frozen-FM base. One queue job = one protocol.
Writes runs/radapt/<base>/<tag>/<proto>/<station>.npz (fold format)."""
from pvfm.common import PVROOT
from functools import partial
from pvfm.common import STATIONS
from pvfm.dataset import build_proto, build_proto_res, build_proto_plan
from pvfm.mtrain import run_training
from pvfm.protocol import available
from pvfm.rpreds import station


def main(job, out, state):
    proto, base = job['proto'], job['base']
    pool = [s for s in STATIONS if proto == 'main' or available(station(s), proto)]
    g = dict(use_fm=1, use_hist=0, use_coarse=0, use_fut=1, max_delta=0.15, use_spread=1)
    f = dict(use_fm=1, use_hist=1, use_coarse=1, use_fut=1, max_delta=0.15, use_spread=1)
    CONF = job.get('conf') or {'pooled_fmgeo': g, 'pooled_full': f}
    if job.get('variants') == 'residual':
        CONF = {'res_own': dict(use_fm=1, use_hist=1, use_coarse=0, use_fut=1, max_delta=0.15, use_spread=1),
                'res_own_nb': dict(use_fm=1, use_hist=1, use_coarse=0, use_fut=1, max_delta=0.15, use_spread=1)}
    if job.get('variants') == 'recency':
        CONF = {'pooled_fmgeo': g, 'fmgeo_rec60': {**g, 'recency_days': 60}, 'fmgeo_rec30': {**g, 'recency_days': 30},
                'full_rec30': {**f, 'recency_days': 30}}
    specs = [dict(station='ALL', seed=sd, tag=tag, **c) for tag, c in CONF.items() for sd in job.get('seeds', (28, 29, 30))]
    if job.get('loso'):  # leave-one-station-out: train on the other stations, write outputs only for the held-out one
        held = job['loso']
        if held not in pool: return {'skipped': 'held-out station not available in this fold'}
        tr_pool = [s for s in pool if s != held]
        sp = [dict(station='ALL', seed=sd, tag='loso', **g) for sd in job.get('seeds', (28, 29, 30))]
        _, meta = run_training(sp, (PVROOT + '/data'), None, out, epochs=80, data_fn=lambda st: build_proto(st, (PVROOT + '/data'), base, proto),
                               fold_out=(f'{PVROOT}/runs/radapt/{base}', proto), pool_stations=tr_pool, extra_predict=[held], save_pool=False)
        return {k: v for k, v in meta.items() if k != 'log'}
    if job.get('plan') == 'control':  # same capacity (max_delta 0.5), no plan
        sp = [dict(station='ALL', seed=sd, tag='noplan_d05', use_fm=1, use_hist=0, use_coarse=0, use_fut=1, max_delta=0.5, use_spread=1) for sd in job.get('seeds', (28, 29, 30))]
        _, meta = run_training(sp, (PVROOT + '/data'), None, out, epochs=80,
                               data_fn=lambda st: build_proto(st, (PVROOT + '/data'), base, proto),
                               fold_out=(f'{PVROOT}/runs/radapt/{base}', proto), pool_stations=pool)
        return {k: v for k, v in meta.items() if k != 'log'}
    if job.get('plan'):  # learned integration of a simulated dispatch plan (future covariate)
        lag, sig = job['plan']; tag = f'plan_l{lag}_s{int(sig * 100)}'
        sp = [dict(station='ALL', seed=sd, tag=tag, use_fm=1, use_hist=0, use_coarse=0, use_fut=1, max_delta=0.5, use_spread=1) for sd in job.get('seeds', (28, 29, 30))]
        _, meta = run_training(sp, (PVROOT + '/data'), None, out, epochs=80,
                               data_fn=lambda st: build_proto_plan(st, (PVROOT + '/data'), base, proto, lag, sig),
                               fold_out=(f'{PVROOT}/runs/radapt/{base}', proto), pool_stations=pool)
        return {k: v for k, v in meta.items() if k != 'log'}
    if job.get('variants') == 'residual':  # two groups: own residual only / own + neighbours (different inputs)
        metas = {}
        for tag, nb in (('res_own', False), ('res_own_nb', True)):
            sp = [x for x in specs if x['tag'] == tag]
            _, metas[tag] = run_training(sp, (PVROOT + '/data'), None, out, epochs=80,
                                         data_fn=lambda st, nb=nb: build_proto_res(st, (PVROOT + '/data'), base, proto, neighbours=nb),
                                         fold_out=(f'{PVROOT}/runs/radapt/{base}', proto), pool_stations=pool)
        return {k: {kk: vv for kk, vv in v.items() if kk != 'log'} for k, v in metas.items()}
    _, meta = run_training(specs, (PVROOT + '/data'), None, out, epochs=80,
                           data_fn=lambda st: build_proto(st, (PVROOT + '/data'), base, proto),
                           fold_out=(f'{PVROOT}/runs/radapt/{base}', proto), pool_stations=pool)
    return {k: v for k, v in meta.items() if k != 'log'}
