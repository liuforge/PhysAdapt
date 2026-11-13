"""Rolling-protocol training of the pooled K-space adapter for the 15-min / 0-4 h task (frozen Chronos-2, Kc c672).
Writes runs/radapt15/<base>/pooled_fmgeo/<proto>/<station>.npz (fold format)."""
from pvfm.common import PVROOT
from pvfm.common import STATIONS
from pvfm.dataset import build15_proto
from pvfm.mtrain import run_training
from pvfm.task15 import Station15, splits15
from pvfm.preds import ROOT


def usable(st, proto):
    if proto == 'main': return True
    S = Station15(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1]); sp = splits15(S, proto)
    return len(sp['test']) > 50 and len(sp['cal']) > 50 and len(sp['es']) > 50 and len(sp['train']) >= 200


def main(job, out, state):
    proto, base = job['proto'], job.get('base', 'Kc_c672_pastfut')
    pool = [s for s in (job.get('stations') or STATIONS) if usable(s, proto)]
    g = dict(use_fm=1, use_hist=0, use_coarse=0, use_fut=1, max_delta=0.15, use_spread=1)
    conf = job.get('conf') or {'pooled_fmgeo': {}}
    specs = [dict(station='ALL', seed=sd, tag=tag, **{**g, **c}) for tag, c in conf.items() for sd in job.get('seeds', (28, 29, 30))]
    _, meta = run_training(specs, (PVROOT + '/data'), None, out, epochs=80, task='15min',
                           data_fn=lambda st: build15_proto(st, (PVROOT + '/data'), base, proto, job.get('target', 'P')),
                           fold_out=(f"{PVROOT}/runs/radapt15/{base}{job.get('suffix', '')}", proto), pool_stations=pool)
    return {k: v for k, v in meta.items() if k != 'log'}
