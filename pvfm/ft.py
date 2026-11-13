"""Weight adaptation baseline: Chronos-2 LoRA / full fine-tuning in the SAME representation (clipped K, geometry
covariates known into the future). Fine-tune on the training period, select on validation, then forecast val/test
exactly like the frozen model so all downstream modules can consume it as an 'fm:' expert."""
from pvfm.common import PVROOT
import json, time
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS
from pvfm.physics import Station
from pvfm.fm_zero import ffill, COVS, run as fm_run

ROOT = Path(PVROOT)


def series(st_name, end, space='Kc'):
    """space 'Kc': M1 representation (clipped K + geometry). space 'P': geometry-blind power space, met past covariates only."""
    cap = STATIONS[st_name][1]
    S = Station(ROOT / 'data' / f'{st_name}.npz', st_name, cap)
    pc = {k: ffill(S.x[k])[:end] for k in COVS if k in S.x}
    if space == 'P':
        return S, dict(target=np.nan_to_num(S.x['P'])[:end].astype(np.float32), past_covariates=pc)
    pc.update(cz=S.cz[:end], cs=S.Ceff[:end])
    return S, dict(target=np.clip(S.K, -0.5, 1.5)[:end], past_covariates=pc)


def main(job, out, state):
    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts
    import torch
    torch.manual_seed(job.get('seed', 0))
    scope = job['scope']  # station name or 'ALL'
    names = list(STATIONS) if scope == 'ALL' else [scope]
    tr, va = [], []
    proto = job.get('proto'); space = job.get('space', 'Kc')
    spec = f"Kc_c{job['ctx']}_pastfut" if space == 'Kc' else f"P_c{job['ctx']}_past"
    if proto and proto != 'main':
        from pvfm.protocol import available
        from pvfm.rpreds import station as _station
        names = [s for s in names if available(_station(s), proto)]
    for s in names:
        S, _ = series(s, None)
        if proto:
            from pvfm.protocol import splits as _sp
            sp = _sp(S, proto); tr_end = int(sp['es'].min()) - 96; va_end = int(sp['test'].min()) - 1
        else:
            tr_end = int(S.origins['validation'].min()) - 96
            va_end = int(S.origins['test'].min()) - 1
        # covariate schemas differ across stations: keep only the shared covariates when pooling
        tr.append(series(s, tr_end, space)[1]); va.append(series(s, va_end, space)[1])
    if scope == 'ALL':
        keys = set.intersection(*[set(d['past_covariates']) for d in tr])
        for d in tr + va:
            d['past_covariates'] = {k: v for k, v in d['past_covariates'].items() if k in keys}
    kn = ['cz', 'cs'] if space == 'Kc' else []
    tr_in = from_list_of_dicts(tr, 96, known_covariates_names=kn)
    va_in = from_list_of_dicts(va, 96, known_covariates_names=kn)
    base = Chronos2Pipeline.from_pretrained(str(ROOT / 'models/chronos-2'), device_map='cuda')
    t = time.time()
    ft = base.fit(tr_in, prediction_length=96, validation_inputs=va_in, finetune_mode=job['mode'],
                  context_length=job['ctx'], learning_rate=job['lr'], num_steps=job['steps'], batch_size=job['batch'],
                  output_dir=str(Path(out) / 'ckpt'), remove_printer_callback=True)
    fit_s = time.time() - t
    ft.model.eval()
    res = {}
    if proto:
        import tempfile
        from pvfm.protocol import splits as _sp
        from pvfm.rpreds import station as _station
        for s in names:
            sp = _sp(_station(s), proto); tmp = Path(tempfile.mkdtemp())
            fm_run(spec, s, str(ROOT / 'data'), str(tmp), ft, splits=('es', 'cal', 'test'),
                   cov_subset=(sorted(keys) if scope == 'ALL' else None), origins_map={k: sp[k] for k in ('es', 'cal', 'test')})
            outd = {}
            for part in ('es', 'cal', 'test'):
                z = np.load(tmp / f'{part}.npz'); outd[f'o_{part}'] = z['origins']; outd[f'q_{part}'] = z['q']
            od = ROOT / 'runs/rbase' / job['tag'] / proto; od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f'{s}.npz', **outd); res[s] = len(outd['o_test'])
        return dict(fit_seconds=fit_s, trainable_params=sum(p.numel() for p in ft.model.parameters() if p.requires_grad), predict=res)
    spec_dir = ROOT / 'runs/stageA' / job['tag']
    for s in names:
        o = spec_dir / s; o.mkdir(parents=True, exist_ok=True)
        res[s] = fm_run(spec, s, str(ROOT / 'data'), str(o), ft, splits=('validation', 'test'),
                        cov_subset=(sorted(keys) if scope == 'ALL' else None))
    n_tr = sum(p.numel() for p in ft.model.parameters() if p.requires_grad)
    return dict(fit_seconds=fit_s, trainable_params=n_tr, predict=res,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
