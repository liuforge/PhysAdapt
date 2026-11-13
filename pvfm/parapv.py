"""Direct comparison with PARA-PV (Fan et al., Energy 2026, doi:10.1016/j.energy.2026.143011) on the 15-min / 0-4 h task, rolling folds.
Runs the authors' code (third_party/PARA-PV, two documented patches: CSV input and fold boundaries) with all branches
on (PA-RAL + analog retrieval, frozen Chronos-2 prior, Chronos residual adapter, PA-DSC), probabilistic mode, their
default hyperparameters, pred_len=16. The Chronos-2 prior is precomputed with the same inputs their branch builds
(normalised window: target + past covariates, median), using their cache option. One queue job = (station, fold).
Output: runs/rbase15/PARA_PV/<proto>/<station>.npz with o_test, q_test (levels 0.05,0.1,0.5,0.9,0.95; MW)."""
import os, sys, json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd
from pvfm.common import STATIONS
from pvfm.fm_zero import COVS
from pvfm.task15 import Station15, splits15, H15

ROOT = Path(os.environ.get('PVFM_ROOT', '/mnt/user/pvfm')); REPO = Path(os.environ.get('PARAPV_REPO', str(ROOT / 'code/third_party/PARA-PV')))
SEQ, LAB = 192, 96
LEVELS5 = np.array([0.05, 0.1, 0.5, 0.9, 0.95])


def _prior(work, cache):
    """Chronos-2 median prior on the authors' normalised windows (target + past covariates), saved in their cache format."""
    if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
    from data_provider.data_loader import Dataset_Custom
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(str(ROOT / 'models/chronos-2'), device_map='cuda' if torch.cuda.is_available() else 'cpu')

    class A: pass
    args = A(); args.augmentation_ratio = 0; args.use_chronos_prior_cache = True
    t0 = time.time()
    for flag in ('train', 'val', 'test'):
        ds = Dataset_Custom(args, str(work), flag=flag, size=[SEQ, LAB, H15], features='MS', data_path='Solarstation1-50MW-2019.xlsx',
                            target='power', timeenc=1, freq='15min')
        inputs = []
        for i in range(len(ds)):
            x = ds.data_x[i:i + SEQ]
            inputs.append({'target': x[:, -1].astype(np.float32), 'past_covariates': {f'cov_{k}': x[:, k].astype(np.float32) for k in range(x.shape[1] - 1)}})
        q, _ = pipe.predict_quantiles(inputs, prediction_length=H15, quantile_levels=[0.5], batch_size=512)
        prior = np.stack([z[0].float().cpu().numpy() for z in q])[..., :1].astype(np.float32)  # (n, H, 1)
        np.save(cache / f'chronos_prior_{flag}.npy', prior)
    del pipe; torch.cuda.empty_cache()
    return time.time() - t0


def main(job, out, state):
    st, proto = job['station'], job['proto']; S = Station15(ROOT / 'data' / f'{st}.npz', st, STATIONS[st][1])
    sp = {k: np.sort(v) for k, v in splits15(S, proto).items()}
    tr_end = int(sp['es'].min()) + 1; va_end = int(sp['test'].min()) + 1; end = int(sp['test'].max()) + H15 + 1
    work = Path(out).resolve() / 'work'; work.mkdir(parents=True, exist_ok=True)
    cols = [k for k in COVS if k in S.x]
    df = pd.DataFrame({'date': pd.to_datetime(S.minutes[:end].astype(np.int64), unit='m')})
    for k in cols: df[k] = np.nan_to_num(S.x[k][:end])
    df['power'] = S.P[:end]
    df.to_csv(work / 'pv.csv', index=False)
    env = dict(os.environ, PARAPV_CSV=str(work / 'pv.csv'), PARAPV_BORDERS=f'{tr_end},{va_end}')
    os.environ.update(PARAPV_CSV=env['PARAPV_CSV'], PARAPV_BORDERS=env['PARAPV_BORDERS'])
    # ---- prior cache with the authors' dataset objects (same normalisation and windows)
    cache = work / 'prior'; cache.mkdir(exist_ok=True)
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        t_prior = _prior(work, cache)
    finally:
        os.chdir(cwd)
    # ---- authors' training and test, unchanged
    cmd = [sys.executable, 'run.py', '--model', 'PARA-PV', '--forecast_mode', 'prob', '--model_id', f'{st}_{proto}' + job.get('tag_suffix', ''),
           '--root_path', str(work), '--data_path', 'Solarstation1-50MW-2019.xlsx', '--seq_len', str(SEQ), '--label_len', str(LAB), '--pred_len', str(H15),
           '--use_pa_ral', 'true', '--use_pa_ral_analog', 'true', '--use_chronos_prior', 'true', '--use_chronos_adapter', 'true',
           '--use_chronos_prior_cache', 'true', '--chronos_prior_cache_dir', str(cache), '--use_pa_dsc', 'true',
           '--chronos_capacity', str(S.cap), '--num_workers', '2', '--gpu', '0', '--checkpoints', str(work / 'ckpt')] + list(job.get('args', []))
    t1 = time.time(); r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    (work / 'log.txt').write_text(r.stdout[-20000:] + '\n' + r.stderr[-20000:])
    if r.returncode != 0: raise RuntimeError(r.stderr[-3000:])
    setting = f"long_term_forecast_{st}_{proto}{job.get('tag_suffix', '')}_PARA-PV_custom_ftMS_sl{SEQ}_ll{LAB}_pl{H15}_dm128_0"
    res = REPO / 'results' / setting
    # their test keeps the standardised scale (no --inverse); invert with the train-split scaler of the target, as their loader fits it
    pw = df['power'].to_numpy(np.float64)[:tr_end]; mu, sd = pw.mean(), pw.std()
    qp = np.load(res / 'quantile_pred.npy')[:, :, -1, :] * sd + mu  # (n, H, 5) MW
    lv = np.load(res / 'quantile_levels.npy'); tr = np.load(res / 'true.npy')[:, :, -1] * sd + mu
    n = qp.shape[0]; o = (va_end - 1) + np.arange(n)
    keep = np.isin(o, sp['test'])
    od = ROOT / 'runs/rbase15' / job.get('tag', 'PARA_PV') / proto; od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f'{st}.npz', o_test=o[keep], q_test=np.sort(qp[keep], -1).astype(np.float32), levels=lv, y_check=tr[keep].astype(np.float32))
    fi = o[keep][:, None] + np.arange(1, H15 + 1)[None]
    align_err = float(np.abs(tr[keep] - S.P[fi]).max())
    return dict(n_test=int(keep.sum()), prior_s=t_prior, train_test_s=time.time() - t1, align_max_abs_err=align_err)
