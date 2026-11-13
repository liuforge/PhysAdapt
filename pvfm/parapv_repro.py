"""Reproduce PARA-PV (Fan et al., Energy 2026) on its own public data and protocol: the authors' code unchanged (native
Excel reader, 8:1:1 split, online Chronos-2 prior), all branches, the configuration of their Appendix A (L=192, 50 epochs,
lr 1e-3, batch 32, d_model 128, patch memory 4096, top-K 5). Dataset 1 = site 1 in 2019, Dataset 2 = site 6 in 2019-2020.
Metrics are the authors' own, on the standardised target scale, read from their result files.
Job: {dataset: 'D1'|'D2', mode: 'point'|'prob', pred_len}."""
import os, sys, json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd
from pvfm.common import PVROOT

ROOT = Path(PVROOT); REPO = Path(os.environ.get('PARAPV_REPO', '/mnt/user/pvfm/code/third_party/PARA-PV')); SRC = Path(os.environ.get('SGCC_XLSX_DIR', '/mnt/user/pvfm/data_ext/sgcc'))
DS = {'D1': ('Solar station site 1 (Nominal capacity-50MW).xlsx', 'Solarstation1-50MW-2019.xlsx', 50.0, '2019-12-31 23:45'),
      'D2': ('Solar station site 6 (Nominal capacity-35MW).xlsx', 'Solarstation6-35MW-20192020.xlsx', 35.0, '2020-12-31 23:45')}
PAPER = ['--train_epochs', '50', '--learning_rate', '0.001', '--batch_size', '32', '--d_model', '128', '--patch_memory_size', '4096', '--top_k', '5', '--dropout', '0.1']


def main(job, out, state):
    ds, mode, H = job['dataset'], job['mode'], int(job['pred_len']); src, name, cap, end = DS[ds]
    work = Path(out).resolve() / 'work'; work.mkdir(parents=True, exist_ok=True)
    if not (work / name).exists():
        x = pd.read_excel(SRC / src); t = pd.to_datetime(x.iloc[:, 0]); x[t <= end].to_excel(work / name, index=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PARAPV_')}
    mid = f'repro_{ds}_{mode}_{H}'
    cmd = [sys.executable, 'run.py', '--model', 'PARA-PV', '--forecast_mode', mode, '--model_id', mid, '--root_path', str(work), '--data_path', name,
           '--seq_len', '192', '--label_len', '96', '--pred_len', str(H), '--use_pa_ral', 'true', '--use_pa_ral_analog', 'true',
           '--use_chronos_prior', 'true', '--chronos_model_path', str(ROOT / 'models/chronos-2'), '--use_chronos_adapter', 'true',
           '--use_pa_dsc', 'true', '--chronos_capacity', str(cap), '--num_workers', '2', '--gpu', '0', '--checkpoints', str(work / 'ckpt')] + PAPER
    if mode == 'point': cmd += ['--point_train_loss', 'adaptive_segment_pv']
    t0 = time.time(); r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    (work / 'log.txt').write_text(r.stdout[-30000:] + '\n' + r.stderr[-20000:])
    if r.returncode != 0: raise RuntimeError(r.stderr[-3000:])
    res = next((REPO / 'results').glob(f'long_term_forecast_{mid}_PARA-PV_*'))
    out_ = dict(dataset=ds, mode=mode, pred_len=H, seconds=time.time() - t0, result_dir=res.name)
    m = np.load(res / 'metrics.npy').tolist()
    if mode == 'point': out_.update(dict(zip(['MAE', 'MSE', 'RMSE', 'R2'], m)))
    else: out_.update(dict(zip(['PICP80', 'PINAW80', 'PICP90', 'PINAW90', 'AQL'], m)))
    return out_
