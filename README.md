# PhysAdapt

Code for **"Physics-consistent adaptation of frozen time-series foundation models for ultra-short-term probabilistic
photovoltaic forecasting under dispatch-following operation"** (J. Liu, C. Peng, K. Zou, P. Huang, A. C. Tsoi;
submitted to *Renewable Energy*).

PhysAdapt keeps a time-series foundation model (TSFM) frozen and adds three light modules:

| Module | What it does | Code |
|---|---|---|
| **M1** physics-consistent input | clear-sky envelope `C_eff` (Haurwitz + fitted scale and timing offset), clear-sky index `K = P / C_eff`, multi-day context, solar geometry as a known future covariate; no gradient training | `pvfm/physics.py`, `pvfm/common.py`, `pvfm/fm_zero.py` (1 min), `pvfm/task15.py` (15 min) |
| **M2** shared output adapter | 45k-parameter adapter on the TSFM quantiles: median + bounded shift, softplus gaps (non-crossing), zero-initialised head, dilated convolutions over leads; `C_eff`-weighted pinball loss in K space | `pvfm/mtrain.py`, `pvfm/dataset.py`, `pvfm/radapt.py` (1 min), `pvfm/radapt15.py` (15 min) |
| **M3** delayed online calibration | global + per-clear-sky-stratum quantile offsets and location tracking in K space, updated only with complete windows (delay `d = ceil(H/stride)`); median-preserving outward repair; Mondrian (certified) variant | `pvfm/calib.py` |

Evaluation and baselines: `pvfm/final2.py` (1-min comparison), `pvfm/t15_full.py` (15-min comparison),
`pvfm/gridcode15.py` (2026 Shandong deviation-area penalty), `pvfm/base15.py`, `pvfm/dlbase.py`, `pvfm/lgbm_base.py`,
`pvfm/ft.py` (PatchTST, iTransformer, DLinear, TiDE, LightGBM, full fine-tuning and LoRA of Chronos-2),
`pvfm/probbase.py` (probabilistic smart persistence, CH-PeEn), `pvfm/parapv*.py` (PARA-PV comparison and reproduction).

## Data

* **Shandong SCADA data (main paper).** Seven PV stations, 1-min, May 2025 – April 2026. The data belong to the station
  operators and cannot be shared, so these experiments cannot be rerun from this repository; the code is provided for
  transparency.
* **Public data (Supplementary Section S3).** Solar sites 1 and 6 of the State Grid forecasting-competition data
  (Chen & Xu, *Scientific Data* 9, 577, 2022; CC0), the data used by PARA-PV. Download
  `Solar station site 1 (Nominal capacity-50MW).xlsx` and `Solar station site 6 (Nominal capacity-35MW).xlsx` from
  [Figshare](https://doi.org/10.6084/m9.figshare.17304221). Everything below runs on these data.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PVFM_ROOT=$PWD/workspace          # data, models and runs live here
mkdir -p $PVFM_ROOT/models
# Chronos-2 weights (Hugging Face: amazon/chronos-2) into $PVFM_ROOT/models/chronos-2
huggingface-cli download amazon/chronos-2 --local-dir $PVFM_ROOT/models/chronos-2
```

## Reproducing the public-data results (Supplementary Tables S6 and S7)

```bash
export PVFM_ROOT=$PWD/workspace PVFM_DATASET=sgcc PYTHONPATH=$PWD
export SGCC_XLSX_DIR=/path/to/the/two/xlsx/files

# 1. data: 15-min values expanded to 1 min, 8:1:1 split, coordinates fitted from the day length
python -m pvfm.prep_sgcc $SGCC_XLSX_DIR $PVFM_ROOT/data

# 2. PARA-PV (only for the comparison): clone the authors' code and apply our two-hunk patch
#    (CSV input and fold boundaries; without the environment variables it behaves exactly as the original)
git clone https://github.com/weican1103/PARA-PV.git $PVFM_ROOT/code/third_party/PARA-PV
git -C $PVFM_ROOT/code/third_party/PARA-PV apply $PWD/patches/parapv_data_loader.patch
export PARAPV_REPO=$PVFM_ROOT/code/third_party/PARA-PV

# 3. experiments: a simple file-based job queue; start one worker per GPU (several machines may share PVFM_ROOT)
python scripts/make_jobs_public.py
CUDA_VISIBLE_DEVICES=0 python -m pvfm.queue $PVFM_ROOT/runs/q/jobs_p1.jsonl $PVFM_ROOT/runs/q/p1 w0
python -m pvfm.base15                                    # average the three seeds of the deep baselines
CUDA_VISIBLE_DEVICES=0 python -m pvfm.queue $PVFM_ROOT/runs/q/jobs_p2.jsonl $PVFM_ROOT/runs/q/p2 w0

# 4. evaluation (CPU): MAE in MW and on PARA-PV's standardised scale, pinball, QCE, coverage, bootstrap CIs
PVFM_NO_TORCH=1 python -m pvfm.sgcc_eval $PVFM_ROOT/runs/eval
```

The reproduction jobs (`repro_*`) run PARA-PV unchanged in its own protocol and write its own metrics to
`$PVFM_ROOT/runs/q/p1/repro_*/result.json`. They are slow because PARA-PV evaluates Chronos-2 inside every training
batch; remove them from `jobs_p1.jsonl` if only Table S7 is needed.

## Main results

On seven Shandong stations (five rolling folds across the 2026 switch to dispatch-following operation), PhysAdapt on a
frozen Chronos-2 matches full fine-tuning of the same backbone on the 1-min task (MAE +0.5 %, n.s.) with half its
calibration error and 1/886 of its trainable parameters, beats supervised models by 17–26 % and PARA-PV by 28 %
(15-min task). On PARA-PV's public data it has the lowest MAE of all methods (−30 % vs. PARA-PV; standardised MAE
0.1034 / 0.0821 against the 0.1343 / 0.0977 published for PARA-PV).

## License

MIT (see `LICENSE`). PARA-PV is not redistributed here; it is obtained from its authors' repository.
