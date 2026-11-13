"""Unified zero-shot inference for several time-series foundation models in P- or clipped-K space.
spec = {model}_{space}_c{ctx}_{cov}; model in {timesfm, tirex, toto, moirai2, sundial, bolt}; space in {P, Kc};
cov in {none, pastfut} (pastfut only for models with covariate support: toto, moirai2).
Models returning only deciles get tails (0.01/0.05/0.95/0.99) extrapolated with normal ratios; evaluation of the
cross-model table uses deciles only (eval_a column pin_dec)."""
from pvfm.common import PVROOT
import time
from pathlib import Path
import numpy as np
import torch
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station
from pvfm.fm_zero import ffill

M = Path(PVROOT + '/models')
DEC = np.round(np.arange(0.1, 0.91, 0.1), 2)
Z = {0.01: 2.3263, 0.05: 1.6449, 0.1: 1.2816}


def from_deciles(qd):
    """qd [n,H,9] deciles -> [n,H,13] on LEVELS with normal-ratio tail extrapolation."""
    med = qd[..., 4]; lo = med - qd[..., 0]; hi = qd[..., 8] - med
    out = {}
    for j, t in enumerate(DEC): out[round(float(t), 2)] = qd[..., j]
    out[0.01] = med - lo * Z[0.01] / Z[0.1]; out[0.05] = med - lo * Z[0.05] / Z[0.1]
    out[0.95] = med + hi * Z[0.05] / Z[0.1]; out[0.99] = med + hi * Z[0.01] / Z[0.1]
    return np.stack([out[round(float(t), 2)] for t in LEVELS], -1)


def from_samples(s):
    """s [n,S,H] -> [n,H,13]"""
    return np.moveaxis(np.quantile(s, LEVELS, axis=1), 0, -1)


class Wrap:
    def __init__(self, name, ctx, H):
        self.name, self.ctx, self.H = name, ctx, H
        dev = 'cuda'
        if name == 'timesfm':
            import timesfm
            self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(M / 'timesfm-2.5-200m-pytorch'))
            self.m.compile(timesfm.ForecastConfig(max_context=max(ctx, 32), max_horizon=128, normalize_inputs=True,
                                                  use_continuous_quantile_head=True, force_flip_invariance=True,
                                                  infer_is_positive=False, fix_quantile_crossing=True, per_core_batch_size=64))
        elif name == 'tirex':
            from tirex import load_model
            try:
                self.m = load_model('NX-AI/TiRex', device=dev, backend='cuda')
            except Exception:
                self.m = load_model('NX-AI/TiRex', device=dev, backend='torch')
        elif name == 'toto':
            from toto.model.toto import Toto
            from toto.inference.forecaster import TotoForecaster
            import json as _j
            cfg = _j.load(open(M / 'Toto-Open-Base-1.0' / 'config.json'))
            t = Toto._from_pretrained(model_id=str(M / 'Toto-Open-Base-1.0'), revision=None, cache_dir=None, force_download=False, proxies=None,
                                      resume_download=False, local_files_only=True, token=None, **cfg).to(dev); t.eval()
            self.m = TotoForecaster(t.model)
        elif name == 'moirai2':
            from uni2ts.model.moirai2 import Moirai2Module
            self.module = Moirai2Module.from_pretrained(str(M / 'moirai-2.0-R-small'))
        elif name == 'sundial':
            from transformers import AutoModelForCausalLM
            self.m = AutoModelForCausalLM.from_pretrained(str(M / 'sundial-base-128m'), trust_remote_code=True).to(dev).eval()
        elif name == 'bolt':
            from chronos import BaseChronosPipeline
            self.m = BaseChronosPipeline.from_pretrained(str(M / 'chronos-bolt-base'), device_map=dev)
        else:
            raise ValueError(name)

    @torch.no_grad()
    def predict(self, X, past=None, fut=None):
        """X [n,L] target context; past [n,C,L] covariates; fut [n,C2,H] known-future covariates (subset of past)."""
        n, H = len(X), self.H
        if self.name == 'timesfm':
            _, q = self.m.forecast(horizon=H, inputs=[x.astype(np.float32) for x in X])
            return from_deciles(np.asarray(q)[..., 1:10])
        if self.name == 'tirex':
            outs = []
            for a in range(0, n, 256):
                q, _ = self.m.forecast(context=torch.tensor(X[a:a + 256], dtype=torch.float32), prediction_length=H, batch_size=256)
                outs.append(q.cpu().numpy())
            return from_deciles(np.concatenate(outs))
        if self.name == 'bolt':
            outs = []
            for a in range(0, n, 128):
                q, _ = self.m.predict_quantiles(inputs=torch.tensor(X[a:a + 128], dtype=torch.float32), prediction_length=H, quantile_levels=DEC.tolist())
                outs.append(q.float().cpu().numpy())
            return from_deciles(np.concatenate(outs))
        if self.name == 'sundial':
            outs = []
            for a in range(0, n, 64):
                s = self.m.generate(torch.tensor(X[a:a + 64], dtype=torch.float32, device='cuda'), max_new_tokens=H, num_samples=100)
                outs.append(from_samples(s.float().cpu().numpy()))
            return np.concatenate(outs)
        if self.name == 'toto':
            from toto.data.util.dataset import MaskedTimeseries
            outs = []
            V = 1 + (0 if past is None else past.shape[1])
            for a in range(0, n, 16):
                x = X[a:a + 16][:, None]
                if past is not None: x = np.concatenate([x, past[a:a + 16]], 1)
                b = len(x); s = torch.tensor(x, dtype=torch.float32, device='cuda')
                inp = MaskedTimeseries(series=s, padding_mask=torch.ones_like(s, dtype=torch.bool),
                                       id_mask=torch.zeros_like(s, dtype=torch.int), timestamp_seconds=torch.zeros_like(s, dtype=torch.int),
                                       time_interval_seconds=torch.full((b, V), 60, dtype=torch.int, device='cuda'),
                                       num_exogenous_variables=0 if past is None else past.shape[1])
                fe = None if fut is None else torch.tensor(np.concatenate([fut[a:a + 16], np.zeros((b, past.shape[1] - fut.shape[1], H))], 1), dtype=torch.float32, device='cuda')
                fc = self.m.forecast(inp, prediction_length=H, num_samples=128, samples_per_batch=128, future_exogenous_variables=fe)
                smp = fc.samples[:, 0].float().cpu().numpy()  # [b,H,S]
                outs.append(from_samples(np.moveaxis(smp, -1, 1)))
            return np.concatenate(outs)
        if self.name == 'moirai2':
            from uni2ts.model.moirai2 import Moirai2Forecast
            from gluonts.dataset.common import ListDataset
            import pandas as pd
            cf = 0 if fut is None else fut.shape[1]; cp = 0 if past is None else past.shape[1] - cf
            model = Moirai2Forecast(module=self.module, prediction_length=H, context_length=X.shape[1], target_dim=1,
                                    feat_dynamic_real_dim=cf, past_feat_dynamic_real_dim=cp)
            pred = model.create_predictor(batch_size=64, device='cuda')
            items = []
            for j in range(n):
                it = {'start': pd.Period('2000-01-01 00:00', freq='min'), 'target': X[j].astype(np.float32)}
                if cf: it['feat_dynamic_real'] = np.concatenate([past[j, :cf], fut[j]], -1).astype(np.float32)
                if cp: it['past_feat_dynamic_real'] = past[j, cf:].astype(np.float32)
                items.append(it)
            fcs = list(pred.predict(ListDataset(items, freq='min', one_dim_target=True)))
            qd = np.stack([np.stack([f.quantile(float(t)) for t in DEC], -1) for f in fcs])
            return from_deciles(qd)
        raise ValueError(self.name)


def run(spec, station, data_dir, out_dir, state, splits=('validation', 'test'), H=96):
    model, space, c, cov = spec.split('_'); ctx = int(c[1:])
    key = (model, ctx)
    if state.get('key') != key:
        state.clear(); torch.cuda.empty_cache(); state['w'] = Wrap(model, ctx, H); state['key'] = key
    w = state['w']
    st = Station(Path(data_dir) / f'{station}.npz', station, STATIONS[station][1])
    tgt = np.clip(st.K, -0.5, 1.5) if space == 'Kc' else np.nan_to_num(st.x['P'])
    res = {}
    for split in splits:
        if split == 'all':
            from pvfm.protocol import all_origins
            o = all_origins(st)
        else:
            o = st.origins[split]
        o = o[o - ctx + 1 >= 0]
        ci = o[:, None] + np.arange(-ctx + 1, 1)[None]; fi = o[:, None] + np.arange(1, H + 1)[None]
        past = fut = None
        if cov == 'pastfut':
            past = np.stack([st.cz[ci], st.Ceff[ci]], 1); fut = np.stack([st.cz[fi], st.Ceff[fi]], 1)
        t = time.time(); q = w.predict(tgt[ci].astype(np.float32), past, fut); el = time.time() - t
        if space == 'Kc': q = q * st.Ceff[fi][..., None]
        q = np.maximum.accumulate(np.nan_to_num(q), -1).astype(np.float32)
        np.savez_compressed(Path(out_dir) / f'{split}.npz', q=q, origins=o, seconds=el)
        res[split] = dict(n=len(o), seconds=round(el, 2))
    return res
