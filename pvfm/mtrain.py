"""Multi-model GPU-resident trainer for the K-space foundation-model adapter.

All models (stations x seeds x ablation configs) share one process. Every learnable tensor carries a leading model
axis M; convolutions are grouped (groups=M), linears are batched einsums. AdamW is element-wise, so this is exactly
equivalent to M independent trainings, without per-job process start-up, DataLoaders or host-device copies.

Splits: train -> fit; validation first half (time) -> early stopping; validation second half -> calibration; test -> score only.
"""
import json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn
from pvfm.common import LEVELS, STATIONS, score
from pvfm.dataset import build, build15, HIST_F, COARSE_F

Q = len(LEVELS); MID = int(np.argmin(np.abs(LEVELS - 0.5)))
TAU = torch.tensor(LEVELS, dtype=torch.float32)


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class MLin(nn.Module):
    def __init__(self, M, i, o, zero=False):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(M, i, o) if zero else torch.randn(M, i, o) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(M, o))

    def forward(self, x):  # x [M, ..., i]
        sh = x.shape
        y = torch.bmm(x.reshape(sh[0], -1, sh[-1]), self.w) + self.b[:, None]
        return y.reshape(*sh[:-1], -1)


class MConv(nn.Module):
    """Per-model dilated 1-D conv as one grouped conv (groups=M). Benchmarked: 4.5 ms fwd+bwd under bf16 at M=93,B=256,T=96,d=32."""

    def __init__(self, M, ci, co, k=3, dil=1):
        super().__init__()
        self.M, self.co = M, co
        self.conv = nn.Conv1d(M * ci, M * co, k, dilation=dil, padding=dil * (k - 1) // 2, groups=M)

    def forward(self, x):  # x [M, B, T, C]
        M, B, T, C = x.shape
        y = self.conv(x.permute(1, 0, 3, 2).reshape(B, M * C, T))
        return y.reshape(B, M, self.co, T).permute(1, 0, 3, 2)


class Encoder(nn.Module):
    def __init__(self, M, ci, d, dils):
        super().__init__()
        self.inp = MConv(M, ci, d, 3, 1)
        self.blocks = nn.ModuleList([MConv(M, d, d, 3, g) for g in dils])
        self.out = MLin(M, 2 * d, d)

    def forward(self, x):
        h = Fn.gelu(self.inp(x))
        for b in self.blocks:
            h = h + Fn.gelu(b(h))
        return Fn.gelu(self.out(torch.cat([h.mean(2), h[:, :, -1]], -1)))


class MultiAdapter(nn.Module):
    """Per model: q_K = base median + bounded delta; spread = cumulative softplus increments initialised at base gaps."""

    def __init__(self, M, d=32, nfut=3, max_delta=0.5, nhist=None, nfut_in=None):
        super().__init__()
        self.register_buffer('max_delta', torch.full((M,), float(max_delta)))
        self.register_buffer('spread', torch.ones(M))
        self.hist = Encoder(M, nhist or len(HIST_F), d, (2, 4, 8, 16))
        self.coarse = Encoder(M, len(COARSE_F), d, (2, 4, 8))
        self.ctx = MLin(M, 2 * d, d)
        self.film = MLin(M, d, 2 * d, zero=True)
        self.hin = MLin(M, Q + 2 + (nfut_in or nfut), d)
        self.hblocks = nn.ModuleList([MConv(M, d, d, 3, g) for g in (1, 2, 4, 8)])
        self.head = MLin(M, d, Q, zero=True)

    def forward(self, base, klast, hist, coarse, fut, mask):
        # base [M,B,H,Q] sorted; mask [M,4] = use_fm, use_hist, use_coarse, use_fut
        m = mask[:, None, None, :]
        hist = hist * m[..., 1:2]; coarse = coarse * m[..., 2:3]; fut = fut * m[..., 3:4]
        z = Fn.gelu(self.ctx(torch.cat([self.hist(hist), self.coarse(coarse)], -1)))  # [M,B,d]
        med = base[..., MID:MID + 1]
        x = torch.cat([(base - med) * m[..., 0:1], med - klast[..., None, None], klast[..., None, None].expand_as(med), fut], -1)
        h = Fn.gelu(self.hin(x))
        g, b = self.film(z).chunk(2, -1)
        h = h * (1 + g[:, :, None]) + b[:, :, None]
        for blk in self.hblocks:
            h = h + Fn.gelu(blk(h))
        o = self.head(h)  # [M,B,H,Q]
        median = med[..., 0] + self.max_delta.view(-1, 1, 1) * torch.tanh(o[..., MID])
        sp = self.spread.view(-1, 1, 1, 1)
        gaps = (base[..., 1:] - base[..., :-1]).clamp_min(1e-3)
        lo = Fn.softplus(sp * o[..., :MID] + inv_softplus(gaps[..., :MID]))            # increments below median
        hi = Fn.softplus(sp * o[..., MID + 1:] + inv_softplus(gaps[..., MID:]))       # increments above median
        lower = median[..., None] - torch.flip(torch.cumsum(torch.flip(lo, [-1]), -1), [-1])
        upper = median[..., None] + torch.cumsum(hi, -1)
        return torch.cat([lower, median[..., None], upper], -1)


def mview(t, M):
    """View a parameter with its model axis first: MLin params are [M,...]; grouped conv params are [M*co,...]."""
    if t.shape[0] == M:
        return t
    return t.view(M, t.shape[0] // M, *t.shape[1:])


def reinit(model, specs):
    """Initialise every model slice from its own seed so results do not depend on which models share a batch."""
    M = len(specs)
    with torch.no_grad():
        for pi, (name, p) in enumerate(model.named_parameters()):
            v = mview(p, M)
            for m, s in enumerate(specs):
                if name.endswith('.b') or name.endswith('bias') or float(v[m].abs().sum()) == 0:
                    continue
                g = torch.Generator().manual_seed(int(s['seed']) * 7919 + pi)
                fan_in = v[m][0].numel() if name.endswith('conv.weight') else v[m].shape[0]
                v[m].copy_(torch.randn(v[m].shape, generator=g) / math.sqrt(fan_in))


def pinball_w(y, q, w, tau):
    e = y[..., None] - q
    return (torch.maximum(tau * e, (tau - 1) * e) * w[..., None]).mean((-1, -2, -3))  # per model


class Bank:
    """Station-level tensors on GPU, indexed by model -> station map."""

    def __init__(self, parts, keys, dev):
        n = max(len(p['o']) for p in parts)
        self.n = torch.tensor([len(p['o']) for p in parts], device=dev)
        self.t = {}
        for k in keys:
            arr = np.zeros((len(parts), n) + parts[0][k].shape[1:], np.float32)
            for i, p in enumerate(parts):
                arr[i, :len(p['o'])] = p[k]
            self.t[k] = torch.from_numpy(arr).to(dev)

    def get(self, st_of_m, idx):
        return {k: v[st_of_m[:, None], idx] for k, v in self.t.items()}


def split_val(d, h=None):
    n = len(d['o']); h = n // 2 if h is None else h
    pick = lambda a, s: {k: (v[s] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v) for k, v in a.items()}
    return pick(d, slice(0, h)), pick(d, slice(h, n))


def base_of(b, mask):
    """FM quantiles if use_fm else smart-persistence base (K_last flat) with unit spread seed."""
    fm = b['qK']
    sp = b['Klast'][..., None, None].expand_as(fm) + torch.tensor(np.linspace(-0.3, 0.3, Q), device=fm.device, dtype=fm.dtype)
    u = mask[:, 0][:, None, None, None]
    return u * fm + (1 - u) * sp


def run_training(specs, data_dir, fm_dir, out_dir, epochs=80, batch=256, lr=1e-3, wd=1e-4, patience=10, d=32, dev='cuda', amp=True, steps_per_epoch=None, task='1min', data_fn=None, fold_out=None, pool_stations=None, extra_predict=None, save_pool=True):
    """specs: list of dict(station, seed, use_fm, use_hist, use_coarse, use_fut, tag).
    extra_predict: stations not used for training whose fold outputs are written (leave-one-station-out); save_pool=False
    writes only those."""
    t_start = time.time()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    real = sorted({s['station'] for s in specs if s['station'] != 'ALL'} | (set(pool_stations or STATIONS) if any(s['station'] == 'ALL' for s in specs) else set()))
    builder = build15 if task == '15min' else build
    data = {st: (data_fn(st) if data_fn else builder(st, data_dir, fm_dir)) for st in real}
    stations = real + (['ALL'] if any(s['station'] == 'ALL' for s in specs) else [])
    if 'ALL' in stations:
        cat = lambda parts: {k: (np.concatenate([p[k] for p in parts]) if isinstance(parts[0][k], np.ndarray) and parts[0][k].ndim else parts[0][k]) for k in parts[0]}
        data['ALL'] = {'train': cat([data[s]['train'] for s in real]),
                       'validation': cat([split_val(data[s]['validation'], data[s].get('n_es'))[0] for s in real]), 'test': None}
    t_data = time.time() - t_start
    keys = ['qK', 'Klast', 'hist', 'coarse', 'fut', 'yP', 'Ceff']
    tr = Bank([data[s]['train'] for s in stations], keys, dev)
    es_parts, ca_parts = zip(*[split_val(data[s]['validation'], data[s].get('n_es')) if s != 'ALL' else (data['ALL']['validation'], None) for s in stations])
    es = Bank(list(es_parts), keys, dev)
    M = len(specs)
    st_of_m = torch.tensor([stations.index(s['station']) for s in specs], device=dev)
    mask = torch.tensor([[s['use_fm'], s['use_hist'], s['use_coarse'], s['use_fut']] for s in specs], dtype=torch.float32, device=dev)
    model = MultiAdapter(M, d, nhist=tr.t['hist'].shape[-1], nfut_in=tr.t['fut'].shape[-1]).to(dev)
    model.max_delta.copy_(torch.tensor([float(s.get('max_delta', 0.5)) for s in specs]))
    model.spread.copy_(torch.tensor([float(s.get('use_spread', 1)) for s in specs]))
    reinit(model, specs)
    tau = TAU.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device=dev).manual_seed(1234)
    best = torch.full((M,), float('inf'), device=dev); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = torch.zeros(M, device=dev); active = torch.ones(M, device=dev)
    steps = steps_per_epoch or int(math.ceil(tr.n.max().item() / batch)); hist_log = []
    # sampling weights: exp(-(o_last - o)/tau) within each station's training set (tau from spec 'recency_days')
    nmax = int(tr.n.max()); Wt = torch.zeros(len(specs), nmax, device=dev)
    for m, s_ in enumerate(specs):
        part = data[s_['station']]['train']; o = torch.tensor(part.get('ot', part['o']) - np.max(part.get('ot', part['o'])), dtype=torch.float32, device=dev)
        rec_min = float(s_.get('recency_days', 0)) * 1440  # do NOT reuse the name `tau` (quantile levels)
        Wt[m, :len(o)] = torch.exp(-(o.max() - o) / rec_min) if rec_min > 0 else 1.0

    def forward(bank, idx):
        b = bank.get(st_of_m, idx)
        q = model(base_of(b, mask), b['Klast'], b['hist'], b['coarse'], b['fut'], mask)
        return q, b

    def es_loss():
        model.eval(); tot = torch.zeros(M, device=dev); cnt = torch.zeros(M, device=dev)
        with torch.no_grad():
            nmax = int(es.n.max())
            for a in range(0, nmax, 512):
                idx = torch.arange(a, min(a + 512, nmax), device=dev)[None].expand(M, -1)
                valid = (idx < es.n[st_of_m][:, None]).float()
                q, b = forward(es, idx.clamp_max(es.t['yP'].shape[1] - 1))
                e = (b['yP'] / b['Ceff'])[..., None] - q
                l = (torch.maximum(tau * e, (tau - 1) * e) * b['Ceff'][..., None]).mean((-1, -2))
                tot += torch.where(valid > 0, l, torch.zeros_like(l)).sum(1); cnt += valid.sum(1)
        model.train(); return tot / cnt

    for ep in range(epochs):
        t0 = time.time()
        for _ in range(steps):
            if os.environ.get('PVFM_UNIFORM'):
                idx = (torch.rand(M, batch, generator=gen, device=dev) * tr.n[st_of_m][:, None]).long()
            else:
                idx = torch.multinomial(Wt, batch, replacement=True, generator=gen)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                q, b = forward(tr, idx)
            loss = pinball_w(b['yP'] / b['Ceff'], q.float(), b['Ceff'], tau)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        v = es_loss()
        imp = (v < best - 1e-6) & (active > 0)
        best = torch.where(imp, v, best); bad = torch.where(imp, torch.zeros_like(bad), bad + active)
        with torch.no_grad():
            for k, t in model.state_dict().items():
                tv, bv = mview(t, M), mview(best_state[k], M)
                bv.copy_(torch.where(imp.view(-1, *[1] * (tv.dim() - 1)), tv, bv))
        active = (bad < patience).float()
        hist_log.append(dict(epoch=ep + 1, seconds=time.time() - t0, active=int(active.sum()), es_mean=float(v.mean())))
        print(f'epoch {ep + 1} {time.time() - t0:.2f}s active={int(active.sum())} es={float(v.mean()):.4f}', flush=True)
        if active.sum() == 0:
            break
    model.load_state_dict(best_state)
    t_train = time.time() - t_start - t_data
    if fold_out is not None and save_pool:
        save_fold(model, specs, real, data, dev, mask, fold_out)
    if fold_out is not None and extra_predict:
        for st in extra_predict:
            save_fold(model, specs, [st], {st: data_fn(st)}, dev, mask, fold_out)
        results = []
    else:
        results = evaluate(model, specs, real, data, dev, mask, out_dir)
    meta = dict(M=M, stations=stations, seconds_data=t_data, seconds_train=t_train, epochs_run=len(hist_log),
                steps_per_epoch=steps, log=hist_log, n_params_per_model=sum(p.numel() for p in model.parameters()) // M)
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1))
    return results, meta


def predict(model, parts, stations, st_of_m, mask, dev):
    """Batched multi-model prediction; each model on its own station's split. Returns list of P-space quantiles."""
    bank = Bank(list(parts), ['qK', 'Klast', 'hist', 'coarse', 'fut', 'Ceff'], dev)
    M = len(st_of_m); nmax = int(bank.n.max()); outs = []
    model.eval()
    with torch.no_grad():
        for a in range(0, nmax, 512):
            idx = torch.arange(a, min(a + 512, nmax), device=dev)[None].expand(M, -1)
            b = bank.get(st_of_m, idx)
            q = model(base_of(b, mask), b['Klast'], b['hist'], b['coarse'], b['fut'], mask) * b['Ceff'][..., None]
            outs.append(q.cpu().numpy())
    q = np.concatenate(outs, 1)
    n = bank.n[st_of_m].cpu().numpy()
    return [q[m, :n[m]] for m in range(M)]


def regime(Ceff, cap):
    r = Ceff / cap
    return np.where(r <= 0.0201, 0, np.where(r < 0.3, 1, 2))  # night / low-sun / day


def calibrate(q_cal, y_cal, g_cal, q, g):
    """Regime-conditional, median-locked split-conformal shift per level (Mondrian)."""
    out = q.copy()
    for r in np.unique(g):
        sel = g_cal == r
        for j, t in enumerate(LEVELS):
            if j == MID or sel.sum() < 50: continue
            delta = np.quantile(y_cal[sel] - q_cal[sel][:, j], t)
            out[..., j] = np.where(g == r, out[..., j] + delta, out[..., j])
    lower = np.minimum.accumulate(out[..., :MID + 1][..., ::-1], -1)[..., ::-1]
    upper = np.maximum.accumulate(out[..., MID:], -1)
    return np.concatenate([lower[..., :MID], out[..., MID:MID + 1], upper[..., 1:]], -1)


def evaluate(model, specs, real, data, dev, mask, out_dir):
    """Each model is scored on every station it applies to: its own station, or all stations for pooled ('ALL') models."""
    te_parts = [data[s]['test'] for s in real]
    ca_parts = [split_val(data[s]['validation'])[1] for s in real]
    va_parts = [data[s]['validation'] for s in real]
    rows = []
    for si, st in enumerate(real):
        sel = [m for m, s in enumerate(specs) if s['station'] in (st, 'ALL')]
        if not sel: continue
        som = torch.tensor([si] * len(specs), device=dev)
        q_te_all = predict(model, te_parts, real, som, mask, dev)
        q_ca_all = predict(model, ca_parts, real, som, mask, dev)
        q_va_all = predict(model, va_parts, real, som, mask, dev)
        te, ca = te_parts[si], ca_parts[si]; cap = te['cap']
        y, yc = te['yP'], ca['yP']; g, gc = regime(te['Ceff'], cap), regime(ca['Ceff'], cap)
        fm, fmc = te['qK'] * te['Ceff'][..., None], ca['qK'] * ca['Ceff'][..., None]
        ref = {f'fm_{k}': v for k, v in score(y, fm, LEVELS, cap).items()}
        ref.update({f'fmcal_{k}': v for k, v in score(y, calibrate(fmc, yc, gc, fm, g), LEVELS, cap).items()})
        ref['sp_mae'] = float(np.abs(y - te['sp']).mean())
        for m in sel:
            s = specs[m]
            qc = calibrate(q_ca_all[m], yc, gc, q_te_all[m], g)
            r = dict(s, eval_station=st, **ref)
            for name, qq in (('uncal', q_te_all[m]), ('cal', qc)):
                r.update({f'{name}_{k}': v for k, v in score(y, qq, LEVELS, cap).items()})
            rows.append(r)
            np.savez_compressed(Path(out_dir) / f"pred_{s['tag']}_{st}_{s['seed']}.npz", q=qc.astype(np.float32),
                                q_uncal=q_te_all[m].astype(np.float32), o=te['o'],
                                q_val=q_va_all[m].astype(np.float32), o_val=va_parts[si]['o'])
    (Path(out_dir) / 'results.json').write_text(json.dumps(rows, indent=1))
    return rows


def save_fold(model, specs, real, data, dev, mask, fold_out):
    """fold format: <fold_out>/<tag>/<station>.npz with o_/q_ for es, cal, test; seeds of the same tag averaged."""
    parts = {}
    for si, st in enumerate(real):
        es, ca = split_val(data[st]['validation'], data[st].get('n_es'))
        parts[st] = {'es': es, 'cal': ca, 'test': data[st]['test']}
    tags = sorted({s['tag'] for s in specs})
    for si, st in enumerate(real):
        som = torch.tensor([si] * len(specs), device=dev)
        preds = {p: predict(model, [parts[x][p] for x in real], real, som, mask, dev) for p in ('es', 'cal', 'test')}
        for tag in tags:
            ms = [m for m, s in enumerate(specs) if s['tag'] == tag and s['station'] in (st, 'ALL')]
            if not ms: continue
            out = {}
            for p in ('es', 'cal', 'test'):
                out[f'o_{p}'] = parts[st][p]['o']; out[f'q_{p}'] = np.mean([preds[p][m] for m in ms], 0).astype(np.float32)
            od = Path(fold_out[0]) / tag / fold_out[1]; od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f'{st}.npz', **out)


def main(job, out, state):
    """Queue entry: job = {kind: python, module: pvfm.mtrain, specs: [...], data, fm_dir, ...}."""
    rows, meta = run_training(job['specs'], job['data'], job['fm_dir'], out, epochs=job.get('epochs', 80),
                              batch=job.get('batch', 256), d=job.get('d', 32))
    return dict(meta={k: v for k, v in meta.items() if k != 'log'}, n=len(rows))
