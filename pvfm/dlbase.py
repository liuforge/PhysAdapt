"""Supervised deep baselines with the SAME information as the FM pipeline, trained in clear-sky-index space with the
C_eff-weighted pinball loss (Proposition 1), pooled over the 7 stations (station embedding).
Models: PatchTST (channel-independent patches on K + exogenous head), iTransformer (variate tokens incl. future-known
covariates), DLinear (trend/remainder linear + exogenous head), TiDE (dense encoder-decoder with past/future covariates).
Training origins every 15 min inside the training period (denser than the FM caches: favours the baselines).
Writes runs/stageA/DL_<model>_s<seed>/<station>/{validation,test}.npz (P-space quantiles)."""
from pvfm.common import PVROOT
import json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn
from pvfm.common import STATIONS, LEVELS
from pvfm.physics import Station
from pvfm.fm_zero import ffill

ROOT = Path(PVROOT); H = 96; L = 2880; Q = len(LEVELS); MID = int(np.argmin(np.abs(LEVELS - 0.5)))
PAST = ['K', 'kt', 'agcn', 'curt', 'cz']; FUT = ['cz', 'cn']
COVW = 60  # recent-covariate summary window (steps): 1 h at 1 min; base15 sets 4 (1 h at 15 min)


def station_arrays(S):
    cap = S.cap
    ghi = np.clip(ffill(S.x['ghi']), 0, None)
    gcs = np.where(S.cz > 0.05, 1098 * np.clip(S.cz, 0.05, None) * np.exp(-0.057 / np.clip(S.cz, 0.05, None)), np.nan)
    avail, agc = ffill(S.x['avail']), ffill(S.x['agc'])
    a = dict(K=np.clip(S.K, -0.5, 1.5), kt=np.nan_to_num(np.clip(ghi / gcs, 0, 1.5)), agcn=np.clip(agc / cap, 0, 1.5),
             curt=((agc < 0.97 * avail) & (avail > 0.05 * cap)).astype(np.float32), cz=S.cz, cn=S.Ceff / cap,
             y=np.nan_to_num(S.x['P']) / cap, ce=S.Ceff / cap)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in a.items()}


class QHead(nn.Module):
    """maps features [B,H,d] to monotone quantiles: median + cumulative softplus increments."""
    def __init__(self, d):
        super().__init__(); self.lin = nn.Linear(d, Q)

    def forward(self, h):
        o = self.lin(h); med = o[..., MID]
        lo = Fn.softplus(o[..., :MID]) + 1e-4; hi = Fn.softplus(o[..., MID + 1:]) + 1e-4
        lower = med[..., None] - torch.flip(torch.cumsum(torch.flip(lo, [-1]), -1), [-1])
        return torch.cat([lower, med[..., None], med[..., None] + torch.cumsum(hi, -1)], -1)


class ExoHead(nn.Module):
    """shared exogenous head: future-known covariates per step + global context vector -> [B,H,d]"""
    def __init__(self, dctx, d):
        super().__init__(); self.f = nn.Linear(len(FUT) + 1, d); self.c = nn.Linear(dctx, d)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU()); self.q = QHead(d)

    def forward(self, ctx, fut):
        pos = torch.linspace(0, 1, H, device=fut.device)[None, :, None].expand(fut.shape[0], -1, -1)
        return self.q(self.mlp(self.f(torch.cat([fut, pos], -1)) + self.c(ctx)[:, None]))


class PatchTST(nn.Module):
    def __init__(self, d=128, patch=32, stride=16, layers=3, nst=7):
        super().__init__()
        self.patch, self.stride = patch, stride; n = (L - patch) // stride + 1
        self.emb = nn.Linear(patch, d); self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.cov = nn.Linear(len(PAST) - 1, d); self.st = nn.Embedding(nst, d)
        self.flat = nn.Linear(n * d, 256); self.head = ExoHead(256 + 2 * d, 128)

    def forward(self, past, fut, sid):
        x = past[:, 0].unfold(-1, self.patch, self.stride)  # [B,n,patch]
        h = self.enc(self.emb(x) + self.pos)
        ctx = torch.cat([self.flat(h.flatten(1)), self.cov(past[:, 1:, -COVW:].mean(-1)), self.st(sid)], -1)
        return self.head(ctx, fut)


class ITransformer(nn.Module):
    def __init__(self, d=128, layers=2, nst=7):
        super().__init__()
        self.past_emb = nn.Linear(L, d); self.fut_emb = nn.Linear(H, d)
        self.st = nn.Parameter(torch.randn(nst, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.out = nn.Linear(d, H * 32); self.q = QHead(32)

    def forward(self, past, fut, sid):
        tok = torch.cat([self.past_emb(past), self.fut_emb(fut.transpose(1, 2)), self.st[sid][:, None]], 1)
        h = self.enc(tok)[:, 0]  # target-variate token
        return self.q(Fn.gelu(self.out(h)).view(-1, H, 32))


class DLinear(nn.Module):
    def __init__(self, k=25, nst=7):
        super().__init__(); self.k = k
        self.trend = nn.Linear(L, 64); self.rem = nn.Linear(L, 64); self.cov = nn.Linear(len(PAST) - 1, 16); self.st = nn.Embedding(nst, 16)
        self.head = ExoHead(64 + 64 + 32, 64)

    def forward(self, past, fut, sid):
        x = past[:, 0]; tr = Fn.avg_pool1d(Fn.pad(x[:, None], (self.k // 2, self.k // 2), mode='replicate'), self.k, 1)[:, 0]
        ctx = torch.cat([self.trend(tr), self.rem(x - tr), self.cov(past[:, 1:, -COVW:].mean(-1)), self.st(sid)], -1)
        return self.head(ctx, fut)


class TiDE(nn.Module):
    def __init__(self, d=256, nst=7, ds=10):
        super().__init__(); self.ds = ds
        self.enc = nn.Sequential(nn.Linear(L // ds * len(PAST) + H * len(FUT) + 16, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.st = nn.Embedding(nst, 16); self.dec = nn.Linear(d, H * 16); self.temporal = nn.Sequential(nn.Linear(16 + len(FUT), 32), nn.GELU())
        self.q = QHead(32); self.skip = nn.Linear(L // ds, H)

    def forward(self, past, fut, sid):
        p = Fn.avg_pool1d(past, self.ds)  # [B,C,L/ds]
        z = self.enc(torch.cat([p.flatten(1), fut.flatten(1), self.st(sid)], -1))
        h = self.temporal(torch.cat([self.dec(z).view(-1, H, 16), fut], -1))
        q = self.q(h)
        return q + self.skip(p[:, 0])[..., None]


MODELS = dict(PatchTST=PatchTST, iTransformer=ITransformer, DLinear=DLinear, TiDE=TiDE)


def main(job, out, state):
    torch.manual_seed(job['seed']); np.random.seed(job['seed']); dev = 'cuda'
    names = list(STATIONS); Sd = {s: Station(ROOT / 'data' / f'{s}.npz', s, STATIONS[s][1]) for s in names}
    proto = job.get('proto')
    if proto and proto != 'main':
        from pvfm.protocol import available
        names = [s for s in names if available(Sd[s], proto)]
    A = {s: {k: v.to(dev) for k, v in station_arrays(Sd[s]).items()} for s in names}
    def origins(s, split, dense=False):
        S = Sd[s]
        if proto:
            from pvfm.protocol import splits as _sp
            sp = _sp(S, proto)
            if split == 'train':
                lo = L; hi = int(sp['es'].min()) - H
                return np.arange(lo, hi, 15)
            key = {'validation': None, 'es': 'es', 'cal': 'cal', 'test': 'test'}[split]
            o = np.sort(np.concatenate([sp['es'], sp['cal']]) if key is None else sp[key]); return o[o >= L - 1]
        if split == 'train':
            lo = L; hi = int(S.origins['validation'].min()) - H
            return np.arange(lo, hi, 15)
        o = np.sort(S.origins[split]); return o[o >= L - 1]
    tr = [(i, o) for i, s in enumerate(names) for o in origins(s, 'train')]
    va = {s: origins(s, 'validation') for s in names}
    if proto:
        es = [(i, o) for i, s in enumerate(names) for o in origins(s, 'es')]
    else:
        es = [(i, o) for i, s in enumerate(names) for o in va[s][:len(va[s]) // 2]]
    ar = torch.arange(-L + 1, 1, device=dev); fr = torch.arange(1, H + 1, device=dev)

    def batch(pairs):
        sid = torch.tensor([p[0] for p in pairs], device=dev); o = torch.tensor([p[1] for p in pairs], device=dev)
        past, fut, y, ce = [], [], [], []
        for i in sid.unique().tolist():
            m = sid == i; a = A[names[i]]; oi = o[m]; ci = oi[:, None] + ar; fi = oi[:, None] + fr
            past.append((m, torch.stack([a[k][ci] for k in PAST], 1))); fut.append((m, torch.stack([a[k][fi] for k in FUT], -1)))
            y.append((m, a['y'][fi])); ce.append((m, a['ce'][fi]))
        def assemble(parts, shape_tail):
            outp = torch.zeros((len(pairs),) + shape_tail, device=dev)
            for m, v in parts: outp[m] = v
            return outp
        return (assemble(past, (len(PAST), L)), assemble(fut, (H, len(FUT))), assemble(y, (H,)), assemble(ce, (H,)), sid)

    tau = torch.tensor(LEVELS, dtype=torch.float32, device=dev)
    def loss_fn(qk, y, ce):
        e = (y / ce)[..., None] - qk
        return (torch.maximum(tau * e, (tau - 1) * e) * ce[..., None]).mean()

    model = MODELS[job['model']](nst=len(names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=job.get('lr', 1e-3), weight_decay=1e-4)
    bs = job.get('batch', 256); best, bad, state_best = 1e9, 0, None; t0 = time.time(); log = []
    for ep in range(job.get('epochs', 30)):
        model.train(); perm = np.random.permutation(len(tr)); steps = min(len(perm) // bs, job.get('max_steps', 400))
        for k in range(steps):
            p, f, y, ce, sid = batch([tr[j] for j in perm[k * bs:(k + 1) * bs]])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                q = model(p, f, sid)
            l = loss_fn(q.float(), y, ce); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for k in range(0, len(es), 1024):
                p, f, y, ce, sid = batch(es[k:k + 1024]); v += loss_fn(model(p, f, sid), y, ce).item() * min(1024, len(es) - k)
        v /= len(es); log.append(round(v, 5))
        if v < best - 1e-5: best, bad, state_best = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: break
    model.load_state_dict(state_best); model.eval(); fit_s = time.time() - t0
    tag = f"DL_{job['model']}_s{job['seed']}"
    if proto:
        for i, s in enumerate(names):
            outd = {}
            for part in ('es', 'cal', 'test'):
                o = origins(s, part); qs = []
                with torch.no_grad():
                    for k in range(0, len(o), 1024):
                        p, f, y, ce, sid = batch([(i, x) for x in o[k:k + 1024]])
                        qs.append((model(p, f, sid) * ce[..., None] * Sd[s].cap).cpu().numpy())
                outd[f'o_{part}'] = o; outd[f'q_{part}'] = np.concatenate(qs).astype(np.float32)
            od = ROOT / 'runs/rbase' / f"DL_{job['model']}" / proto; od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f"{s}_s{job['seed']}.npz", **outd)
        return dict(fit_seconds=fit_s, epochs=len(log), es_log=log, n_params=sum(x.numel() for x in model.parameters()), n_train=len(tr))
    for i, s in enumerate(names):
        od = ROOT / 'runs/stageA' / tag / s; od.mkdir(parents=True, exist_ok=True)
        for split in ('validation', 'test'):
            o = origins(s, split); qs = []
            with torch.no_grad():
                for k in range(0, len(o), 1024):
                    p, f, y, ce, sid = batch([(i, x) for x in o[k:k + 1024]])
                    qs.append((model(p, f, sid) * ce[..., None] * Sd[s].cap).cpu().numpy())
            np.savez_compressed(od / f'{split}.npz', q=np.concatenate(qs).astype(np.float32), origins=o, seconds=0.0)
    return dict(fit_seconds=fit_s, epochs=len(log), es_log=log, n_params=sum(x.numel() for x in model.parameters()), n_train=len(tr))
