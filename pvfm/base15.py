"""Strong baselines for the 15-min / 0-4 h task under the rolling protocol (same calendar as splits15), with the SAME
information as the FM pipeline at 15-min resolution. Queue job kinds (job['what']):
  dl   : PatchTST / iTransformer / TiDE / DLinear (pooled, station embedding), K-space, C_eff-weighted pinball, L = 672 (7 d)
  lgbm : LightGBM quantile regression in K-space (13 models), C_eff-weighted
  ft   : Chronos-2 full fine-tuning or LoRA in the M1 representation (Kc, ctx 672, geometry known-future covariates)
Outputs fold format runs/rbase15/<tag>/<proto>/<station>[_s<seed>].npz with o_/q_ for es, cal, test (P-space quantiles)."""
from pvfm.common import PVROOT
import time
from pathlib import Path
import numpy as np
from pvfm.common import STATIONS, LEVELS
from pvfm.fm_zero import COVS
from pvfm.task15 import Station15, splits15, H15, CTX_MAX

ROOT = Path(PVROOT); OUT = ROOT / 'runs/rbase15'
L15 = CTX_MAX  # 672 = 7 days


def usable(S, proto):
    if proto == 'main': return True
    sp = splits15(S, proto)
    return len(sp['test']) > 50 and len(sp['cal']) > 50 and len(sp['es']) > 50 and len(sp['train']) >= 200


def stations(proto):
    Sd = {s: Station15(ROOT / 'data' / f'{s}.npz', s, STATIONS[s][1]) for s in STATIONS}
    return {s: S for s, S in Sd.items() if usable(S, proto)}


def parts(S, proto):
    sp = {k: np.sort(v) for k, v in splits15(S, proto).items()}
    return {k: v[v >= L15 - 1] for k, v in sp.items()}


def save(tag, proto, st, outd, seed=None):
    od = OUT / tag / proto; od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / (f'{st}_s{seed}.npz' if seed is not None else f'{st}.npz'), **outd)


def arrays15(S):
    cap = S.cap; gh = np.clip(np.nan_to_num(S.x['ghi']), 0, None)
    gcs = np.where(S.cz > 0.05, 1098 * np.clip(S.cz, 0.05, None) * np.exp(-0.057 / np.clip(S.cz, 0.05, None)), np.nan)
    agc, avail = np.nan_to_num(S.x['agc'], nan=cap), np.nan_to_num(S.x['avail'], nan=cap)
    return dict(K=np.clip(S.K, -0.5, 1.5), kt=np.nan_to_num(np.clip(gh / gcs, 0, 1.5)), agcn=np.clip(agc / cap, 0, 1.5),
                curt=((agc < 0.97 * avail) & (avail > 0.05 * cap)).astype(np.float32), cz=S.cz, cn=S.Ceff / cap,
                y=S.P / cap, ce=S.Ceff / cap)


# ---------------------------------------------------------------- deep supervised baselines
def run_dl(job):
    import torch
    from torch import nn
    import pvfm.dlbase as D
    D.H, D.L, D.COVW = H15, L15, 4
    torch.manual_seed(job['seed']); np.random.seed(job['seed']); dev = 'cuda'; proto = job['proto']
    Sd = stations(proto); names = list(Sd)
    A = {s: {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in arrays15(Sd[s]).items()} for s in names}
    P = {s: parts(Sd[s], proto) for s in names}
    tr = [(i, o) for i, s in enumerate(names) for o in P[s]['train']]
    es = [(i, o) for i, s in enumerate(names) for o in P[s]['es']]
    ar = torch.arange(-L15 + 1, 1, device=dev); fr = torch.arange(1, H15 + 1, device=dev)

    def batch(pairs):
        sid = torch.tensor([p[0] for p in pairs], device=dev); o = torch.tensor([p[1] for p in pairs], device=dev)
        past = torch.zeros((len(pairs), len(D.PAST), L15), device=dev); fut = torch.zeros((len(pairs), H15, len(D.FUT)), device=dev)
        y = torch.zeros((len(pairs), H15), device=dev); ce = torch.zeros((len(pairs), H15), device=dev)
        for i in sid.unique().tolist():
            m = sid == i; a = A[names[i]]; oi = o[m]; ci = oi[:, None] + ar; fi = oi[:, None] + fr
            past[m] = torch.stack([a[k][ci] for k in D.PAST], 1); fut[m] = torch.stack([a[k][fi] for k in D.FUT], -1)
            y[m] = a['y'][fi]; ce[m] = a['ce'][fi]
        return past, fut, y, ce, sid

    tau = torch.tensor(LEVELS, dtype=torch.float32, device=dev)

    def loss_fn(qk, y, ce):
        e = (y / ce)[..., None] - qk
        return (torch.maximum(tau * e, (tau - 1) * e) * ce[..., None]).mean()

    model = D.MODELS[job['model']](nst=len(names)).to(dev)
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
    for i, s in enumerate(names):
        outd = {}
        for part in ('es', 'cal', 'test'):
            o = P[s][part]; qs = []
            with torch.no_grad():
                for k in range(0, len(o), 1024):
                    p, f, y, ce, sid = batch([(i, x) for x in o[k:k + 1024]])
                    qs.append((model(p, f, sid) * ce[..., None] * Sd[s].cap).float().cpu().numpy())
            outd[f'o_{part}'] = o; outd[f'q_{part}'] = np.concatenate(qs).astype(np.float32)
        save(f"DL_{job['model']}", proto, s, outd, job['seed'])
    return dict(fit_seconds=fit_s, epochs=len(log), es_log=log, n_params=sum(x.numel() for x in model.parameters()), n_train=len(tr))


# ---------------------------------------------------------------- LightGBM
def lgb_design(S, a, o, hs):
    cap = S.cap; K = a['K']; n = len(K)
    cs = np.cumsum(np.concatenate([[0], K])); cs2 = np.cumsum(np.concatenate([[0], K ** 2]))
    mean = lambda o, w: (cs[o + 1] - cs[o + 1 - w]) / w
    std = lambda o, w: np.sqrt(np.maximum((cs2[o + 1] - cs2[o + 1 - w]) / w - mean(o, w) ** 2, 0))
    oo = np.repeat(o, len(hs)); hh = np.tile(hs, len(o)); t = oo + hh + 1
    tod = ((S.minutes[t]) % 1440) / 1440
    lag = np.stack([K[t - 96 * k] for k in range(1, 6)], -1)
    temp = np.nan_to_num(S.x.get('temp', np.zeros(n)))[oo]; wind = np.nan_to_num(S.x.get('wind', np.zeros(n)))[oo]
    X = np.column_stack([K[oo], mean(oo, 2), mean(oo, 4), mean(oo, 12), std(oo, 4), a['kt'][oo], lag, lag.mean(-1), lag.std(-1),
                         S.cz[oo], S.cz[t], S.Ceff[t] / cap, (hh + 1) / H15, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                         a['curt'][oo], a['agcn'][oo], temp / 40, wind / 10]).astype(np.float32)
    return X, K[t], S.Ceff[t], t


def run_lgbm(job):
    import lightgbm as lgb
    proto = job['proto']; Sd = stations(proto); names = list(Sd)
    D = {s: (Sd[s], arrays15(Sd[s]), parts(Sd[s], proto)) for s in names}

    def stack(part, hs, sub=1):
        Xs, ys, ws = [], [], []
        for i, s in enumerate(names):
            S, a, P = D[s]; X, y, w, _ = lgb_design(S, a, P[part][::sub], hs)
            Xs.append(np.column_stack([X, np.full(len(X), i, np.float32)])); ys.append(y); ws.append(w)
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    hs = np.arange(H15)
    Xtr, ytr, wtr = stack('train', hs, 2); Xes, yes, wes = stack('es', hs)
    t = time.time(); models = []
    for tau in LEVELS:
        m = lgb.LGBMRegressor(objective='quantile', alpha=float(tau), n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=job.get('threads', 12), verbose=-1)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xes, yes)], eval_sample_weight=[wes], callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
    fit_s = time.time() - t
    for i, s in enumerate(names):
        S, a, P = D[s]; outd = {}
        for part in ('es', 'cal', 'test'):
            o = P[part]; X, _, _, fi = lgb_design(S, a, o, hs); X = np.column_stack([X, np.full(len(X), i, np.float32)])
            q = np.stack([m.predict(X) for m in models], -1).reshape(len(o), H15, len(LEVELS))
            outd[f'o_{part}'] = o; outd[f'q_{part}'] = (np.sort(q, -1) * S.Ceff[fi].reshape(len(o), H15)[..., None]).astype(np.float32)
        save('LGBM_K', proto, s, outd)
    return dict(fit_seconds=fit_s, best_iters=[int(m.best_iteration_ or 0) for m in models], n_train_rows=int(len(Xtr)))


# ---------------------------------------------------------------- Chronos-2 full fine-tuning / LoRA (M1 representation)
def series15(S, end, keys):
    pc = {k: np.nan_to_num(S.x[k][:end]) for k in keys}
    pc.update(cz=S.cz[:end], cs=S.Ceff[:end])
    return dict(target=np.clip(S.K, -0.5, 1.5)[:end], past_covariates=pc)


def predict15(pipe, S, o, keys, ctx=L15):
    tgt = np.clip(S.K, -0.5, 1.5); cov = {k: np.nan_to_num(S.x[k]) for k in keys}; inputs = []
    for x in o:
        ci = np.arange(x - ctx + 1, x + 1); fi = np.arange(x + 1, x + H15 + 1)
        d = {'target': tgt[ci], 'past_covariates': {**{k: v[ci] for k, v in cov.items()}, 'cz': S.cz[ci], 'cs': S.Ceff[ci]},
             'future_covariates': {'cz': S.cz[fi], 'cs': S.Ceff[fi]}}
        inputs.append(d)
    q, _ = pipe.predict_quantiles(inputs, prediction_length=H15, quantile_levels=LEVELS.tolist(), batch_size=256)
    q = np.stack([z[0].float().cpu().numpy() for z in q])
    fi = o[:, None] + np.arange(1, H15 + 1)[None]
    return np.maximum.accumulate(q * S.Ceff[fi][..., None], axis=-1).astype(np.float32)


def run_ft(job):
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts
    torch.manual_seed(job.get('seed', 0)); proto = job['proto']; Sd = stations(proto); names = list(Sd)
    P = {s: parts(Sd[s], proto) for s in names}
    keys = sorted(set.intersection(*[{k for k in COVS if k in Sd[s].x} for s in names]))
    tr = [series15(Sd[s], int(P[s]['es'].min()) - H15, keys) for s in names]
    va = [series15(Sd[s], int(P[s]['test'].min()) - 1, keys) for s in names]
    kn = ['cz', 'cs']
    tr_in = from_list_of_dicts(tr, H15, known_covariates_names=kn); va_in = from_list_of_dicts(va, H15, known_covariates_names=kn)
    base = Chronos2Pipeline.from_pretrained(str(ROOT / 'models/chronos-2'), device_map='cuda'); t = time.time()
    ft = base.fit(tr_in, prediction_length=H15, validation_inputs=va_in, finetune_mode=job['mode'], context_length=L15,
                  learning_rate=job['lr'], num_steps=job['steps'], batch_size=job['batch'], output_dir=str(Path(job['_out']) / 'ckpt'),
                  remove_printer_callback=True)
    fit_s = time.time() - t; ft.model.eval()
    for s in names:
        outd = {}
        for part in ('es', 'cal', 'test'):
            o = P[s][part]; outd[f'o_{part}'] = o; outd[f'q_{part}'] = predict15(ft, Sd[s], o, keys)
        save(job['tag'], proto, s, outd)
    return dict(fit_seconds=fit_s, trainable_params=sum(p.numel() for p in ft.model.parameters() if p.requires_grad), keys=keys)


def main(job, out, state):
    job = {**job, '_out': out}
    return {'dl': run_dl, 'lgbm': run_lgbm, 'ft': run_ft}[job['what']](job)


def average_seeds():
    import re
    from collections import defaultdict
    for tagdir in sorted(OUT.glob('DL_*')):
        for pdir in sorted(tagdir.iterdir()):
            groups = defaultdict(list)
            for f in pdir.glob('*_s*.npz'): groups[re.sub(r'_s\d+\.npz$', '', f.name)].append(f)
            for st, fs in groups.items():
                zs = [np.load(f) for f in fs]
                out = {k: (zs[0][k] if k.startswith('o_') else np.mean([z[k] for z in zs], 0).astype(np.float32)) for k in zs[0].files}
                np.savez_compressed(pdir / f'{st}.npz', **out); print(tagdir.name, pdir.name, st, len(fs))


if __name__ == '__main__':
    average_seeds()
