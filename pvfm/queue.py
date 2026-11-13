"""Minimal NFS job queue. Jobs: JSON lines {id, kind, ...}. Claim = atomic mkdir; markers .done/.failed.
Usage: python -m pvfm.queue <jobs.jsonl> <runs_root> [worker_name]"""
from pvfm.common import PVROOT
import json, os, socket, sys, time, traceback
from pathlib import Path

jobs_file, root = Path(sys.argv[1]), Path(sys.argv[2])
worker = sys.argv[3] if len(sys.argv) > 3 else socket.gethostname()
state = {}


def get_pipe():
    if 'pipe' not in state:
        from chronos import Chronos2Pipeline
        p = Chronos2Pipeline.from_pretrained((PVROOT + '/models/chronos-2'), device_map='cuda')
        p.model.eval(); p.model.requires_grad_(False); state['pipe'] = p
    return state['pipe']


def handle(job, out):
    if job['kind'] == 'fm_zero':
        from pvfm.fm_zero import run
        return run(job['spec'], job['station'], job['data'], job.get('outdir', out), get_pipe(), splits=tuple(job.get('splits', ('validation', 'test'))))
    if job['kind'] == 'fm15':
        from pvfm.task15 import run15
        return run15(job['spec'], job['station'], job['data'], job.get('outdir', out), get_pipe(), splits=tuple(job.get('splits', ('train', 'validation', 'test'))))
    if job['kind'] == 'fm_multi':
        from pvfm.fm_multi import run as mrun
        return mrun(job['spec'], job['station'], job['data'], job.get('outdir', out), state, splits=tuple(job.get('splits', ('validation', 'test'))))
    if job['kind'] == 'python':
        import importlib
        mod = importlib.import_module(job['module'])
        return mod.main(job, out, state)
    raise ValueError(job['kind'])


done = 0
while True:
    jobs = [json.loads(l) for l in jobs_file.read_text().splitlines() if l.strip()]
    claimed = False
    for job in jobs:
        out = root / job['id']
        out.mkdir(parents=True, exist_ok=True)
        if (out / '.done').exists():
            continue
        try:
            os.mkdir(out / '.claim')
        except FileExistsError:
            continue
        claimed = True
        t = time.time()
        (out / '.claim' / 'worker').write_text(f'{worker} {time.strftime("%F %T")}')
        try:
            res = handle(job, out)
            (out / 'result.json').write_text(json.dumps(dict(job=job, result=res, worker=worker, seconds=time.time() - t), indent=1))
            (out / '.done').write_text(worker)
            done += 1
            print(f'[{worker}] done {job["id"]} {time.time()-t:.1f}s', flush=True)
        except Exception:
            tb = traceback.format_exc()
            fails = int((out / '.fails').read_text()) + 1 if (out / '.fails').exists() else 1
            (out / '.fails').write_text(str(fails))
            if fails < 3:  # release the claim so another worker can retry later
                import shutil; shutil.rmtree(out / '.claim', ignore_errors=True)
            else:
                (out / '.failed').write_text(tb)
            print(f'[{worker}] FAILED({fails}) {job["id"]}\n{tb}', flush=True)
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        break
    if not claimed:
        print(f'[{worker}] no more jobs; finished {done}', flush=True)
        break
