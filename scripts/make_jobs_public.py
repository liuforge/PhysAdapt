"""Write the job files for the public-data experiment (Supplementary Section S3 of the paper).
Usage: PVFM_ROOT=<root> python scripts/make_jobs_public.py   ->  <root>/runs/q/jobs_p1.jsonl, jobs_p2.jsonl
Phase 1: zero-shot Chronos-2 (M1 input and geometry-blind), full fine-tuning / LoRA, deep and LightGBM baselines,
PARA-PV in our protocol, and the reproduction of PARA-PV in its own protocol. Phase 2: M2 (needs the phase-1 zero-shot run)."""
import json, os
from pathlib import Path

R = Path(os.environ.get('PVFM_ROOT', './workspace')).resolve(); ST = ['sgcc1_pv', 'sgcc6_pv']
PAPER = ['--train_epochs', '50', '--learning_rate', '0.001', '--batch_size', '32', '--d_model', '128', '--patch_memory_size', '4096', '--top_k', '5', '--dropout', '0.1']
P1 = [dict(id=f'zs_{spec}_{st}', kind='fm15', spec=spec, station=st, data=str(R / 'data'), outdir=str(R / 'runs/t15A' / spec / st))
      for spec in ('Kc_c672_pastfut', 'P_c96_pastfut') for st in ST]
P1 += [dict(id=f'parapv_{st}', kind='python', module='pvfm.parapv', station=st, proto='main', tag='PARA_PV_paper', tag_suffix='_paper', args=PAPER) for st in ST]
P1 += [dict(id='ftfull_main', kind='python', module='pvfm.base15', what='ft', mode='full', lr=1e-6, steps=3000, batch=32, tag='FT_full_c672', proto='main'),
       dict(id='ftlora_main', kind='python', module='pvfm.base15', what='ft', mode='lora', lr=1e-5, steps=3000, batch=32, tag='FT_lora_c672', proto='main')]
P1 += [dict(id=f'dl_{m}_{s}', kind='python', module='pvfm.base15', what='dl', model=m, seed=s, proto='main') for m in ('PatchTST', 'iTransformer', 'DLinear', 'TiDE') for s in (28, 29, 30)]
P1 += [dict(id='lgbm_main', kind='python', module='pvfm.base15', what='lgbm', threads=12, proto='main')]
P1 += [dict(id=f'repro_{ds}_{mode}_{H}', kind='python', module='pvfm.parapv_repro', dataset=ds, mode=mode, pred_len=H)
       for ds in ('D1', 'D2') for mode in ('prob', 'point') for H in (4, 16, 48, 96)]
(R / 'runs/q').mkdir(parents=True, exist_ok=True)
(R / 'runs/q/jobs_p1.jsonl').write_text('\n'.join(json.dumps(j) for j in P1) + '\n')
(R / 'runs/q/jobs_p2.jsonl').write_text(json.dumps(dict(id='r15_main', kind='python', module='pvfm.radapt15', proto='main')) + '\n')
print(len(P1), 'phase-1 jobs and 1 phase-2 job written to', R / 'runs/q')
