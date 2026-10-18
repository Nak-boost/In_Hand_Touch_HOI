"""Bounded sequential replay on GPU 3. Never starts training or stops unrelated work."""
import subprocess, time, json, os
from pathlib import Path
out=Path(__file__).resolve().parent
root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
objects=['sphere55','box45','sphere65','capsule70','box55','capsule80']
jobs=[(obj,'mean','last',True) for obj in ['sphere65','sphere55','box45','capsule70','box55','capsule80']]
jobs += [(obj,'zero','last',False) for obj in objects]
records=[]
for obj,mode,ck,pairs in jobs:
    label=f'{obj}_{mode}_{ck}'+('_pairs' if pairs else '')
    dest=out/f'eval_{label}.json'
    if dest.exists(): continue
    with (out/f'eval_{label}.log').open('w') as log:
        cmd=[str(root/'scripts/python_manip.sh'),str(out/'evaluate.py'),'--object',obj,'--mode',mode,'--checkpoint',ck]
        if pairs:cmd.append('--pairs')
        p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
        start=time.time();written=None;reason=None
        while p.poll() is None:
            if dest.exists():
                if written is None:written=time.time()
                if time.time()-written>5:reason='cleanup_timeout_after_result';break
            if time.time()-start>180:reason='evaluation_timeout';break
            time.sleep(1)
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        records.append(dict(label=label,pid=p.pid,exit=p.returncode,json_written=dest.exists(),seconds=time.time()-start,termination_reason=reason))
        (out/'contact_baseline_processes.json').write_text(json.dumps(records,indent=2))
        print(records[-1],flush=True)
        if not dest.exists():raise RuntimeError('Evaluation failed: '+label)
