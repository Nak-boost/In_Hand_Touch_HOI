import subprocess,os,json,time
import torch
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONUNBUFFERED']='1';env['PYTHONFAULTHANDLER']='1'
records=json.loads((OUT/'preparation_status.json').read_text())
for obj in ['box45','box55','capsule70','capsule80']:
    for mode in ['generate','validate']:
        if obj=='box45' and mode=='generate':continue
        record=dict(object=obj,mode=mode,started=time.time());records.append(record)
        (OUT/'preparation_status.json').write_text(json.dumps(records,indent=2))
        with (OUT/'grasps'/obj/(mode+'.log')).open('w') as log:
            proc=subprocess.run([str(ROOT/'scripts/python_manip.sh'),str(OUT/'prepare_grasps.py'),'--object',obj,'--mode',mode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record.update(returncode=proc.returncode,finished=time.time())
        (OUT/'preparation_status.json').write_text(json.dumps(records,indent=2))
        if proc.returncode:
            logtext=(OUT/'grasps'/obj/(mode+'.log')).read_text()
            if proc.returncode!=-11 or 'CLEANUP_START' not in logtext or 'SAVED ' not in logtext:
                raise SystemExit(f'{obj} {mode} failed: {proc.returncode}')
            suffix='candidates' if mode=='generate' else 'checked'
            path=ROOT/f'cache/digit_{obj}_pair_inward90_{suffix}_20261017.pt'
            bank=torch.load(path,map_location='cpu')
            assert len(bank['states'][obj])>=64 and torch.isfinite(bank['states'][obj]).all() and torch.isfinite(bank['targets'][obj]).all()
            assert bank['asset_sha256']=='c9693d6fa850ea1acb0cb0cd3b5ba2cce77f0e08867c2724a7d668e614552b24'
            record['cleanup_failure_after_saved_artifact']=True
            (OUT/'preparation_status.json').write_text(json.dumps(records,indent=2))
