from pathlib import Path
import subprocess,os,json,time
p=Path(__file__).resolve().parent;root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');records=[]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1');env.pop('MANIP_HEARTBEAT',None)
for ck in ['last','best']:
 rec={'group':'single','checkpoint':ck,'seed':216,'started':time.time()};records.append(rec)
 with (p/f'eval_single_{ck}_216.log').open('w') as log:
  proc=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(p/'evaluate.py'),'--group','single','--checkpoint',ck,'--seed','216'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  rec['pid']=proc.pid;(p/'recheck_processes.json').write_text(json.dumps(records,indent=2));code=proc.wait()
 rec.update(returncode=code,finished=time.time(),result_saved=(p/f'eval_single_{ck}_216.json').exists());(p/'recheck_processes.json').write_text(json.dumps(records,indent=2))
 if not rec['result_saved']:raise SystemExit(f'Incomplete evaluation single {ck}: {code}')
