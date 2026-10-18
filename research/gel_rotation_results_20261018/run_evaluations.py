from pathlib import Path
import subprocess,os,json,time
p=Path(__file__).resolve().parent;root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');records=[]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1');env.pop('MANIP_HEARTBEAT',None)
for group,ck in [('single','last'),('single','best'),('multi','last'),('multi','best')]:
 rec={'group':group,'checkpoint':ck,'started':time.time()};records.append(rec)
 with (p/f'eval_{group}_{ck}.log').open('w') as log:
  proc=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(p/'evaluate.py'),'--group',group,'--checkpoint',ck],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  rec['pid']=proc.pid;(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2));code=proc.wait()
 rec.update(returncode=code,finished=time.time(),result_saved=(p/f'eval_{group}_{ck}_215.json').exists());(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2))
 if not rec['result_saved']:raise SystemExit(f'Incomplete evaluation {group} {ck}: {code}')
