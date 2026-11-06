from pathlib import Path
import subprocess,os,json,time,hashlib,shutil
p=Path(__file__).resolve().parent;old=Path('/root/In_Hand_Reconstruction/research/gel_rotation_results_20261018');root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');records=[]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1');env.pop('MANIP_HEARTBEAT',None)
for group,ck,seed in [('single','last',215),('single','last',216),('single','best',215),('single','best',216),('multi','last',215),('multi','best',215)]:
 rec={'group':group,'checkpoint':ck,'seed':seed,'started':time.time()};records.append(rec)
 dest=p/f'eval_{group}_{ck}_{seed}.json';prior=old/dest.name
 if prior.exists() and json.loads(prior.read_text())['checkpoint_sha256']==hashlib.sha256((p/f'{group}_{ck}.pth').read_bytes()).hexdigest():
  shutil.copy2(prior,dest);rec.update(reused_from=str(prior),result_saved=True);(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2));continue
 with (p/f'eval_{group}_{ck}_{seed}.log').open('w') as log:
  proc=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(p/'evaluate.py'),'--group',group,'--checkpoint',ck,'--seed',str(seed)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  rec['pid']=proc.pid;(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2));code=proc.wait()
 rec.update(returncode=code,finished=time.time(),result_saved=dest.exists());(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2))
 if not rec['result_saved']:raise SystemExit(f'Incomplete evaluation {group} {ck}: {code}')
