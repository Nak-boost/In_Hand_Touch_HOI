from pathlib import Path
import subprocess,os,json,time,hashlib,shutil
p=Path(__file__).resolve().parent;root=Path('/root/In_Hand_Reconstruction/research/current_four_runs_20261019/frozen_project');records=[];old=Path('/root/In_Hand_Reconstruction/research/current_four_results_20261020_1725')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1');env.pop('MANIP_HEARTBEAT',None)
for group in ['air_six','air_sphere55','table_sphere55','table_box45']:
 for ck in ['last','best']:
  rec={'group':group,'checkpoint':ck,'seed':215,'started':time.time()};records.append(rec)
  dest=p/f'eval_{group}_{ck}_215.json'
  prior=old/dest.name
  if prior.exists() and json.loads(prior.read_text())['checkpoint_sha256']==hashlib.sha256((p/f'{group}_{ck}.pth').read_bytes()).hexdigest():
   shutil.copy2(prior,dest);rec.update(reused_from=str(prior),result_saved=True);(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2));continue
  with (p/f'eval_{group}_{ck}_215.log').open('w') as log:
   proc=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(p/'evaluate.py'),'--group',group,'--checkpoint',ck,'--seed','215'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   rec['pid']=proc.pid;(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2));code=proc.wait()
  rec.update(returncode=code,finished=time.time(),result_saved=dest.exists());(p/'evaluation_processes.json').write_text(json.dumps(records,indent=2))
  if not rec['result_saved']:raise SystemExit(f'Incomplete evaluation {group} {ck}: {code}')
