from pathlib import Path
import subprocess, os, json, time
P=Path(__file__).resolve().parent
root=Path('/root/In_Hand_Reconstruction/research/current_four_runs_20261019/frozen_project')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1')
env.pop('MANIP_HEARTBEAT',None)
records=[]
for group,ck in [('table_box45','selected'),('old_baseline','best')]:
    r={'group':group,'checkpoint':ck,'seed':215,'started':time.time()}
    records.append(r)
    with (P/f'eval_{group}_{ck}_215.log').open('w') as log:
        proc=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(P/'evaluate.py'),'--group',group,'--checkpoint',ck,'--seed','215'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        r['pid']=proc.pid
        (P/'additional_processes.json').write_text(json.dumps(records,indent=2))
        code=proc.wait()
    r.update(returncode=code,finished=time.time(),result_saved=(P/f'eval_{group}_{ck}_215.json').exists())
    (P/'additional_processes.json').write_text(json.dumps(records,indent=2))
    if not r['result_saved']:raise SystemExit(f'Incomplete {group} {ck}: {code}')
