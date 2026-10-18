import subprocess,time,json,os
from pathlib import Path
out=Path(__file__).resolve().parent
root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
records=[]
for seed,mode,checkpoint,label in [(44,'mean','ep_1300_step_0005M_reward_-89.32.pth','_5M'),(44,'mean','ep_1800_step_0007M_reward_-218.45.pth','_7M')]:
 dest=out/f'eval_{seed}_{mode}{label}.json'
 with (out/f'eval_{seed}_{mode}{label}.log').open('w') as log:
  p=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(out/'evaluate.py'),'--seed',str(seed),'--mode',mode,'--checkpoint',checkpoint,'--label',label],cwd=root,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  start=time.time();written=None
  while p.poll() is None:
   if dest.exists():
    if written is None:written=time.time()
    if time.time()-written>5:
     p.terminate()
     try:p.wait(timeout=5)
     except subprocess.TimeoutExpired:p.kill();p.wait()
     break
   if time.time()-start>240:
    p.kill();p.wait();break
   time.sleep(1)
  records.append(dict(seed=seed,mode=mode,checkpoint=checkpoint,pid=p.pid,exit=p.returncode,json_written=dest.exists(),seconds=time.time()-start,close_hang=written is not None and p.returncode!=0))
  (out/'intermediate_processes.json').write_text(json.dumps(records,indent=2))
  print(records[-1],flush=True)
