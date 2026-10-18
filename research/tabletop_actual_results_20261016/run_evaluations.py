import subprocess,time,json,os
from pathlib import Path
out=Path(__file__).resolve().parent
root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
records=[]
for seed,mode in [(43,'mean'),(44,'mean'),(42,'sampled'),(43,'sampled'),(44,'sampled')]:
 dest=out/f'eval_{seed}_{mode}.json'
 with (out/f'eval_{seed}_{mode}.log').open('w') as log:
  p=subprocess.Popen([str(root/'scripts/python_manip.sh'),str(out/'evaluate.py'),'--seed',str(seed),'--mode',mode],cwd=root,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
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
  records.append(dict(seed=seed,mode=mode,pid=p.pid,exit=p.returncode,json_written=dest.exists(),seconds=time.time()-start,close_hang=written is not None and p.returncode!=0))
  (out/'evaluation_processes.json').write_text(json.dumps(records,indent=2))
  print(records[-1],flush=True)
