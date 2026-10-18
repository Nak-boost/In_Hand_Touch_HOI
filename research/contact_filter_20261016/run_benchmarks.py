from pathlib import Path
import csv,json,os,subprocess,time
root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');out=Path(__file__).resolve().parent
jobs=[]
for repeat in range(1,4):
 cases=[('gpu',False),('cpu',False),('cpu',True)]
 if repeat==2:cases=cases[::-1]
 for pipeline,enabled in cases:jobs.append((repeat,pipeline,enabled))
results=[]
for repeat,pipeline,enabled in jobs:
 label=f'{pipeline}_{"on" if enabled else "off"}_{repeat}'
 cmd=[str(root/'scripts/python_manip.sh'),str(root/'scripts/benchmark_object_contacts.py'),'--pipeline',pipeline,'--num-envs','256','--tape',str(out/'actions.npz'),'--output',str(out/(label+'.json'))]
 if enabled:cmd+=['--filter']
 if repeat==1 and pipeline=='cpu':cmd+=['--verify']
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  peak=0;start=time.time()
  while p.poll() is None:
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
   mem=sum(int(row[1]) for row in csv.reader(raw.splitlines()) if row[0].strip()==str(p.pid));peak=max(peak,mem)
   if time.time()-start>180:
    p.terminate();p.wait(timeout=10);raise RuntimeError('benchmark timeout '+label)
   time.sleep(1)
  result=dict(label=label,pid=p.pid,exit=p.returncode,peak_gpu_mb=peak,wall_seconds=time.time()-start)
  results.append(result);(out/'benchmark_processes.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(result),flush=True)
  if p.returncode:raise RuntimeError(label+' failed')
