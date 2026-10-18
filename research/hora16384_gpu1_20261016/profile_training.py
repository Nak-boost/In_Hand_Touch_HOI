"""Bounded PPO capacity/throughput probe on GPU 1; preserve each independent run."""
import argparse,csv,datetime,json,os,re,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--envs',type=int,required=True);p.add_argument('--rounds',type=int,default=8)
a=p.parse_args();ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');OUT=Path(__file__).resolve().parent
dest=OUT/f'probe_{a.envs}';dest.mkdir(exist_ok=False)
def snapshot(pid=None):
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines())
    x=next(x for x in rows if int(x[0])==1)
    r=dict(gpu=1,uuid=x[1].strip(),total_mib=int(x[2]),used_mib=int(x[3]),utilization=int(x[4]),time=time.time())
    if pid:
        rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines())
        r['process_mib']=sum(int(x[2]) for x in rows if len(x)==3 and x[0].strip()==str(pid) and x[1].strip()==r['uuid'])
    return r
baseline=snapshot()
if baseline['total_mib']-baseline['used_mib']<12000:raise RuntimeError('Insufficient free memory on GPU 1')
steps=a.envs*8*a.rounds
command=[str(ROOT/'scripts/run_manip.sh'),'--config-name','manip_hora16384_sphere55','stage=teacher',f'num_envs={a.envs}',
    f'train.ppo.max_agent_steps={steps}',f'train.ppo.save_frequency={a.rounds}',f'run_root={dest}/runs','run_name=capacity']
samples=[];reason=None;start=time.time()
with (dest/'console.log').open('w') as log:
    proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
    while proc.poll() is None:
        s=snapshot(proc.pid);samples.append(s)
        (dest/'live.json').write_text(json.dumps(dict(pid=proc.pid,latest=s,elapsed=time.time()-start),indent=2))
        if s['total_mib']-s['used_mib']<2048 or s['process_mib']>20000:reason='memory_headroom_limit';proc.terminate();break
        if time.time()-start>900:reason='15_minute_probe_limit';proc.terminate();break
        time.sleep(2)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
text=(dest/'console.log').read_text()
fps=[float(x) for x in re.findall(r'Last FPS: ([0-9.]+)',text)]
steady=fps[2:]
report=dict(num_envs=a.envs,rounds=a.rounds,steps=steps,pid=proc.pid,exit_code=proc.returncode,
    stop_reason=reason,wall_seconds=time.time()-start,baseline=baseline,command=command,
    peak_process_mib=max((x['process_mib'] for x in samples),default=0),
    last_fps=fps,steady_harmonic_fps=len(steady)/sum(1/x for x in steady) if steady else None,
    completion_marker='max steps achieved' in text,
    samples=samples,note='Full PPO end-to-end Last FPS after first two warmup rollouts. GPU utilization is whole-card, memory is process-specific.')
(dest/'resources.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['samples','command','baseline']},indent=2),flush=True)
if proc.returncode or not report['completion_marker']:raise RuntimeError('Probe did not finish normally')
