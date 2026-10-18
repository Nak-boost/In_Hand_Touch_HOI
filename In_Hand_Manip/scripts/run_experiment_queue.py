"""Run an explicit experiment manifest on designated GPUs with bounded concurrency."""
import argparse, datetime, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);a=p.parse_args()
    plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text());status_path=plan_path.with_name('queue_status.json')
    if status_path.exists():raise FileExistsError(status_path)
    jobs=[dict(job,status='pending') for job in plan['experiments']];running={}
    def save():
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated=now(),supervisor_pid=os.getpid(),jobs=jobs),indent=2)+'\n');tmp.replace(status_path)
    save()
    while any(j['status'] in ['pending','running'] for j in jobs):
        for idx,(proc,log) in list(running.items()):
            code=proc.poll()
            if code is not None:
                log.close();jobs[idx].update(status='completed' if code==0 else 'failed',exit_code=code,finished=now());del running[idx]
                print(jobs[idx]['run_name'],jobs[idx]['status'],flush=True);save()
        for gpu in plan['gpus']:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if sum(jobs[i]['gpu']==gpu for i in running)>=plan['max_jobs_per_gpu'] or free<plan['minimum_free_mb']:continue
            # Grasp checks can finish while earlier experiments are training.
            # A missing report is never interpreted as a passed validation.
            pending=next((i for i,j in enumerate(jobs) if j['status']=='pending' and Path(j['quality_report']).is_file()),None)
            if pending is None:break
            job=jobs[pending];quality=json.loads(Path(job['quality_report']).read_text())
            if quality['accepted']<plan['minimum_checked_grasps']:
                job.update(status='rejected',reason='Insufficient individually checked grasps');save();continue
            run=ROOT.parent/'runs/manip'/job['run_name']
            if run.exists():raise FileExistsError(run)
            logpath=run.with_suffix('.console.log');log=logpath.open('x')
            cmd=[str(ROOT/'scripts/run_manip.sh'),'--config-name',job['config'],'stage=teacher',
                'num_envs='+str(plan['num_envs']),'graphics_device_id=-1','run_name='+job['run_name'],
                'train.ppo.max_agent_steps='+str(plan['max_agent_steps']), 'seed='+str(job.get('seed',42))]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job.update(status='running',gpu=gpu,pid=proc.pid,started=now(),command=cmd,console_log=str(logpath))
            running[pending]=(proc,log);print(job['run_name'],'started pid',proc.pid,'gpu',gpu,flush=True);save()
        time.sleep(10)
    save()
if __name__=='__main__':main()
