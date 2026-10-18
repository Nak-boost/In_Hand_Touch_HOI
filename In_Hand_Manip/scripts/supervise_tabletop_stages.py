"""One authorized tabletop experiment per GPU, with checkpointed evaluation stages."""
import argparse, csv, datetime, hashlib, json, os, signal, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def completed_before_teardown(run, logpath, target):
    """Recognize only a saved, complete stage followed by the known Gym teardown crash."""
    import torch
    log = Path(logpath).read_text()
    marker = 'Fatal Python error: Segmentation fault'
    if marker not in log or 'max steps achieved' not in log.split(marker)[0]:
        return False
    trace = log.split('Current thread', 1)[-1]
    if 'allegro_digit_two_axis.py' not in trace or 'in close' not in trace:
        return False
    try:
        checkpoint = torch.load(Path(run)/'stage1_nn/last.pth', map_location='cpu')
        if checkpoint.get('agent_steps', 0) < target or 'optimizer' not in checkpoint:
            return False
        def finite(value):
            if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
            if isinstance(value, dict): return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
            return True
        return bool(checkpoint.get('model')) and finite(checkpoint)
    except (OSError, RuntimeError, ValueError, EOFError):
        return False


def verify(plan):
    for path,digest in plan['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen experiment changed: '+path)
    for job in plan['experiments']:
        quality=json.loads(Path(job['quality_report']).read_text())
        resource=json.loads(Path(job['resource_report']).read_text())
        if not quality['passed'] or quality['accepted']<64 or quality['action_reference']!='actual_position':
            raise ValueError('Tabletop grasp gate failed')
        cleanup_only = (plan.get('allow_verified_teardown_failure', False)
            and resource['exit_code'] == -11
            and completed_before_teardown(Path(job['resource_report']).parent/'runs/resource_probe',
                Path(job['resource_report']).parent/'console.log', plan.get('probe_steps', 524288)))
        if (resource['exit_code'] and not cleanup_only) or resource['num_envs']!=job['num_envs']:
            raise ValueError('PPO resource gate failed')


def gpu_stats():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    return {int(r[0]):dict(uuid=r[1].strip(),total_mb=int(r[2]),used_mb=int(r[3]),utilization=int(r[4]))
            for r in csv.reader(lines.splitlines())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();path=Path(a.plan).resolve();plan=json.loads(path.read_text());out=path.parent
    verify(plan)
    if a.once:
        print(json.dumps(dict(verified=True,gpus=gpu_stats()),indent=2));return
    import fcntl
    lock=path.with_suffix('.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statuspath=out/'queue_status.json'
    if statuspath.exists():raise FileExistsError('Existing supervisor status; refusing duplicate start')
    jobs=[dict(j,status='waiting',stage_index=0,phase='train',history=[],process_peak_mb=0) for j in plan['experiments']]
    active={};stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def save():
        temp=statuspath.with_suffix('.tmp');temp.write_text(json.dumps(dict(updated=now(),supervisor_pid=os.getpid(),
            frozen_root=str(ROOT),jobs=jobs),indent=2)+'\n');temp.replace(statuspath)
    def launch(job):
        verify(plan)
        gpu=gpu_stats()[job['gpu']]
        report=json.loads(Path(job['resource_report']).read_text())
        needed=int(report['process_peak_mb']*1.3)+2048
        idle_required=plan.get('require_idle_gpu', True)
        if gpu['total_mb']-gpu['used_mb']<needed or (idle_required and (gpu['used_mb']>1024 or gpu['utilization']>10)):
            job['waiting_reason']='Insufficient measured headroom or requested idle GPU unavailable';return
        milestone=plan['milestones'][job['stage_index']]
        run_name=job['run_prefix']+'_s'+str(milestone//1000000)+'m'
        run=Path(plan['run_root'])/run_name
        phase=job['phase']
        if phase=='train':
            cmd=[str(ROOT/'scripts/run_manip.sh'),'--config-name',job['config'],'stage=teacher',
                 'run_root='+plan['run_root'],'run_name='+run_name,'train.ppo.max_agent_steps='+str(milestone)]
            if job['stage_index']:
                prev=job['run_prefix']+'_s'+str(plan['milestones'][job['stage_index']-1]//1000000)+'m'
                cmd+=['checkpoint='+str(Path(plan['run_root'])/prev/'stage1_nn/last.pth')]
        else:
            ck=phase.split('_',1)[1]
            # A resumed stage may never improve the historical best reward;
            # then PPO legitimately creates no new best.pth in this run.
            if ck=='best' and not (run/'stage1_nn/best.pth').is_file():
                job['history'].append(dict(phase=phase,milestone=milestone,skipped=True,
                    reason='No new reward-best checkpoint in this stage',finished=now()))
                job['stage_index']+=1;job['phase']='train'
                if job['stage_index']==len(plan['milestones']):job.update(status='completed',finished=now())
                save();return
            cmd=[str(ROOT/'scripts/python_manip.sh'),str(ROOT/'scripts/evaluate_tabletop_checkpoint.py'),
                 '--run-dir',str(run),'--checkpoint',ck,'--envs','64','--steps','600',
                 '--output',str(out/(run_name+'_'+ck+'_evaluation.json'))]
        logpath=out/(run_name+'_'+phase+'.log');log=logpath.open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[job['object']]=(proc,log)
        job.update(status='running',pid=proc.pid,started=now(),run_dir=str(run),log=str(logpath),command=cmd)
        job.pop('waiting_reason',None);save()
    try:
        while not stop and any(j['status'] in ('waiting','running') for j in jobs):
            stats=gpu_stats()
            app_rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines())
            memory={(int(r[0]),r[1].strip()):int(r[2]) for r in app_rows if len(r)==3 and r[2].strip().isdigit()}
            for job in jobs:
                key=job['object']
                if key in active:
                    proc,log=active[key];code=proc.poll()
                    used=memory.get((proc.pid,stats[job['gpu']]['uuid']),0)
                    job['last_sample']=dict(time=now(),process_memory_mb=used,**stats[job['gpu']])
                    job['process_peak_mb']=max(job['process_peak_mb'],used)
                    with (out/(key+'_resources.jsonl')).open('a') as f:f.write(json.dumps(job['last_sample'])+'\n')
                    if code is not None:
                        log.close();del active[key]
                        job['history'].append(dict(phase=job['phase'],milestone=plan['milestones'][job['stage_index']],
                            exit_code=code,log=job['log'],finished=now()))
                        cleanup_only=(code == -11 and job['phase']=='train'
                            and plan.get('allow_verified_teardown_failure',False)
                            and completed_before_teardown(job['run_dir'],job['log'],plan['milestones'][job['stage_index']]))
                        if cleanup_only:
                            job['history'][-1]['warning']='Training checkpoint verified complete; native simulator teardown failed'
                        if code and not cleanup_only:job.update(status='failed',exit_code=code);continue
                        if job['phase']=='train':job['phase']='eval_last'
                        elif job['phase']=='eval_last':job['phase']='eval_best'
                        else:
                            job['stage_index']+=1;job['phase']='train'
                            if job['stage_index']==len(plan['milestones']):job.update(status='completed',finished=now());continue
                        job['status']='waiting'
                if job['status']=='waiting':launch(job)
            # Publish evaluated alternatives; best.pth remains explicitly reward-selected.
            evaluations=[]
            for f in out.glob('*_evaluation.json'):
                r=json.loads(f.read_text());evaluations.append({k:r[k] for k in ['object','agent_steps','checkpoint',
                    'qualified_one_turn_fraction','one_turn_retained_fraction','late_signed_speed_all_starts','frame_means']})
            (out/'evaluated_checkpoints.json').write_text(json.dumps(evaluations,indent=2)+'\n')
            save();time.sleep(5)
    finally:
        for key,(proc,log) in active.items():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
            next(j for j in jobs if j['object']==key).update(status='interrupted',finished=now())
        save()


if __name__=='__main__':main()
