"""Wait for stable GPU 0 headroom, then resume the interrupted mixed-object run once."""
import os,json,time,signal,subprocess,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
F=P/'frozen_project'
state={'pid':os.getpid(),'status':'waiting_for_gpu_memory','gpu':0,'minimum_free_mib':12000,'required_consecutive_samples':4,'poll_seconds':30,'samples':0,'started':time.time()}
stop=False
def save():
    tmp=P/'multi_wait_status.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(P/'multi_wait_status.json')
def stopping(signum,frame):
    global stop
    stop=True
signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
def digest():
    paths=[F/'train_manip.py']
    for folder in ('hora','configs','scripts'):paths.extend(p for p in (F/folder).rglob('*') if p.suffix in ('.py','.yaml','.sh'))
    h=hashlib.sha256()
    for p in sorted(set(paths)):h.update(str(p.relative_to(F)).encode());h.update(p.read_bytes())
    return h.hexdigest()
expected=json.loads((P/'multi_monitor/status.json').read_text())['source_sha256']
checkpoint=Path('/root/In_Hand_Reconstruction/runs/manip/teacher_six_objects_gel16384_20261017_attempt00/stage1_nn/best.pth')
state['checkpoint']=str(checkpoint);save()
while not stop:
    try:
        row=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
        free,used=map(int,row)
        state.update(checked=time.time(),free_mib=free,used_mib=used,samples=state['samples']+1 if free>=12000 else 0)
        save()
        if state['samples']>=4:
            if digest()!=expected:state.update(status='blocked_source_changed');save();break
            if (P/'multi_resume_launch.json').exists():state.update(status='blocked_duplicate_launch');save();break
            cmd=[str(F/'scripts/python_manip.sh'),str(F/'scripts/monitor_manip.py'),'--config','manip_hora16384_six_objects_gel_rotation','--gpu','0','--output',str(P/'multi_resume_monitor'),'--run-prefix','teacher_six_objects_gel16384_20261017_resumed','--checkpoint',str(checkpoint),'--max-steps','1500000000','--save-frequency','50','--max-restarts','2','--start-free-mib','12000']
            with (P/'multi_resume_supervisor.log').open('x') as log:
                proc=subprocess.Popen(cmd,cwd=F,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            record=dict(pid=proc.pid,command=cmd,time=time.time())
            (P/'multi_resume_launch.json').write_text(json.dumps(record,indent=2))
            state.update(status='supervisor_launched',supervisor_pid=proc.pid,monitor=str(P/'multi_resume_monitor/status.json'));save();break
    except (subprocess.SubprocessError,ValueError) as exc:
        state.update(last_error=str(exc),samples=0);save()
    for _ in range(30):
        if stop:break
        time.sleep(1)
if stop:state.update(status='stopped',finished=time.time());save()
