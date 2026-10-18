"""Run one authorized experiment on physical GPU 1 and persist process status."""
from pathlib import Path
import csv
import datetime
import json
import os
import signal
import subprocess
import time

ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
OUT=Path(__file__).resolve().parent
RUN='teacher_sphere55_z_hora16384_pairfilter_20261016'

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def snapshot(pid=None):
    rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()))
    row=next(r for r in rows if int(r[0])==1)
    r=dict(gpu=1,uuid=row[1].strip(),total_mib=int(row[2]),used_mib=int(row[3]),utilization=int(row[4]))
    if pid is not None:
        apps=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines())
        r['process_mib']=sum(int(x[2]) for x in apps if len(x)==3 and x[0].strip()==str(pid) and x[1].strip()==r['uuid'])
    return r

def save(state):
    temp=OUT/'status.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(OUT/'status.json')

if (OUT/'status.json').exists() or (ROOT.parent/'runs/manip'/RUN).exists():
    raise FileExistsError('Refusing duplicate experiment launch')
baseline=snapshot()
if baseline['total_mib']-baseline['used_mib']<12000:
    raise RuntimeError('GPU 1 has less than 12000 MiB free; do not displace other work')
command=[str(ROOT/'scripts/run_manip.sh'),'--config-name','manip_hora16384_sphere55',
         'stage=teacher','run_name='+RUN,
         'checkpoint='+str(OUT/'probe_16384/runs/capacity/stage1_nn/last.pth')]
with (OUT/'train.log').open('x') as log, (OUT/'resources.jsonl').open('x') as resources:
    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
    state=dict(status='running',started=now(),supervisor_pid=os.getpid(),pid=process.pid,
        physical_gpu=1,run_name=RUN,run_dir=str(ROOT.parent/'runs/manip'/RUN),
        config='manip_hora16384_sphere55',command=command,baseline=baseline,
        process_peak_mib=0,log=str(OUT/'train.log'),minimum_free_mib=2048,
        maximum_process_mib=20000)
    save(state)
    while process.poll() is None:
        try:
            sample=dict(time=now(),**snapshot(process.pid))
            resources.write(json.dumps(sample)+'\n');resources.flush()
            state['process_peak_mib']=max(state['process_peak_mib'],sample['process_mib'])
            state['last_sample']=sample;state.pop('monitor_error',None)
            if sample['total_mib']-sample['used_mib'] < state['minimum_free_mib'] or sample['process_mib']>state['maximum_process_mib']:
                state.update(stop_reason='memory_headroom_limit',status='stopping_for_memory_pressure')
                save(state)
                process.send_signal(signal.SIGINT)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                break
        except Exception as error:
            state['monitor_error']=str(error)
        state['updated']=now();save(state)
        time.sleep(5)
    state.update(status=('stopped_for_memory_pressure' if state.get('stop_reason') else
                         'completed' if process.returncode==0 else 'failed'),
                 exit_code=process.returncode,finished=now(),updated=now())
    save(state)
