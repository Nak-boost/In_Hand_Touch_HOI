import subprocess,json,datetime
from pathlib import Path
p=Path(__file__).resolve().parent
report=json.loads((p/'mixed_validation.json').read_text())
assert report['reset_mapping_verified']
assert all(r['retained']>=52 and r['contact_quality_pass']>=48 for r in report['rows'])
frozen=p/'frozen_project'
cmd=[str(frozen/'scripts/python_manip.sh'),str(frozen/'scripts/monitor_manip.py'),'--config','manip_hora16384_six_objects_gel_rotation','--gpu','0','--output',str(p/'multi_monitor'),'--run-prefix','teacher_six_objects_gel16384_20261017','--max-steps','1500000000','--save-frequency','50','--max-restarts','2','--start-free-mib','10000']
if (p/'multi_launch.json').exists():raise FileExistsError('Refusing duplicate launch')
with (p/'multi_supervisor.log').open('x') as log:
 proc=subprocess.Popen(cmd,cwd=frozen,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record=dict(pid=proc.pid,command=cmd,time=datetime.datetime.now().isoformat(),start_memory_reason='16384 single run about 6 GiB, prior baseline peak about 7.4 GiB; require 10000 MiB free at launch and 2048 MiB while running')
(p/'multi_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
