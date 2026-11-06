import pathlib,json,hashlib,os,xml.etree.ElementTree as ET
from omegaconf import OmegaConf
import torch
base=pathlib.Path('/root/In_Hand_Reconstruction'); project=base/'In_Hand_Manip'; out=base/'research/full_task_audit_20261018'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for pid in (224934,354488,366939):
 cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0'); env=dict(x.split('=',1) for x in pathlib.Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in x)
 src=pathlib.Path(cmd[1]).parent; run=base/'runs/manip'/next(x.split('=',1)[1] for x in cmd if x.startswith('run_name=')); cfg=OmegaConf.to_container(OmegaConf.load(run/'config.yaml'),resolve=True)
 task=cfg['task']; asset=src/task['env']['asset']['handAsset']; cache=src/task['env']['grasp']['cache']; bank=torch.load(cache,map_location='cpu'); root=ET.parse(asset).getroot()
 files=['train_manip.py','hora/tasks/allegro_digit_two_axis.py','hora/tasks/allegro_digit_tabletop.py','hora/tasks/base/vec_task.py','hora/algo/ppo/ppo.py','hora/utils/object_contacts.py','hora/algo/models/tactile_student.py','hora/utils/manip_control.py']
 r=dict(pid=pid,command=cmd,source=str(src),run=str(run),cuda_visible=env.get('CUDA_VISIBLE_DEVICES'),cuda_order=env.get('CUDA_DEVICE_ORDER'),config=cfg,asset=str(asset),asset_sha256=sha(asset),collision_shapes=len(root.findall('.//collision')),cache=str(cache),cache_sha256=sha(cache),cache_meta={k:v for k,v in bank.items() if k not in ('states','targets')},cache_counts={k:len(v) for k,v in bank['states'].items()},files={f:dict(runtime=sha(src/f),workspace=sha(project/f),saved=sha(run/'source'/f)) for f in files})
 records.append(r)
(out/'inventory.json').write_text(json.dumps(records,indent=2,default=str))
for r in records:
 c=r['config'];t=c['task']['env'];print(json.dumps(dict(pid=r['pid'],gpu=r['cuda_visible'],run=r['run'],task=c['task']['name'],shapes=r['collision_shapes'],cache_counts=r['cache_counts'],cache_meta=r['cache_meta'],manip=t['manipulation'],controller=t['controller'],digit=t['digit'],domain=t['domain'],ppo=c['train']['ppo'],network=c['train']['network'],source_different=[f for f,h in r['files'].items() if h['runtime']!=h['workspace']],source_saved_match=all(h['runtime']==h['saved'] for h in r['files'].values())),default=str))
