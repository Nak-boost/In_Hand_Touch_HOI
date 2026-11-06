import json,pathlib,hashlib,subprocess
import torch
import yaml
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from hora.utils.grasp_validation import require_validation_contract
root=pathlib.Path('/root/In_Hand_Reconstruction/In_Hand_Manip');out=pathlib.Path(__file__).parent
manifest=json.loads((out/'cache_manifest.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for tag,record in manifest.items():
 path=pathlib.Path(record['cache']);assert sha(path)==record['sha256'];bank=torch.load(path,map_location='cpu')
 quality=json.loads(pathlib.Path(record['quality']).read_text());assert sha(record['quality'])==bank['quality_validation']['report_sha256']
 old=torch.load(quality['input_cache'],map_location='cpu');assert sha(quality['input_cache'])==quality['input_cache_sha256']
 for trial in quality['trials']:assert sha(trial['path'])==trial['sha256']
 for name,indices in quality['accepted_indices'].items():
  assert torch.equal(bank['states'][name],old['states'][name][indices])
  assert torch.equal(bank['targets'][name],old['targets'][name][indices])
  assert len(torch.unique(bank['states'][name],dim=0))==len(indices)

def differences(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):return sum([differences(a.get(k),b.get(k),path+'.'+k if path else k) for k in sorted(a.keys()|b.keys())],[])
 return [] if a==b else [dict(field=path,before=a,after=b)]
files=['configs/manip_hora16384_sphere55.yaml','configs/manip_hora16384_six_objects_gel_rotation.yaml','configs/manip_scale2560_sphere55.yaml']+[str(p.relative_to(root)) for p in (root/'configs').glob('manip_tabletop*.yaml')]
allowed={'train.network.mlp.units','train.network.priv_mlp.units','task.env.grasp.cache','task.env.manipulation.rotation_contact_multipliers'}
diffs={}
for f in files:
 old=yaml.safe_load(subprocess.check_output(['git','show','HEAD:In_Hand_Manip/'+f],cwd=root));new=yaml.safe_load((root/f).read_text());diffs[f]=differences(old,new)
 assert all(d['field'] in allowed for d in diffs[f]),diffs[f]
(out/'final_config_diff.json').write_text(json.dumps(diffs,indent=2)+'\n')
# Resolver registration without importing simulator packages.
for name,func in {'eq':lambda x,y:str(x).lower()==str(y).lower(),'contains':lambda x,y:x.lower() in y.lower(),'if':lambda p,a,b:a if p else b,'resolve_default':lambda d,a:d if a=='' else a}.items():
 if not OmegaConf.has_resolver(name):OmegaConf.register_new_resolver(name,func)
configs=['manip_hora16384_six_objects_gel_rotation','manip_hora16384_sphere55','manip_hora16384_sphere55_gel_rotation','manip_scale2560_sphere55']+[p.stem for p in (root/'configs').glob('manip_tabletop_inward90_*.yaml')]
with initialize_config_dir(version_base='1.3',config_dir=str(root/'configs')):
 for name in configs:
  cfg=compose(config_name=name);task=OmegaConf.to_container(cfg.task,resolve=True);bank=torch.load(root/task['env']['grasp']['cache'],map_location='cpu')
  require_validation_contract(bank,task);assert bank['objects']==task['env']['objects']
  assert list(cfg.train.network.mlp.units)==[256,256,128] and list(cfg.train.network.priv_mlp.units)==[128,64,8]
print('VERIFIED: cache content/provenance, unique states, control contracts, network sizes, authorized config diff only')
