"""Cross-check the six configured training caches and their quality evidence."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import torch
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from hora.utils.hand_collision import collision_spec,collision_signature
from hora.utils.asset_geometry import verify_geometry_manifest
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');OUT=Path(__file__).resolve().parent
results=[]
for obj in ['sphere55','sphere65','box45','box55','capsule70','capsule80']:
 with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):cfg=compose(config_name='manip_official_'+obj)
 asset=ROOT/cfg.task.env.asset.handAsset;verify_geometry_manifest(asset)
 cache=ROOT/cfg.task.env.grasp.cache;b=torch.load(cache,map_location='cpu')
 states=b['states'][obj];targets=b['targets'][obj]
 report=Path(b['quality_validation']['report']);q=json.loads(report.read_text())
 assert hashlib.sha256(report.read_bytes()).hexdigest()==b['quality_validation']['report_sha256']
 assert b['asset_sha256']==q['asset_sha256']==hashlib.sha256(asset.read_bytes()).hexdigest()
 assert b['collision_sha256']==q['collision_sha256']==collision_signature(collision_spec(asset))
 assert b['objects']==OmegaConf.to_container(cfg.task.env.objects,resolve=True)
 assert len(states)==q['accepted'] and len(states)>=16
 assert len(states.unique(dim=0))==len(states)
 assert states.shape[1]==23 and targets.shape==(len(states),16)
 assert torch.isfinite(states).all() and torch.isfinite(targets).all()
 limits={j.attrib['name']:[float(j.find('limit').attrib[k]) for k in ['lower','upper']] for j in ET.parse(asset).getroot().findall('joint') if j.attrib['type']=='revolute'}
 lower=torch.tensor([limits[n][0] for n in b['joint_names']]);upper=torch.tensor([limits[n][1] for n in b['joint_names']])
 assert (targets>=lower-1e-6).all() and (targets<=upper+1e-6).all()
 margin=torch.minimum(targets-lower,upper-targets)
 results.append(dict(object=obj,cache=str(cache),report=str(report),tested=q['n'],retained=q['retained'],count=len(states),retraction_pass=q['retraction_pass_among_hold'],mean_preload_rad=(targets-states[:,:16]).abs().mean().item(),max_preload_rad=(targets-states[:,:16]).abs().max().item(),minimum_target_margin_rad=margin.min().item(),target_near_limit_fraction=(margin<.03).float().mean().item(),asset_sha256=b['asset_sha256'],collision_sha256=b['collision_sha256']))
(OUT/'checked_cache_validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
