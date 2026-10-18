"""Merge verified object banks and check mixed-catalog reset under new domain draws."""
import isaacgym
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.utils.manip_control import tracking_action
from scripts.check_grasp_cache import contact_pairs
OUT=Path(__file__).resolve().parent
catalog=json.loads((OUT/'objects.json').read_text())
np.random.seed(20261017);torch.manual_seed(20261017);torch.set_num_threads(1)
merged=None;provenance={}
for obj in catalog:
    name=obj['name']
    path=ROOT/('cache/digit_sphere55_scale2560_inward90_checked_20261016.pt' if name=='sphere55' else f'cache/digit_{name}_pair_inward90_checked_20261017.pt')
    bank=torch.load(path,map_location='cpu')
    assert bank['objects']==[obj] and len(bank['states'][name])>=64
    assert bank['generation_action_reference']=='actual_position'
    if merged is None:
        merged={k:bank[k] for k in ['asset_sha256','collision_sha256','joint_names','generation_action_reference','generation_controller','grasp_contact_validation']}
        merged.update(objects=catalog,states={},targets={})
    for key in ['asset_sha256','collision_sha256','joint_names']:assert merged[key]==bank[key]
    assert torch.isfinite(bank['states'][name]).all() and torch.isfinite(bank['targets'][name]).all()
    merged['states'][name]=bank['states'][name];merged['targets'][name]=bank['targets'][name]
    provenance[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),count=len(bank['states'][name]),quality_validation=bank['quality_validation'])
merged['per_object_provenance']=provenance
path=ROOT/'cache/digit_six_objects_inward90_checked_20261017.pt'
if path.exists():raise FileExistsError(path)
torch.save(merged,path)
(OUT/'cache_manifest.json').write_text(json.dumps(dict(cache=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),objects=provenance),indent=2))
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
    cfg=compose(config_name='manip_hora16384_six_objects_gel_rotation',overrides=['stage=teacher','num_envs=384','rl_device=cpu'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
env=AllegroDigitTwoAxis(task,'cuda:0',-1,True);env.stage='grasps'
try:
    env.reset();desired=env.targets.clone();initial=env.object_state[:,:3].clone()
    failed=torch.zeros(env.num_envs,dtype=torch.bool);gcounts=torch.zeros(env.num_envs);noncounts=torch.zeros(env.num_envs)
    mapping=[]
    for i in range(env.num_envs):
        obj=catalog[int(env.object_ids[i])];states=merged['states'][obj['name']]
        match=((states[:,:16]-env.q[i]).abs().max(-1).values<1e-6)&((states[:,16:19]-initial[i]).abs().max(-1).values<1e-6)
        assert match.any(), f'Wrong cache reset in env {i}'
        mapping.append(int(match.nonzero()[0]))
    for step in range(600):
        env.step(tracking_action(env.q,env.targets,desired,env.actions,env.cfg))
        _,non=contact_pairs(env);failed|=env.terminal['failed']
        gcounts+=(env.task_gel_contacts.sum(-1)>=2).float();noncounts+=non.float()
        if (step+1)%100==0:print('mixed hold',step+1,'retained',int((~failed).sum()),flush=True)
    quality=(~failed)&(gcounts>=540)&(noncounts<=60)
    rows=[]
    for oid,obj in enumerate(catalog):
        mask=env.object_ids==oid
        rows.append(dict(object=obj['name'],n=int(mask.sum()),retained=int((~failed&mask).sum()),contact_quality_pass=int((quality&mask).sum()),mean_two_gel_fraction=float((gcounts[mask]/600).mean()),mean_nongel_fraction=float((noncounts[mask]/600).mean())))
    report=dict(seed=20261017,hold_seconds=30,reset_mapping_verified=True,rows=rows,criterion='No task failure, >=2 gel loads >0.1N on >=90% frames, nongel contacts <=10% frames; independent domain randomization, 64 resets per object',selected_cache_indices=mapping)
    (OUT/'mixed_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows),flush=True)
    if any(r['retained']<52 or r['contact_quality_pass']<48 for r in rows):raise RuntimeError('Mixed validation below launch gate; inspect object-specific report')
finally:
    print('CLEANUP_START',flush=True)
    env.close()
    print('CLEANUP_DONE',flush=True)
