"""Prepare and validate inward DIGIT grasps through bounded policy target tracking."""
import isaacgym
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import pybullet as pb
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.utils.manip_control import tracking_action
from scripts.check_grasp_cache import install, contact_pairs

OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generate','validate'],required=True)
p.add_argument('--object',required=True);p.add_argument('--count',type=int,default=256);p.add_argument('--batches',type=int,default=20)
a=p.parse_args()
obj_config=next(x for x in json.loads((OUT/'objects.json').read_text()) if x['name']==a.object)
OUT=OUT/'grasps'/a.object
candidate_path=ROOT/f'cache/digit_{a.object}_pair_inward90_candidates_20261017.pt'
checked_path=ROOT/f'cache/digit_{a.object}_pair_inward90_checked_20261017.pt'
np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
bank=None
if a.mode=='validate':
    bank=torch.load(candidate_path,map_location='cpu');n=len(bank['states'][a.object])
else:n=256
with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
    cfg=compose(config_name=f'grasp_pair_{a.object}_20261017',overrides=['stage=grasps','rl_device=cpu',f'num_envs={n}'])
task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='grasps'
env=AllegroDigitTwoAxis(task,'cuda:0',-1,True)

def step_hold(desired):
    env.step(tracking_action(env.q,env.targets,desired,env.actions,env.cfg))

try:
    if a.mode=='generate':
        if candidate_path.exists():raise FileExistsError(candidate_path)
        seeds=json.loads((OUT/'static_candidates.json').read_text())
        pb.connect(pb.DIRECT)
        quat=pb.multiplyTransforms([0,0,0],pb.getQuaternionFromEuler([0,-np.pi/2,0]),[0,0,0],pb.getQuaternionFromEuler([np.pi/2,0,0]))[1]
        hand=pb.loadURDF(str(ROOT/cfg.task.env.asset.handAsset),[0,0,.5],quat,useFixedBase=True)
        size=obj_config['size'];shape=obj_config['shape'];rot=[0,0,0,1]
        if shape=='sphere':geom=pb.createCollisionShape(pb.GEOM_SPHERE,radius=size[0]/2)
        elif shape=='box':geom=pb.createCollisionShape(pb.GEOM_BOX,halfExtents=np.array(size)/2)
        else:
            geom=pb.createCollisionShape(pb.GEOM_CAPSULE,radius=size[1]/2,height=size[0]-size[1]);rot=pb.getQuaternionFromEuler([0,np.pi/2,0])
        obj=pb.createMultiBody(baseMass=0,baseCollisionShapeIndex=geom)
        pb.changeDynamics(obj,-1,collisionMargin=0)
        links={};joints={}
        for j in range(pb.getNumJoints(hand)):
            info=pb.getJointInfo(hand,j);links[info[12].decode()]=j;pb.changeDynamics(hand,j,collisionMargin=0)
            if info[2]==pb.JOINT_REVOLUTE:joints[int(info[1].decode().split('_')[1].split('.')[0])]=j
        for seed in seeds:
            q=np.array(seed['q']);delta=np.zeros(16)
            for k,j in joints.items():pb.resetJointState(hand,j,q[k])
            pb.resetBasePositionAndOrientation(obj,seed['center'],rot)
            for finger in seed['selected']:
                link=links['digit_'+['index','middle','ring','thumb'][finger]+'_gel'];grad=[]
                for k in range(4*finger,4*finger+4):
                    distances=[]
                    for eps in [-.001,.001]:
                        pb.resetJointState(hand,joints[k],q[k]+eps)
                        distances.append(min(c[8] for c in pb.getClosestPoints(hand,obj,.15,linkIndexA=link)))
                    pb.resetJointState(hand,joints[k],q[k]);grad.append((distances[1]-distances[0])/.002)
                grad=np.array(grad);delta[4*finger:4*finger+4]=np.clip(-.0015*grad/(grad@grad+1e-8),-.04,.04)
            seed['preload_delta']=delta.tolist()
        pb.disconnect();(OUT/'seeds_with_preload.json').write_text(json.dumps(seeds,indent=2))
        elite=torch.empty(0,23);elite_targets=torch.empty(0,16);records=[]
        for batch in range(a.batches):
            indices=torch.arange(n)%len(seeds)
            states=torch.zeros(n,23);states[:,22]=1
            states[:,:16]=torch.tensor([seeds[i]['q'] for i in indices])[...,env.canonical_order]
            states[:,16:19]=torch.tensor([seeds[i]['center'] for i in indices])
            states[:,:16]+=(torch.rand(n,16)*2-1)*.02
            states[:,16:18]+=(torch.rand(n,2)*2-1)*.0015
            preload=torch.tensor([seeds[i]['preload_delta'] for i in indices])[...,env.canonical_order]
            desired=states[:,:16]+preload*(.6+torch.rand(n,1)*1.6)
            if len(elite):
                k=n//2;chosen=torch.randint(len(elite),(k,));noise=(torch.rand(k,16)*2-1)*.008
                states[:k]=elite[chosen];states[:k,:16]+=noise
                desired[:k]=elite_targets[chosen]+noise
                states[:k,16:18]+=(torch.rand(k,2)*2-1)*.0005
            states[:,:16]=torch.maximum(torch.minimum(states[:,:16],env.upper),env.lower)
            desired=torch.maximum(torch.minimum(desired,env.upper),env.lower)
            install(env,states,desired)
            valid=torch.ones(n,dtype=torch.bool);gel_count=torch.zeros(n);non_count=torch.zeros(n)
            for step in range(80):
                step_hold(desired)
                if step>=25:
                    valid &= ~env.terminal['failed'] & (env.object_state[:,7:10].norm(dim=-1)<.1) & (env.object_state[:,10:13].norm(dim=-1)<1.)
                    _,non=contact_pairs(env)
                    gel_count+=(env.task_gel_contacts.sum(-1)>=2).float();non_count+=non.float()
            keep=valid & (gel_count>=53) & (non_count<=5)
            physical=torch.cat([env.q,env.object_state[:,:7]],-1).clone()
            elite=torch.cat([elite,physical[keep]]);elite_targets=torch.cat([elite_targets,env.targets[keep].clone()])
            record=dict(batch=batch+1,accepted=int(keep.sum()),total=len(elite),retained=int(valid.sum()),two_gels=int((gel_count>=53).sum()),non_gel_free=int((non_count<=5).sum()))
            records.append(record);print(json.dumps(record),flush=True)
            if len(elite)>=a.count:break
        if len(elite)<64:raise RuntimeError('Insufficient physical candidates; inspect proposals without relaxing contact criteria')
        bank=dict(asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,objects=env.catalog,joint_names=env.joint_names,
            states={a.object:elite[:a.count]},targets={a.object:elite_targets[:a.count]},
            generation_action_reference=env.action_reference,generation_controller='bounded_target_tracking_v1',
            grasp_contact_validation='object_pair_normal_loads_v1',generation=dict(seed=42,config=OmegaConf.to_container(cfg,resolve=True),batches=records))
        torch.save(bank,candidate_path)
        (OUT/'generation.json').write_text(json.dumps(dict(count=len(bank['states'][a.object]),batches=records),indent=2))
        print('SAVED',candidate_path,flush=True)
    else:
        if checked_path.exists():raise FileExistsError(checked_path)
        assert bank['asset_sha256']==env.asset_hash and bank['collision_sha256']==env.collision_hash
        states=bank['states'][a.object];desired=bank['targets'][a.object]
        install(env,states,desired)
        failed=torch.zeros(n,dtype=torch.bool);gcounts=torch.zeros(n);noncounts=torch.zeros(n)
        maxdisp=torch.zeros(n);maxerror=torch.zeros(n);fingers=torch.zeros(n,4)
        margin=torch.minimum(states[:,:16]-env.lower,env.upper-states[:,:16])
        for step in range(600):
            step_hold(desired);_,non=contact_pairs(env)
            failed|=env.terminal['failed'];gcounts+=(env.task_gel_contacts.sum(-1)>=2).float();noncounts+=non.float()
            fingers+=env.task_gel_contacts.float()
            maxdisp=torch.maximum(maxdisp,(env.object_state[:,:3]-states[:,16:19]).norm(dim=-1))
            maxerror=torch.maximum(maxerror,(env.targets-desired).abs().max(-1).values)
            if (step+1)%100==0:print('hold',step+1,'retained',int((~failed).sum()),flush=True)
        selected=(~failed)&(gcounts>=540)&(noncounts<=60)&((margin<.03).sum(-1)<=2)
        report=dict(input_cache=str(candidate_path.relative_to(ROOT)),input_cache_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,n=n,seed=42,hold_seconds=600*env.control_dt,
            accepted=int(selected.sum()),retained=int((~failed).sum()),action_reference=env.action_reference,
            controller='bounded_target_tracking_v1; same smoothing/action limits/PD as policy',
            criteria='30s no task failure; >=2 gel-object loads >0.1N on >=90% frames; non-gel object contact <=10%; <=2 joints within .03rad of limit',
            accepted_indices=selected.nonzero().flatten().tolist(),hold_failed=failed.tolist(),
            gel_ge2_fraction=(gcounts/600).tolist(),non_gel_fraction=(noncounts/600).tolist(),
            finger_contact_fraction=(fingers/600).tolist(),max_displacement_m=maxdisp.tolist(),max_target_error_rad=maxerror.tolist(),
            minimum_joint_margin_rad=margin.min(-1).values.tolist())
        quality=OUT/'quality.json';quality.write_text(json.dumps(report,indent=2))
        if selected.sum()<64:raise RuntimeError('Too few 30s validated grasps; do not train')
        bank['states']={a.object:states[selected].clone()};bank['targets']={a.object:desired[selected].clone()}
        bank['quality_validation']=dict(report=str(quality),report_sha256=hashlib.sha256(quality.read_bytes()).hexdigest(),action_reference=env.action_reference)
        torch.save(bank,checked_path);print('SAVED',checked_path,'accepted',int(selected.sum()),flush=True)
finally:
    print('CLEANUP_START',flush=True)
    env.close()
    print('CLEANUP_DONE',flush=True)
