"""Validate every cached grasp with real contact pairs and gentle single-finger retraction."""
import isaacgym
from isaacgym import gymapi, gymtorch
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis


def install(env, states, targets=None):
    env.reset()
    ids=torch.arange(env.num_envs,device=env.device)
    env.q.copy_(states[:,:16]);env.qdot.zero_();env.targets.copy_(env.q if targets is None else targets)
    if hasattr(env,'initial_q'):env.initial_q.copy_(env.q)
    env.root_states[env.object_indices,:7]=states[:,16:]
    env.root_states[env.object_indices,7:]=0
    env.initial_position.copy_(states[:,16:19]);env.prev_pose.copy_(states[:,16:])
    hi=env.hand_indices.to(torch.int32);oi=env.object_indices.to(torch.int32)
    env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_states),gymtorch.unwrap_tensor(hi),len(ids))
    env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_states),gymtorch.unwrap_tensor(oi),len(ids))
    env._refresh();env._observe()


def contact_pairs(env, active=None):
    gels=torch.zeros(env.num_envs,4,dtype=torch.bool)
    non=torch.zeros(env.num_envs,dtype=torch.bool)
    for i,e in enumerate(env.envs):
        if active is not None and not bool(active[i]):continue
        ob=env.gym.get_actor_rigid_body_index(e,1,0,gymapi.DOMAIN_ENV)
        for c in env.gym.get_env_rigid_contacts(e):
            a,b=int(c['body0']),int(c['body1'])
            other=b if a==ob else a if b==ob else -1
            if c['lambda']<=0 or other<0:continue
            if other in env.gel_ids:gels[i,env.gel_ids.index(other)]=True
            else:non[i]=True
    return gels,non


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--output-cache',required=True)
    p.add_argument('--hold-steps',type=int,default=600)
    args=p.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    dest=ROOT/args.output_cache
    if dest.exists():raise FileExistsError(dest)
    bank=torch.load(ROOT/args.cache,map_location='cpu');name=bank['objects'][0]['name'];states=bank['states'][name]
    targets=bank.get('targets',{}).get(name,states[:,:16])
    n=len(states);np.random.seed(20261014);torch.manual_seed(20261014)
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
        cfg=compose(config_name=args.config,overrides=['stage=teacher','pipeline=cpu','rl_device=cpu',f'num_envs={n}',
            'graphics_device_id=-1','task.env.grasp.cache='+args.cache])
    task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
    env=AllegroDigitTwoAxis(task,'cuda:0',-1,True);env.stage='grasps'
    try:
        install(env,states,targets)
        failed=torch.zeros(n,dtype=torch.bool);counts=torch.zeros(n);noncounts=torch.zeros(n)
        maxdisp=torch.zeros(n);minmargin=torch.minimum(states[:,:16]-env.lower,env.upper-states[:,:16])
        for step in range(args.hold_steps):
            env.step(torch.zeros_like(env.actions));gel,non=contact_pairs(env)
            failed|=env.terminal['failed'];counts+=(gel.sum(-1)>=2).float();noncounts+=non.float()
            maxdisp=torch.maximum(maxdisp,(env.object_state[:,:3]-states[:,16:19]).norm(dim=-1))
        hold=(~failed)&(counts/args.hold_steps>=.9)&(noncounts/args.hold_steps<=.1)&((minmargin<.03).sum(-1)<=2)
        print(json.dumps(dict(phase='hold',n=n,retained=int((~failed).sum()),hold_pass=int(hold.sum()))),flush=True)
        hold_failed=failed.clone()
        results=[]
        for finger,number in enumerate([1,5,9,13]):
            install(env,states,targets);failed=torch.zeros(n,dtype=torch.bool);regained=torch.zeros(n)
            initial_contact=torch.zeros(n,dtype=torch.bool);base_q=torch.zeros(n);actual_open=torch.zeros(n)
            joint=env.canonical_order.index(number)
            for step in range(80):
                # Hold 1 s, withdraw 0.08 rad over 0.5 s, hold 0.5 s,
                # restore over 0.5 s, then hold 1.5 s. Targets always stay inside limits.
                amount=0 if step<20 else min((step-19)/10,1) if step<40 else max(1-(step-39)/10,0)
                env.targets.copy_(targets);env.targets[:,joint]=(targets[:,joint]-.08*amount).clamp(env.lower[joint],env.upper[joint])
                env.step(torch.zeros_like(env.actions));gel,non=contact_pairs(env);failed|=env.terminal['failed']
                if step==19:initial_contact=gel[:,finger].clone();base_q=env.q[:,joint].clone()
                if 20<=step<40:actual_open=torch.maximum(actual_open,base_q-env.q[:,joint])
                if step>=70:regained+=(gel.sum(-1)>=2).float()
            success=(~failed)&initial_contact&(actual_open>=.02)&(regained>=9)
            print(json.dumps(dict(phase='retraction',finger=finger,success=int(success.sum()))),flush=True)
            results.append(dict(finger=['index','middle','ring','thumb'][finger],success=success.tolist(),
                initial_contact=initial_contact.tolist(),achieved_open_rad=actual_open.tolist(),failed=failed.tolist()))
        retraction=torch.tensor([r['success'] for r in results]).any(0)
        # Uncoordinated one-finger withdrawal is diagnostic: a learned gait may first add another contact.
        selected=hold
        report=dict(config=args.config,input_cache=args.cache,input_cache_sha256=hashlib.sha256((ROOT/args.cache).read_bytes()).hexdigest(),
            asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,n=n,seed=20261014,
            hold_seconds=args.hold_steps*env.control_dt,hold_pass=int(hold.sum()),retained=int((~hold_failed).sum()),
            accepted=int(selected.sum()),retraction_pass_among_hold=int((hold & retraction).sum()),criteria='30 s no failure, >=2 real gel-object contacts on >=90% frames, non-gel object contact <=10%, <=2 joints within .03 rad of limit; single-finger .08 rad retraction is reported separately, not a hard gate for coordinated manipulation.',
            hold_failed=hold_failed.tolist(),gel_ge2_fraction=(counts/args.hold_steps).tolist(),non_gel_fraction=(noncounts/args.hold_steps).tolist(),
            max_displacement_m=maxdisp.tolist(),minimum_joint_margin_rad=minmargin.min(-1).values.tolist(),
            hold_pass_per_state=hold.tolist(),accepted_indices=selected.nonzero(as_tuple=False).flatten().tolist(),retractions=results)
        output.write_text(json.dumps(report,indent=2)+'\n')
        if selected.any():
            bank['states']={name:states[selected].clone()};bank['targets']={name:targets[selected].clone()};bank['quality_validation']=dict(report=str(output.resolve()),report_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),seed=20261014)
            dest.parent.mkdir(parents=True,exist_ok=True);torch.save(bank,dest)
        print(json.dumps({k:report[k] for k in ['config','n','hold_pass','accepted']}),flush=True)
    finally:env.close()

if __name__=='__main__':main()
