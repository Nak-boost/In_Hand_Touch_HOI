"""Search/refine light tabletop grasps using the policy's bounded action path."""
import isaacgym
from isaacgym import gymtorch
import argparse, json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.manip_control import tracking_action
from hora.utils.tabletop_contacts import object_contact_summary
from scripts.check_grasp_cache import install


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output-cache',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--count',type=int,default=128)
    p.add_argument('--envs',type=int,default=128)
    p.add_argument('--batches',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    dest=Path(a.output_cache).resolve(); reportpath=Path(a.report).resolve()
    if dest.exists() or reportpath.exists():raise FileExistsError('Choose unused candidate output paths')
    cfg=OmegaConf.load(ROOT/'configs'/(a.config+'.yaml'))
    task=OmegaConf.to_container(cfg.task,resolve=True)
    task['stage']='grasps';task['env']['numEnvs']=a.envs;task['rl_device']='cpu'
    task['sim']['use_gpu_pipeline']=False
    if not task['env']['manipulation']['object_contact_filter']:
        raise ValueError('This validator requires object-only force signals')
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    env=AllegroDigitTabletop(task,'cuda:0',-1,True)
    name=env.catalog[0]['name'];rows=[];elite=torch.empty(0,23);targets=torch.empty(0,16)
    gc=task['env']['grasp'];init=task['env']['tabletop']['initialization']
    try:
        for batch in range(a.batches):
            env.reset();closed=env.targets.clone()
            preload=torch.tensor(gc.get('preload_delta',[0.]*16))[env.canonical_order]
            closed=(closed+preload).clamp(env.lower,env.upper)
            opened=closed.clone()
            for number in [1,5,9,13]:
                j=env.canonical_order.index(number)
                opened[:,j]=(opened[:,j]-init['opening_delta']).clamp(env.lower[j],env.upper[j])
            states=torch.cat([opened,env.object_state[:,:7]],dim=-1)
            if len(elite):
                k=3*a.envs//4;ids=torch.randint(len(elite),(k,))
                states[:k]=elite[ids]
                noise=(torch.rand(k,16)*2-1)*.004
                states[:k,:16]=(states[:k,:16]+noise).clamp(env.lower,env.upper)
                states[:k,16:18]+=(torch.rand(k,2)*2-1)*.0004
                opened[:k]=states[:k,:16]
                closed[:k]=(targets[ids]+noise).clamp(env.lower,env.upper)
            install(env,states,opened)
            good=torch.zeros(a.envs);two=torch.zeros_like(good);table=torch.zeros_like(good);other=torch.zeros_like(good)
            valid=torch.ones(a.envs,dtype=torch.bool);maxforce=torch.zeros(a.envs)
            moving=torch.zeros(a.envs,dtype=torch.bool);failed=torch.zeros_like(moving)
            finger_counts=torch.zeros(a.envs,4)
            for step in range(init['close_steps']+init['hold_steps']):
                alpha=min(1.,(step+1)/init['close_steps'])
                desired=opened+alpha*(closed-opened)
                env.step(tracking_action(env.q,env.targets,desired,env.actions,env.cfg))
                if step < init['close_steps']:continue
                c=object_contact_summary(env)
                effective=env.object_gel_normal_force>init['min_gel_force']
                rawforce=env.contact_forces[:,env.gel_ids].norm(dim=-1)
                two+=(effective.sum(-1)>=2).float();table+=c['table'].float();other+=c['other_hand'].float()
                finger_counts+=effective.float()
                good+=((effective.sum(-1)>=2)&c['table']&~c['other_hand']).float()
                maxforce=torch.maximum(maxforce,rawforce.max(-1).values)
                failed|=env.terminal['failed'];moving|=env.object_state[:,7:10].norm(dim=-1)>=.05
                if c['forbidden']:raise RuntimeError('Forbidden table/gel collision detected')
            valid=(good/init['hold_steps']>=.9)&~failed&~moving&(maxforce<=init['max_gel_force'])
            valid&=(effective.sum(-1)>=2)&c['table']&~c['other_hand']
            new=torch.cat([env.q,env.object_state[:,:7]],dim=-1)[valid].clone()
            elite=torch.cat([elite,new]);targets=torch.cat([targets,env.targets[valid].clone()])
            row=dict(batch=batch+1,accepted=int(valid.sum()),total=len(elite),
                two_gels=int((two/init['hold_steps']>=.9).sum()),table_supported=int((table/init['hold_steps']>=.9).sum()),
                non_gel_free=int((other/init['hold_steps']<=.1).sum()),force_ok=int((maxforce<=init['max_gel_force']).sum()),
                not_failed=int((~failed).sum()),not_moving=int((~moving).sum()),
                finger_contact_fraction=(finger_counts/init['hold_steps']).mean(0).tolist())
            rows.append(row);print(json.dumps(row),flush=True)
            reportpath.write_text(json.dumps(dict(config=a.config,seed=a.seed,accepted=min(len(elite),a.count),batches=rows),indent=2)+'\n')
            if len(elite)>=a.count:break
        if len(elite)<a.count:raise RuntimeError('Insufficient validated candidates: '+str(len(elite)))
        bank=dict(asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,objects=env.catalog,
            joint_names=env.joint_names,grasp_contact_validation='tabletop_object_normal_load_refined_v1',
            generation_action_reference=env.action_reference,generation_controller='bounded_target_tracking_v1',
            scene=env.table_cfg,states={name:elite[:a.count]},targets={name:targets[:a.count]},
            generation=dict(config=a.config,seed=a.seed,report=str(reportpath)))
        torch.save(bank,dest);print('Saved',dest,flush=True)
    finally:env.close()


if __name__=='__main__':main()
