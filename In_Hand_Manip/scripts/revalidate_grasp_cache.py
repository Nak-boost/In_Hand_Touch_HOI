"""Validate all cached states under one domain seed using the policy action path.

Run independent seeds in separate processes, then intersect accepted indices.
No auto-reset, no force/threshold changes, no direct target writes during rollout.
"""
import isaacgym
from isaacgym import gymapi
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.manip_control import tracking_action
from hora.utils.grasp_validation import validation_contract, duration_steps, withdrawal_fraction
from scripts.check_grasp_cache import install


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--margin-force',type=float,default=.15)
    p.add_argument('--retractions',action='store_true')
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    source=(ROOT/a.cache).resolve();bank=torch.load(source,map_location='cpu')
    torch.set_num_threads(1);np.random.seed(a.seed);torch.manual_seed(a.seed)
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
        cfg=compose(config_name=a.config)
    task=OmegaConf.to_container(cfg.task,resolve=True)
    catalog=task['env']['objects'];assert catalog==bank['objects']
    m=len(catalog);width=max(len(bank['states'][o['name']]) for o in catalog);n=width*m
    task['env']['numEnvs']=n;task['rl_device']='cpu';task['stage']='teacher'
    task['env']['grasp']['cache']=str(source)
    if not task['env']['manipulation'].get('object_contact_filter'):
        raise ValueError('Validation requires the task object contact filter')
    if a.margin_force<task['env']['digit']['contact_threshold']:
        raise ValueError('Margin must not weaken the task contact threshold')
    tabletop=task['name']=='AllegroDigitTabletop'
    cls=AllegroDigitTabletop if tabletop else AllegroDigitTwoAxis
    env=cls(task,'cuda:0',-1,True);env.stage='grasps'
    indices=[j%len(bank['states'][o['name']]) for j in range(width) for o in catalog]
    states=torch.stack([bank['states'][catalog[i%m]['name']][k] for i,k in enumerate(indices)])
    targets=torch.stack([bank['targets'][catalog[i%m]['name']][k] for i,k in enumerate(indices)])
    bodies=env.gym.get_asset_rigid_body_count(env.hand_asset)
    masks=np.array([env.collision_spec['body_masks'][s] for s in env.gym.get_asset_rigid_body_names(env.hand_asset)]+[0]+([env.collision_spec['table_filter']] if tabletop else []),dtype=np.int64)

    def pairs():
        other=torch.zeros(n,dtype=torch.bool);support=torch.zeros(n,dtype=torch.bool)
        for i,e in enumerate(env.envs):
            contacts=env.gym.get_env_rigid_contacts(e)
            x,y=contacts['body0'],contacts['body1'];positive=contacts['lambda']>0
            valid=positive&(x>=0)&(y>=0)
            if np.any(masks[x[valid]]&masks[y[valid]]):raise AssertionError('Filtered physical contact')
            ob=env.object_contact_reader.object_bodies[i]
            hit=positive&((x==ob)|(y==ob));partner=np.where(x[hit]==ob,y[hit],x[hit])
            other[i]=bool(np.any((partner>=0)&(partner<bodies)&~np.isin(partner,env.gel_ids)))
            if tabletop:support[i]=bool(np.any(partner==bodies+1))
        return other,support

    def rollout(joint=None,finger=None):
        install(env,states,targets)
        steps=duration_steps(a.seconds if joint is None else 4.,env.control_dt)
        failed=torch.zeros(n,dtype=torch.bool);two=torch.zeros(n);margin=torch.zeros(n)
        nongel=torch.zeros(n);table=torch.zeros(n);force=torch.zeros(n);sat=torch.zeros(n)
        missing=torch.zeros(n);maxmissing=torch.zeros(n);baseq=env.q[:,joint].clone() if joint is not None else None
        opened=torch.zeros(n);initial=torch.zeros(n,dtype=torch.bool);recovered=torch.zeros(n);late_steps=0
        gain=env.kp.clone()
        for step in range(steps):
            t=(step+1)*env.control_dt;desired=targets.clone()
            if joint is not None:desired[:,joint]=(desired[:,joint]-.06*withdrawal_fraction(t)).clamp(env.lower[joint],env.upper[joint])
            act=tracking_action(env.q,env.targets,desired,env.actions,env.cfg)
            obs,reward,_,_=env.step(act)
            if not torch.isfinite(reward).all() or not torch.isfinite(env.q).all():raise FloatingPointError('Nonfinite validation')
            gel=env.task_gel_contacts;loads=env.object_gel_normal_force
            other,support=pairs();failed|=env.terminal['failed'];two+=(gel.sum(-1)>=2).float()
            margin+=((loads>a.margin_force).sum(-1)>=2).float();nongel+=other.float();table+=support.float()
            force=torch.maximum(force,env.contact_forces[:,env.gel_ids].norm(dim=-1).max(-1).values)
            sat+=(act.abs()>=1-1e-6).float().mean(-1)
            missing=torch.where(gel.any(-1),torch.zeros_like(missing),missing+1);maxmissing=torch.maximum(maxmissing,missing)
            if joint is not None:
                if abs(t-1)<env.control_dt/2:baseq=env.q[:,joint].clone();initial=gel[:,finger].clone()
                if 1<t<=2:opened=torch.maximum(opened,baseq-env.q[:,joint])
                if t>3.5:recovered+=(gel.sum(-1)>=2).float();late_steps+=1
            if (step+1)%200==0:print('progress',a.seed,'hold' if joint is None else 'retract',step+1,'retained',int((~failed).sum()),flush=True)
        passed=(~failed)&(two/steps>=.9)&(margin/steps>=.9)&(nongel/steps<=.1)
        limits=torch.minimum(states[:,:16]-env.lower,env.upper-states[:,:16])
        if tabletop:passed&=(table/steps>=.9)&(force<=task['env']['tabletop']['initialization']['max_gel_force'])
        else:passed&=((limits<.03).sum(-1)<=2)
        data=dict(passed=passed.tolist(),failed=failed.tolist(),two_contact_fraction=(two/steps).tolist(),margin_contact_fraction=(margin/steps).tolist(),nongel_fraction=(nongel/steps).tolist(),table_fraction=(table/steps).tolist(),max_gel_net_force_N=force.tolist(),max_missing_seconds=(maxmissing*env.control_dt).tolist(),action_saturation_fraction=(sat/steps).tolist(),kp=gain.tolist())
        if joint is not None:data.update(achieved_withdrawal_rad=opened.tolist(),initial_contact=initial.tolist(),retraction_success=((~failed)&initial&(opened>=.015)&(recovered>=.9*late_steps)).tolist())
        return data

    try:
        hold=rollout();accepted={}
        for oid,obj in enumerate(catalog):
            mask=torch.ones(len(bank['states'][obj['name']]),dtype=torch.bool)
            for i in range(oid,n,m):mask[indices[i]] &= hold['passed'][i]
            accepted[obj['name']]=mask.nonzero().flatten().tolist()
        report=dict(config=a.config,seed=a.seed,input_cache=str(source),input_cache_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),asset_sha256=env.asset_hash,collision_sha256=env.collision_hash,validation_contract=validation_contract(task),control_dt=env.control_dt,hold_seconds=duration_steps(a.seconds,env.control_dt)*env.control_dt,margin_force_N=a.margin_force,objects=catalog,env_cache_indices=indices,domain_params=env.domain_params.tolist(),accepted_indices=accepted,hold=hold,retractions=[])
        print('HOLD_RESULT',a.seed,{k:len(v) for k,v in accepted.items()},flush=True)
        if a.retractions:
            for finger,num in enumerate([1,5,9,13]):
                result=rollout(env.canonical_order.index(num),finger);result['finger']=finger
                report['retractions'].append(result)
                print('RETRACTION_RESULT',finger,sum(result['retraction_success']),flush=True)
        report['note']='All candidates tested; duplicate padding must pass every copy. Retraction diagnostic is not a hard gate for coordinated finger gait. No auto-reset.'
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        print('CLOSING',flush=True);env.close();print('CLOSED',flush=True)


if __name__=='__main__':main()
