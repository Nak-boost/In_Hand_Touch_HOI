"""Evaluate a privileged teacher on tabletop starts with true object contacts."""
import isaacgym
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import require_action_reference
from scripts.check_grasp_cache import install


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--checkpoint',default='last')
    p.add_argument('--output',required=True);p.add_argument('--envs',type=int,default=64)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seed',type=int,default=215)
    a=p.parse_args();run=Path(a.run_dir);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    cfg=OmegaConf.load(run/'config.yaml')
    if cfg.task.name!='AllegroDigitTabletop':raise ValueError('Tabletop-only evaluation')
    bank=torch.load(ROOT/cfg.task.env.grasp.cache,map_location='cpu')
    obj=bank['objects'][0]['name'];n=min(a.envs,len(bank['states'][obj]))
    ids=torch.linspace(0,len(bank['states'][obj])-1,n).long()
    cfg.task.env.numEnvs=n;cfg.task.rl_device='cuda:0';cfg.task.sim.use_gpu_pipeline=False
    cfg.task.env.manipulation.object_contact_filter=True
    task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
    np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    env=AllegroDigitTabletop(task,'cuda:0',-1,True)
    try:
        path=run/'stage1_nn'/(a.checkpoint+'.pth');ck=torch.load(path,map_location='cuda:0')
        require_collision_signature(ck,env.collision_hash,'Teacher')
        require_action_reference(ck,env.cfg,'Teacher')
        if ck['asset_sha256']!=env.asset_hash:raise ValueError('Teacher asset mismatch')
        model=ActorCritic(network_spec(cfg,env.num_obs)).cuda().eval();model.load_state_dict(ck['model'])
        rms=RunningMeanStd((env.num_obs,)).cuda().eval();rms.load_state_dict(ck['running_mean_std'])
        env.stage='grasps' # No silent reset: only first episode contributes.
        install(env,bank['states'][obj][ids],bank['targets'][obj][ids])
        obs=env._observations();alive=torch.ones(n,dtype=torch.bool);failed=torch.zeros(n,dtype=torch.bool)
        angle=torch.zeros(n);half_angle=torch.zeros(n);turn_ever=torch.zeros(n,dtype=torch.bool)
        sums={};counts=0;late_sums={};late_counts=0;per_two=torch.zeros(n);per_steps=torch.zeros(n)
        for step in range(a.steps):
            active=alive.clone()
            with torch.no_grad():
                mu,*_=model._actor_critic({'obs':rms(obs['obs']),'priv_info':obs['priv_info']})
                obs,rew,_,_=env.step(torch.where(active.cuda()[:,None],mu,torch.zeros_like(mu)).clamp(-1,1))
            if not torch.isfinite(rew).all():raise FloatingPointError('Nonfinite evaluation reward')
            gel=env.task_gel_contacts
            fields={'object_gel_count':gel.sum(-1).float(),'two_object_gels':(gel.sum(-1)>=2).float(),
                'any_object_gel':gel.any(-1).float(),'target_increment_lost':env.target_increment_lost,
                'raw_action_clipped':(mu.cpu().abs()>1).float().mean(-1),
                'off_axis_squared':env.reward_diagnostics['off_axis_squared'],
                'reward':rew.cpu()}
            for key,value in fields.items():
                value=value.cpu();sums[key]=sums.get(key,0.)+float(value[active].sum())
                if step>=a.steps//2:late_sums[key]=late_sums.get(key,0.)+float(value[active].sum())
            counts+=int(active.sum())
            if step>=a.steps//2:late_counts+=int(active.sum())
            per_two[active]+=(gel.sum(-1)>=2)[active].float();per_steps+=active.float()
            angle[active]=env.total_angle[active,0]
            if step<a.steps//2:half_angle[active]=angle[active]
            turn_ever|=active&(angle>=2*math.pi)
            failed|=active&env.terminal['failed'];alive&=~env.terminal['done']
            if not alive.any():break
        # At 600 steps normal time-limit termination counts as retained.
        retained=~failed
        late_rotation=angle-half_angle
        per_contact=per_two/per_steps.clamp_min(1)
        r=dict(object=obj,task='AllegroDigitTabletop',teacher_only=True,n=n,seed=a.seed,
            checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            agent_steps=ck['agent_steps'],seconds=a.steps*env.control_dt,
            retained_fraction=float(retained.float().mean()),
            one_turn_retained_fraction=float((retained&(angle>=2*math.pi)).float().mean()),
            ever_one_turn_fraction=float(turn_ever.float().mean()),
            mean_turns=float((angle/(2*math.pi)).mean()),
            late_signed_speed_all_starts=float((late_rotation/(a.steps-a.steps//2)/env.control_dt).mean()),
            frame_means={k:v/max(counts,1) for k,v in sums.items()},
            last_half_frame_means={k:v/max(late_counts,1) for k,v in late_sums.items()},
            qualified_one_turn_fraction=float((retained&(angle>=2*math.pi)&(per_contact>=.8)&(late_rotation>0)).float().mean()),
            qualification='retained, >=1 signed turn, >=80% observed episode frames with >=2 object-gel contacts, positive signed rotation in last half; diagnostic selection rule, not training reward',
            episodes=dict(failed=failed.tolist(),turns=(angle/(2*math.pi)).tolist(),two_contact_fraction=per_contact.tolist(),late_radians=late_rotation.tolist()),
            note='Fixed cached starts; no held-out shape generalization claim. Failed episodes contribute zero further rotation to all-start late-speed metric.')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({k:r[k] for k in ['agent_steps','retained_fraction','one_turn_retained_fraction','qualified_one_turn_fraction']}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
