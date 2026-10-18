"""Read-only diagnostics of privileged teachers; never a deployment evaluation."""
import isaacgym
import argparse, hashlib, json, math, importlib.util
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.tasks.manipulation_math import rotation_increment, rotation_terms
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature

OUT=Path(__file__).resolve().parent

def evaluate(obj,which="last",stochastic=False,n=256,zero=False):
    run=ROOT.parent/'runs/manip'/('teacher_'+obj+'_z_self_collision_v1')
    cfg=OmegaConf.load(run/'config.yaml')
    ckpt_path=run/'stage1_nn'/(which+'.pth')
    for rel in ['hora/utils/hand_collision.py','hora/tasks/manipulation_math.py','hora/algo/models/models.py','hora/algo/models/running_mean_std.py']:
        assert (ROOT/rel).read_bytes()==(run/'source'/rel).read_bytes(), 'Source mismatch: '+rel
    cfg.task.env.numEnvs=n
    cfg.task.rl_device='cuda:0'
    seed=20261013
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
    # Load the exact historical task; resolve its preserved legacy asset/cache at the project root.
    spec=importlib.util.spec_from_file_location('hora.tasks.historical_diagnostic',run/'source/hora/tasks/allegro_digit_two_axis.py')
    historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical);historical.ROOT=ROOT
    env=historical.AllegroDigitTwoAxis(task,'cuda:0',-1,True)
    try:
        ck=torch.load(ckpt_path,map_location=env.device)
        assert ck['kind']=='privileged_teacher' and ck['asset_sha256']==env.asset_hash
        require_collision_signature(ck,env.collision_hash,'Teacher')
        model=ActorCritic(network_spec(cfg,env.num_obs)).to(env.device).eval()
        model.load_state_dict(ck['model'])
        rms=RunningMeanStd((env.num_obs,)).to(env.device).eval();rms.load_state_dict(ck['running_mean_std'])
        torch.manual_seed(seed+1);torch.cuda.manual_seed_all(seed+1)
        obs=env.reset()
        # Suppress automatic resets only; teacher reward/tolerance/physics stay identical.
        # Each environment contributes exactly its first episode, not repeated failures.
        env.stage='grasps'
        alive=torch.ones(n,dtype=torch.bool,device=env.device)
        reached=torch.zeros_like(alive);failed=torch.zeros_like(alive)
        duration=torch.zeros(n,device=env.device);angle=torch.zeros_like(duration);returns=torch.zeros_like(duration)
        net=torch.zeros_like(duration);positive=torch.zeros_like(duration);negative=torch.zeros_like(duration)
        sums={k:0. for k in ['rotation','off_axis','overspeed','milestone','gel','housing','position','linear_velocity','action_rate','torque','drop','gel_count','housing_count','gel_ge2','raw_action_clipped','target_at_limit','palm_contact','nontip_contact','mu_mean']}
        count=0;trace=[]
        initial_hash=hashlib.sha256(torch.cat([env.q, env.object_state], dim=-1).cpu().numpy().tobytes()).hexdigest()
        windows=[dict(start_s=a,end_s=b,frames=0,episodes=set(),sums={}) for a,b in [(0,5),(5,10),(10,20),(20,30)]]
        names=env.gym.get_asset_rigid_body_names(env.hand_asset)
        palm=names.index('base_link');nontip=[i for i in range(len(names)) if i not in env.gel_ids]
        for step in range(int(cfg.task.env.episodeLength)):
            prev=env.object_state[:,3:7].clone();goal=env.next_goal.clone();old_targets=env.targets.clone()
            with torch.no_grad():
                inputs={'obs':rms(obs['obs']),'priv_info':obs['priv_info']}
                mu,logstd,_,_,_=model._actor_critic(inputs)
                raw=mu+torch.exp(logstd)*torch.randn_like(mu) if stochastic else mu
                if zero: raw=torch.zeros_like(raw)
                raw=torch.where(alive[:,None],raw,torch.zeros_like(raw))
                obs,reward,done,info=env.step(raw.clamp(-1,1))
            assert torch.isfinite(reward[alive]).all() and torch.isfinite(env.object_state[alive]).all()
            increment=rotation_increment(env.object_state[:,3:7],prev)
            omega=increment/env.control_dt
            rot,off,fast,along=rotation_terms(omega,env.axes,env.cfg['target_speed'])
            gel=env.contact_forces[:,env.gel_ids].norm(dim=-1)>cfg.task.env.digit.contact_threshold
            housing=env.contact_forces[:,env.housing_ids].norm(dim=-1)>cfg.task.env.digit.contact_threshold
            hit=env.segment_angle>=goal-env.cfg['goal_tolerance']
            disp=(env.object_state[:,:3]-env.initial_position).norm(dim=-1)
            w=env.cfg
            terms=dict(rotation=w['rotation_weight']*rot,off_axis=-w['off_axis_weight']*off,
              overspeed=-w['overspeed_weight']*fast,milestone=w['goal_weight']*hit,
              gel=w['contact_weight']*(gel.sum(-1)>=2),housing=-w['housing_weight']*housing.float().mean(-1),
              position=-w['position_weight']*disp,linear_velocity=-w['linear_velocity_weight']*env.object_state[:,7:10].norm(dim=-1),
              action_rate=-w['action_rate_weight']*(env.actions-env.last_actions).square().sum(-1),
              torque=-w['torque_weight']*env.torques.square().sum(-1),drop=-w['drop_penalty']*env.terminal['failed'],
              gel_count=gel.float().sum(-1),housing_count=housing.float().sum(-1),gel_ge2=(gel.sum(-1)>=2).float(),
              raw_action_clipped=(raw.abs()>1).float().mean(-1),mu_mean=mu.mean(-1),
              target_at_limit=((env.targets-env.lower<.001)|(env.upper-env.targets<.001)).float().mean(-1),
              palm_contact=(env.contact_forces[:,palm].norm(dim=-1)>.1).float(),
              nontip_contact=(env.contact_forces[:,nontip].norm(dim=-1)>.1).any(-1).float())
            rebuilt=sum(terms[k] for k in ['rotation','off_axis','overspeed','milestone','gel','housing','position','linear_velocity','action_rate','torque','drop'])
            assert torch.allclose(rebuilt[alive],reward[alive],atol=1e-4,rtol=1e-4)
            a=alive.float();count+=int(alive.sum())
            window=next(wi for wi in windows if wi['start_s'] <= step*env.control_dt < wi['end_s'])
            window['frames']+=int(alive.sum())
            window['episodes'].update(alive.nonzero(as_tuple=False).flatten().tolist())
            diagnostics=dict(z_speed=omega[:,2],abs_z_speed=omega[:,2].abs(),angular_speed=omega.norm(dim=-1),
                mean_abs_joint_speed=env.qdot.abs().mean(-1), action_change=(env.actions-env.last_actions).abs().mean(-1),
                gel_count=gel.float().sum(-1),target_at_limit=terms['target_at_limit'])
            requested=env.cfg['action_scale']*env.actions
            actual=env.targets-old_targets
            diagnostics.update(mean_abs_action=env.actions.abs().mean(-1),
                positive_action=(env.actions>0.05).float().mean(-1),negative_action=(env.actions<-.05).float().mean(-1),
                requested_increment_l1=requested.abs().sum(-1),blocked_increment_l1=(requested-actual).abs().sum(-1),
                actual_increment_l1=actual.abs().sum(-1),mean_abs_target_error=(env.targets-env.q).abs().mean(-1),
                actual_joint_at_limit=((env.q-env.lower<.001)|(env.upper-env.q<.001)).float().mean(-1),
                last_substep_torque_at_limit=(env.torques.abs()>=env.effort-1e-5).float().mean(-1))
            diagnostics.update({'reward_'+k:terms[k] for k in ['rotation','off_axis','position','action_rate','torque','gel','housing']})
            for key,value in diagnostics.items():window['sums'][key]=window['sums'].get(key,0.)+float((value*a).sum())
            for k,v in terms.items():sums[k]+=float((v*a).sum())
            returns+=reward*a;positive+=increment[:,2].clamp_min(0)*a;negative+=(-increment[:,2]).clamp_min(0)*a
            reached|=alive & (env.total_angle[:,0]>=2*math.pi)
            angle[alive]=env.total_angle[alive,0]
            duration[alive]=(step+1)*env.control_dt
            failed|=alive & env.terminal['failed']
            alive &= ~env.terminal['done']
            if step%20==19:
                trace.append(dict(seconds=(step+1)*env.control_dt,remaining=int(alive.sum()),mean_net_turns=float(angle.mean()/(2*math.pi))))
            if not alive.any():break
        def avg(x):return float(x.float().mean())
        result=dict(object=obj,checkpoint=which,stochastic=stochastic,n=n,seed=seed,privileged_teacher=True,
          checkpoint_steps=ck['agent_steps'],checkpoint_best_reward=ck['best_rewards'],
          retention_30s=avg(~failed),one_turn_and_retained=avg((~failed)&(angle>=2*math.pi)),ever_reached_one_turn=avg(reached),
          mean_seconds=avg(duration),mean_net_turns=avg(angle/(2*math.pi)),median_net_turns=float((angle/(2*math.pi)).median()),
          mean_angular_speed=float(angle.sum()/duration.sum()),mean_return=avg(returns),
          mean_positive_turns=avg(positive/(2*math.pi)),mean_negative_turns=avg(negative/(2*math.pi)),
          frame_means={k:v/count for k,v in sums.items()},trace=trace,
          episodes=dict(seconds=duration.tolist(),net_turns=(angle/(2*math.pi)).tolist(),failed=failed.tolist(),reached_one_turn=reached.tolist(),returns=returns.tolist()))
        result['zero_action']=zero
        result['initial_state_sha256']=initial_hash
        result['windows']=[dict(start_s=wi['start_s'],end_s=wi['end_s'],frames=wi['frames'],episodes=len(wi['episodes']),
            means={k:v/max(1,wi['frames']) for k,v in wi['sums'].items()}) for wi in windows]
        result['historical_task_sha256']=hashlib.sha256((run/'source/hora/tasks/allegro_digit_two_axis.py').read_bytes()).hexdigest()
        result['asset_sha256']=env.asset_hash
        result['method']='Historical task snapshot, preserved legacy asset and checkpoint; torque sampled at last physics substep of each control step.'
        dest=OUT/(obj+'_actuation.json')
        dest.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','trace']}),flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--object',required=True);args=p.parse_args()
    evaluate(args.object)
