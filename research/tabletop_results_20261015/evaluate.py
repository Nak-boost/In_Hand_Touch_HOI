"""CPU PhysX diagnostics of saved tabletop teachers, with true contact identities."""
import isaacgym
from isaacgym import gymapi
import argparse, json, math, hashlib
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
from hora.utils.tabletop_contacts import object_contact_summary
from scripts.check_grasp_cache import install

OUT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=['mean','sampled','zero'],default='mean');a=p.parse_args()
    run=ROOT.parent/'runs/manip'/f'tabletop_box45_z_seed{a.seed}_20261015'
    for rel in ['hora/tasks/allegro_digit_tabletop.py','hora/tasks/allegro_digit_two_axis.py',
                'hora/tasks/tabletop_math.py','hora/tasks/manipulation_math.py','hora/algo/models/models.py',
                'hora/algo/models/running_mean_std.py','hora/tasks/base/vec_task.py']:
        assert (ROOT/rel).read_bytes()==(run/'source'/rel).read_bytes(),rel
    cfg=OmegaConf.load(run/'config.yaml');cfg.task.env.numEnvs=32
    cfg.task.rl_device='cpu';cfg.task.sim.use_gpu_pipeline=False;cfg.task.sim.physx.use_gpu=False
    cfg.task.sim.physx.num_threads=4;cfg.task.sim.physx.num_subscenes=1
    np.random.seed(215);torch.manual_seed(215);torch.set_num_threads(1)
    task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
    env=AllegroDigitTabletop(task,'cpu',-1,True)
    try:
        ckpath=run/'stage1_nn/last.pth';ck=torch.load(ckpath,map_location='cpu')
        require_collision_signature(ck,env.collision_hash,'Teacher')
        assert ck['asset_sha256']==env.asset_hash
        model=ActorCritic(network_spec(cfg,env.num_obs)).eval();model.load_state_dict(ck['model'])
        rms=RunningMeanStd((env.num_obs,)).eval();rms.load_state_dict(ck['running_mean_std'])
        bank=torch.load(env.cache_path,map_location='cpu');name=env.catalog[0]['name']
        states=bank['states'][name].repeat(2,1);targets=bank['targets'][name].repeat(2,1)
        env.stage='grasps';install(env,states,targets);obs=env._observations()
        n=env.num_envs;alive=torch.ones(n,dtype=torch.bool);failed=torch.zeros_like(alive)
        turns=torch.zeros(n);duration=torch.zeros(n);returns=torch.zeros(n);ever=torch.zeros_like(alive)
        lost=torch.zeros_like(alive);outside=torch.zeros_like(alive)
        sums={};count=0;trace=[];chunk_sums={};chunk_count=0;forbidden=0;contact_sources={}
        body_names=env.gym.get_asset_rigid_body_names(env.hand_asset)
        for step in range(600):
            with torch.no_grad():
                mu,logstd,*_=model._actor_critic({'obs':rms(obs['obs']),'priv_info':obs['priv_info']})
                raw=mu if a.mode=='mean' else mu+logstd.exp()*torch.randn_like(mu)
                if a.mode=='zero':raw=torch.zeros_like(raw)
                obs,rew,_,_=env.step(torch.where(alive[:,None],raw,torch.zeros_like(raw)).clamp(-1,1))
            assert torch.isfinite(rew).all() and torch.isfinite(env.object_state).all()
            pairs=object_contact_summary(env);forbidden+=len(pairs['forbidden'])
            rawgel=env.contact_forces[:,env.gel_ids].norm(dim=-1)>.1
            actual=pairs['gels'] & rawgel
            if step%20==0:
                for i in (alive & rawgel.any(-1) & ~actual.any(-1)).nonzero().flatten().tolist():
                    for c in env.gym.get_env_rigid_contacts(env.envs[i]):
                        if c['lambda']<=0:continue
                        x,y=int(c['body0']),int(c['body1'])
                        if x not in env.gel_ids and y not in env.gel_ids:continue
                        def label(b):return body_names[b] if 0<=b<len(body_names) else str(b)
                        key=' / '.join(sorted([label(x),label(y)]))
                        contact_sources[key]=contact_sources.get(key,0)+1
            dq=env.total_angle[:,0]/(2*math.pi)-turns
            terms=dict(speed=dq*2*math.pi/env.control_dt,
                raw_action_clipped=(raw.abs()>1).float().mean(-1),
                target_lost=env.target_increment_lost,
                target_at_limit=((env.targets-env.lower<.001)|(env.upper-env.targets<.001)).float().mean(-1),
                object_gel_count=actual.sum(-1).float(),raw_gel_count=rawgel.sum(-1).float(),
                any_object_gel=actual.any(-1).float(),two_object_gels=(actual.sum(-1)>=2).float(),
                false_contact_only=(rawgel.any(-1)&~actual.any(-1)).float(),
                non_gel_object=pairs['other_hand'].float(),table_support=pairs['table'].float(),
                reward=rew,**{'reward_'+k:v for k,v in env.reward_terms.items()})
            active=int(alive.sum());count+=active;chunk_count+=active
            for k,v in terms.items():
                val=float(v[alive].sum());sums[k]=sums.get(k,0)+val;chunk_sums[k]=chunk_sums.get(k,0)+val
            turns[alive]=env.total_angle[alive,0]/(2*math.pi);duration[alive]=(step+1)*env.control_dt
            returns[alive]+=rew[alive];ever|=alive&(turns>=1)
            failed|=alive&env.terminal['failed'];lost|=alive&env.contact_lost;outside|=alive&env.outside_workspace
            alive&=~env.terminal['done']
            if (step+1)%100==0 or not alive.any():
                trace.append(dict(seconds=(step+1)*env.control_dt,alive=int(alive.sum()),
                                  frame_means={k:v/max(chunk_count,1) for k,v in chunk_sums.items()}))
                chunk_sums={};chunk_count=0
            if not alive.any():break
        r=dict(seed=a.seed,mode=a.mode,n=n,backend='CPU PhysX; training used GPU PhysX',
               checkpoint_step=ck['agent_steps'],checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
               retained_30s=float((~failed).float().mean()),one_turn_retained=float(((~failed)&(turns>=1)).float().mean()),
               ever_one_turn=float(ever.float().mean()),mean_turns=float(turns.mean()),
               median_turns=float(turns.median()),mean_seconds=float(duration.mean()),mean_return=float(returns.mean()),
               missing_failures=int(lost.sum()),outside_failures=int(outside.sum()),
               frame_means={k:v/count for k,v in sums.items()},trace=trace,forbidden_contacts=forbidden,
               non_object_contact_sources_sampled_every_20_steps=contact_sources,
               episodes=dict(turns=turns.tolist(),seconds=duration.tolist(),failed=failed.tolist(),returns=returns.tolist()))
        (OUT/f'eval_{a.seed}_{a.mode}.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({k:v for k,v in r.items() if k not in ['episodes','trace','frame_means']}),flush=True)
    finally:env.close()

if __name__=='__main__':main()
