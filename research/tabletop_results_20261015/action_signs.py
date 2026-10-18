"""Short CPU replay: distinguish signed outputs from persistent outward commands."""
from evaluate import *
import os

def probe(seed):
    run=ROOT.parent/'runs/manip'/f'tabletop_box45_z_seed{seed}_20261015'
    cfg=OmegaConf.load(run/'config.yaml');cfg.task.env.numEnvs=16
    cfg.task.rl_device='cpu';cfg.task.sim.use_gpu_pipeline=False;cfg.task.sim.physx.use_gpu=False
    cfg.task.sim.physx.num_threads=4;cfg.task.sim.physx.num_subscenes=1
    np.random.seed(215);torch.manual_seed(215);torch.set_num_threads(1)
    task=OmegaConf.to_container(cfg.task,resolve=True);task['stage']='teacher'
    env=AllegroDigitTabletop(task,'cpu',-1,True)
    ck=torch.load(run/'stage1_nn/last.pth',map_location='cpu')
    require_collision_signature(ck,env.collision_hash,'Teacher')
    model=ActorCritic(network_spec(cfg,env.num_obs)).eval();model.load_state_dict(ck['model'])
    rms=RunningMeanStd((env.num_obs,)).eval();rms.load_state_dict(ck['running_mean_std'])
    bank=torch.load(env.cache_path,map_location='cpu');name=env.catalog[0]['name']
    env.stage='grasps';install(env,bank['states'][name],bank['targets'][name]);obs=env._observations()
    alive=torch.ones(16,dtype=torch.bool);count=0;sums={};initial=None
    for step in range(200):
        with torch.no_grad():
            mu,logstd,*_=model._actor_critic({'obs':rms(obs['obs']),'priv_info':obs['priv_info']})
            if initial is None:initial=mu.mean(0).tolist()
            obs,_,_,_=env.step(torch.where(alive[:,None],mu,torch.zeros_like(mu)).clamp(-1,1))
        values={'mu':mu,'positive':(mu>0).float(),'negative':(mu<0).float(),
                'positive_clipped':(mu>1).float(),'negative_clipped':(mu< -1).float(),
                'upper_limit':(env.upper-env.targets<.001).float(),
                'lower_limit':(env.targets-env.lower<.001).float(),
                'smoothed_positive':(env.actions>0).float(),'smoothed_negative':(env.actions<0).float(),
                'negative_sample_probability':.5*(1+torch.erf(-mu/(logstd.exp()*2**.5)))}
        count+=int(alive.sum())
        for k,v in values.items():sums[k]=sums.get(k,torch.zeros(16))+v[alive].sum(0)
        alive&=~env.terminal['done']
        if not alive.any():break
    r={'seed':seed,'backend':'CPU PhysX','horizon_seconds':10,'first_episode_only':True,
       'frame_count':count,'joint_names':env.joint_names,'initial_mean_mu':initial,
       'sigma':model.sigma.exp().tolist(),
       'per_joint':{k:(v/count).tolist() for k,v in sums.items()},
       'overall':{k:float(v.mean()/count) for k,v in sums.items()}}
    (OUT/f'action_signs_{seed}.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['overall']),flush=True)
    # CPU Gym destroy_sim hung in the prior diagnostics. This isolated process
    # releases all simulator resources on exit, after durable result output.
    os._exit(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    probe(p.parse_args().seed)
