"""Read-only first-episode evaluation using each run's exact saved Python source."""
import argparse
import sys
import json
import math
import hashlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--object', default='sphere55')
p.add_argument('--seed', type=int, default=215)
p.add_argument('--mode', choices=['mean', 'sampled', 'zero'], default='mean')
p.add_argument('--checkpoint', default='last')
p.add_argument('--pairs', action='store_true')
p.add_argument('--detail', action='store_true')
a = p.parse_args()
ROOT = Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
OUT = Path(__file__).resolve().parent
import fcntl
label=f'{a.object}_{a.mode}_{a.checkpoint}_seed{a.seed}'
if a.detail: label += '_detail'
lock=(OUT/f'eval_{label}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
if (OUT/f'eval_{label}.json').exists():
    print('Existing completed result; skipped', label, flush=True)
    sys.exit(0)
run = ROOT.parent/'runs/manip'/'teacher_sphere55_z_hora16384_pairfilter_20261016'
sys.path.insert(0, str(run/'source'))
import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
from omegaconf import OmegaConf
import hora.tasks.allegro_digit_two_axis as task_module
task_module.ROOT = ROOT  # Assets/caches stay at their original verified paths.
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import require_action_reference
from scripts.check_grasp_cache import install, contact_pairs

cfg = OmegaConf.load(run/'config.yaml')
bank = torch.load(ROOT/cfg.task.env.grasp.cache, map_location='cpu')
name = cfg.task.env.objects[0].name
states, targets = bank['states'][name], bank['targets'][name]
cfg.task.env.numEnvs = len(states)
dev = 'cpu'
cfg.task.rl_device = dev
cfg.task.sim.use_gpu_pipeline = False
cfg.task.sim.physx.use_gpu = True
np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
torch.set_num_threads(1)
task = OmegaConf.to_container(cfg.task, resolve=True); task['stage'] = 'teacher'
env = task_module.AllegroDigitTwoAxis(task, 'cuda:0', -1, True)
try:
    ckpath = run/'stage1_nn'/f'{a.checkpoint}.pth'
    ck = torch.load(ckpath, map_location=dev)
    require_collision_signature(ck, env.collision_hash, 'Teacher')
    require_action_reference(ck, env.cfg, 'Teacher')
    assert ck['asset_sha256'] == env.asset_hash
    model = ActorCritic(network_spec(cfg, env.num_obs)).to(dev).eval()
    model.load_state_dict(ck['model'])
    rms = RunningMeanStd((env.num_obs,)).to(dev).eval()
    rms.load_state_dict(ck['running_mean_std'])
    env.stage = 'grasps'
    install(env, states.to(dev), targets.to(dev))
    obs = env._observations()
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    failed = torch.zeros_like(alive); ever = torch.zeros_like(alive)
    turns = torch.zeros(n, device=dev); duration = torch.zeros_like(turns)
    returns = torch.zeros_like(turns)
    late_angle = torch.zeros_like(turns); late_count = torch.zeros_like(turns)
    sums = {}; count = 0; trace = []; chunk_sums = {}; chunk_count = 0
    body_names = env.gym.get_asset_rigid_body_names(env.hand_asset)
    sources = {}; object_sources = {}
    for step in range(600):
        with torch.no_grad():
            mu, logstd, *_ = model._actor_critic({'obs': rms(obs['obs']), 'priv_info': obs['priv_info']})
            raw = mu if a.mode == 'mean' else mu+logstd.exp()*torch.randn_like(mu)
            if a.mode == 'zero': raw = torch.zeros_like(raw)
            obs, rew, _, _ = env.step(torch.where(alive[:,None], raw, torch.zeros_like(raw)).clamp(-1,1))
        assert torch.isfinite(rew).all() and torch.isfinite(env.object_state).all()
        rawgel = env.contact_forces[:,env.gel_ids].norm(dim=-1) > .1
        dq = env.total_angle[:,0]/(2*math.pi)-turns
        if step >= 500:
            late_angle[alive] += dq[alive]*2*math.pi
            late_count[alive] += 1
        terms = dict(speed=dq*2*math.pi/env.control_dt,
            raw_action_clipped=(raw.abs()>1).float().mean(-1),
            target_lost=env.target_increment_lost,
            target_at_limit=((env.targets-env.lower<.001)|(env.upper-env.targets<.001)).float().mean(-1),
            target_tracking_error=(env.targets-env.q).abs().mean(-1),
            raw_gel_count=rawgel.sum(-1).float(),
            two_raw_gels=(rawgel.sum(-1)>=2).float(),
            off_axis_squared=env.reward_diagnostics['off_axis_squared'],
            object_gel_count=env.task_gel_contacts.sum(-1).float(),
            two_object_gels=(env.task_gel_contacts.sum(-1)>=2).float(),
            no_object_gel=(~env.task_gel_contacts.any(-1)).float(),
            displacement=(env.object_state[:,:3]-env.initial_position).norm(dim=-1),
            reward=rew, **{'reward_'+k:v for k,v in env.reward_terms.items()})
        if step % 10 == 0:
            gels, non = contact_pairs(env, alive)
            sources['sampled_frames'] = sources.get('sampled_frames',0)+int(alive.sum())
            sources['non_gel_object_frames'] = sources.get('non_gel_object_frames',0)+int(non[alive].sum())
            for i in alive.nonzero().flatten().tolist():
                ob=env.gym.get_actor_rigid_body_index(env.envs[i],1,0,gymapi.DOMAIN_ENV)
                touched=set()
                for c in env.gym.get_env_rigid_contacts(env.envs[i]):
                    x,y=int(c['body0']),int(c['body1'])
                    other=y if x==ob else x if y==ob else -1
                    if c['lambda']<=0 or not 0<=other<len(body_names): continue
                    touched.add(body_names[other])
                for key in touched:object_sources[key]=object_sources.get(key,0)+1
        active = int(alive.sum()); count += active; chunk_count += active
        for k,v in terms.items():
            val = float(v[alive].sum()); sums[k] = sums.get(k,0)+val; chunk_sums[k] = chunk_sums.get(k,0)+val
        turns[alive] = env.total_angle[alive,0]/(2*math.pi)
        duration[alive] = (step+1)*env.control_dt
        returns[alive] += rew[alive]; ever |= alive & (turns>=1)
        failed |= alive & env.terminal['failed']
        alive &= ~env.terminal['done']
        if (step+1)%100 == 0 or not alive.any():
            print('progress', step+1, 'alive', int(alive.sum()), flush=True)
            trace.append(dict(seconds=(step+1)*env.control_dt,alive=int(alive.sum()),
                mean_turns=float(turns.mean()),per_environment_turns=turns.tolist(),frame_means={k:v/max(chunk_count,1) for k,v in chunk_sums.items()}))
            chunk_sums={};chunk_count=0
        if not alive.any(): break
    r = dict(object=a.object, mode=a.mode, n=n, seed=a.seed,
        backend='GPU PhysX / CPU tensor pipeline; saved contact filter enabled; inference on CPU for small-batch evaluation',
        source=str(run/'source'), checkpoint=a.checkpoint, checkpoint_step=ck['agent_steps'],
        checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
        retained_30s=float((~failed).float().mean()),
        one_turn_retained=float(((~failed)&(turns>=1)).float().mean()),
        ever_one_turn=float(ever.float().mean()),mean_turns=float(turns.mean()),
        median_turns=float(turns.median()),mean_seconds=float(duration.mean()),mean_return=float(returns.mean()),
        last5s_speed=(late_angle/(late_count*env.control_dt).clamp_min(1e-6)).tolist(),
        last5s_steps=late_count.tolist(),
        frame_means={k:v/count for k,v in sums.items()},trace=trace,
        contact_sources_sampled_every_10_steps=sources,
        object_contact_frame_counts_by_body_every_10_steps=object_sources,
        episodes=dict(turns=turns.tolist(),seconds=duration.tolist(),failed=failed.tolist(),returns=returns.tolist()))
    dest=OUT/f'eval_{label}.json'
    dest.write_text(json.dumps(r,indent=2)+'\n')
    print('RESULT_SAVED',dest,flush=True)
finally:
    env.close()
