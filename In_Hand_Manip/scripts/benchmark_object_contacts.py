"""Short fixed-action replay benchmark; no learning or checkpoint modification.

Run separately for GPU/off, CPU/off and CPU/on on the same physical GPU. An
optional teacher records an action tape once; all timed runs replay that tape.
"""
import isaacgym  # before torch
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.padapt.tactile_distill import network_spec
from hora.utils.hand_collision import require_collision_signature
from hora.utils.manip_control import require_action_reference
from hora.utils.object_contacts import GelObjectContactReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline', choices=['cpu', 'gpu'], required=True)
    p.add_argument('--filter', action='store_true')
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--seed', type=int, default=215)
    p.add_argument('--tape', required=True)
    p.add_argument('--record-teacher')
    p.add_argument('--output', required=True)
    p.add_argument('--verify', action='store_true', help='Query reference pairs outside timed section')
    a = p.parse_args()
    output, tape = Path(a.output).resolve(), Path(a.tape).resolve()
    if output.exists() or (a.record_teacher and tape.exists()):
        raise FileExistsError('Use a fresh output and recording path')
    output.parent.mkdir(parents=True, exist_ok=True)
    if a.record_teacher:
        tape.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(1)
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT/'configs')):
        cfg = compose(config_name='manip_tabletop_box45', overrides=[
            'stage=teacher', 'num_envs='+str(a.num_envs), 'pipeline='+a.pipeline,
            'sim_device=cuda:0', 'rl_device=cuda:0', 'graphics_device_id=-1',
            'task.env.manipulation.object_contact_filter='+str(a.filter).lower()])
    task = OmegaConf.to_container(cfg.task, resolve=True); task['stage'] = 'teacher'
    env = AllegroDigitTabletop(task, 'cuda:0', -1, True)
    obs = env.reset()
    initial = torch.cat((env.q, env.targets, env.object_state, env.kp, env.kd), -1).cpu().numpy()
    # Avoid different resets confounding the cost comparison. Failures remain
    # observable but never reset physics; this is not a policy success evaluation.
    env.stage = 'grasps'
    total = a.warmup+a.steps
    model = rms = None
    if a.record_teacher:
        ck = torch.load(a.record_teacher, map_location='cuda:0')
        require_collision_signature(ck, env.collision_hash, 'Teacher')
        require_action_reference(ck, env.cfg, 'Teacher')
        if ck['asset_sha256'] != env.asset_hash:
            raise ValueError('Asset mismatch')
        model = ActorCritic(network_spec(cfg, env.num_obs)).cuda().eval()
        model.load_state_dict(ck['model'])
        rms = RunningMeanStd((env.num_obs,)).cuda().eval()
        rms.load_state_dict(ck['running_mean_std'])
        actions = torch.empty(total, a.num_envs, 16, device='cuda:0')
    else:
        data = np.load(tape)
        actions = torch.from_numpy(data['actions']).cuda()
        if actions.shape != (total, a.num_envs, 16):
            raise ValueError('Action tape shape mismatch')
    arrays = {k: [] for k in ['q', 'object', 'raw', 'task', 'reward_contact', 'reward_failure',
                              'failed', 'missing', 'obs', 'history_last', 'pair_load']}
    observer = GelObjectContactReader(env) if a.verify and a.pipeline == 'cpu' else None
    elapsed = 0.; pair_seconds = [0.]
    if env.object_contact_reader is not None:
        refresh = env.object_contact_reader.refresh
        def timed_refresh(e):
            start = time.perf_counter(); result = refresh(e)
            pair_seconds[0] += time.perf_counter()-start
            return result
        env.object_contact_reader.refresh = timed_refresh
    sample_times = []
    with torch.no_grad():
        for step in range(total):
            if model is not None:
                actions[step] = model.act_inference({'obs': rms(obs['obs']), 'priv_info': obs['priv_info']}).clamp(-1, 1)
            torch.cuda.synchronize()
            if step == a.warmup: pair_seconds[0] = 0.
            start = time.perf_counter()
            obs, _, _, _ = env.step(actions[step])
            torch.cuda.synchronize()
            duration = time.perf_counter()-start
            if step >= a.warmup:
                elapsed += duration; sample_times.append(duration)
            # Capture after timing, including optional reference query in off mode.
            raw = env.contact_forces[:, env.gel_ids].norm(dim=-1) > cfg.task.env.digit.contact_threshold
            if a.verify:
                load = (observer.refresh(env).clone() if observer is not None
                        else torch.full_like(raw, float('nan'), dtype=torch.float32))
                values = dict(q=env.q, object=env.object_state, raw=raw, task=env.task_gel_contacts,
                    reward_contact=env.reward_terms['contact'], reward_failure=env.reward_terms['failure_penalty'],
                    failed=env.terminal['failed'], missing=env.missing_steps,
                    obs=obs['obs'], history_last=obs['proprio_hist'][:, -1], pair_load=load)
                for key, value in values.items(): arrays[key].append(value.cpu().numpy().copy())
            if (step+1) % 100 == 0: print('step', step+1, flush=True)
    if a.record_teacher:
        np.savez_compressed(tape, actions=actions.cpu().numpy(), initial=initial)
    if a.verify:
        np.savez_compressed(output.with_suffix('.npz'), initial=initial,
                            **{k: np.stack(v) for k, v in arrays.items()})
    result = dict(pipeline=a.pipeline, object_contact_filter=a.filter, num_envs=a.num_envs,
        warmup_steps=a.warmup, measured_steps=a.steps, seed=a.seed, sim_device='cuda:0', rl_device='cuda:0',
        state_device=env.device, elapsed_seconds=elapsed, step_ms=1000*elapsed/a.steps,
        env_steps_per_second=a.num_envs*a.steps/elapsed,
        step_ms_p50=float(np.median(sample_times)*1000), step_ms_p95=float(np.percentile(sample_times,95)*1000),
        pair_query_ms_per_step=1000*pair_seconds[0]/a.steps, pid=os.getpid(),
        initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        actions_sha256=hashlib.sha256(actions.cpu().numpy().tobytes()).hexdigest(),
        source_config=OmegaConf.to_container(cfg, resolve=True),
        note='Fixed action tape, no auto-reset; per-step CUDA synchronized. No PPO updates timed. '
             'Optional verification queries/copies are outside timing. Normal loads, last physics step.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k,v in result.items() if k != 'source_config'}), flush=True)
    # Preview 4 destroy_sim hangs on this host in CPU-pipeline diagnostics.
    # Isolated benchmark process exits only after complete durable result output.
    os._exit(0)


if __name__ == '__main__':
    main()
