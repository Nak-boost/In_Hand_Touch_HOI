"""Evaluate cached tabletop starts using actual-position actions and true body pairs."""
import isaacgym
import argparse
import hashlib
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.manip_control import tracking_action
from hora.utils.tabletop_contacts import object_contact_summary
from hora.utils.misc import set_seed
from scripts.check_grasp_cache import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='manip_tabletop_box45')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--output-cache')
    parser.add_argument('--hold-steps', type=int, default=200)
    parser.add_argument('--zero-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=215)
    args = parser.parse_args()
    if min(args.hold_steps, args.zero_steps) < 1:
        parser.error('Step counts must be positive')
    source = (ROOT/args.cache).resolve()
    output = Path(args.output).resolve()
    dest = (ROOT/args.output_cache).resolve() if args.output_cache else None
    if output.exists() or (dest and dest.exists()):
        raise FileExistsError('Use new output/cache paths')
    bank = torch.load(source, map_location='cpu')
    name = bank['objects'][0]['name']
    states, targets = bank['states'][name], bank['targets'][name]
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT/'configs')):
        cfg = compose(config_name=args.config, overrides=[
            'stage=teacher', 'pipeline=cpu', 'rl_device=cpu', 'graphics_device_id=-1',
            'num_envs='+str(len(states)), 'task.env.grasp.cache='+str(source)])
    task = OmegaConf.to_container(cfg.task, resolve=True)
    # Fully resolved experiment configs deliberately do not inherit mutable
    # shared defaults/interpolations. Set probe devices/count explicitly.
    task['env']['numEnvs'] = len(states)
    task['rl_device'] = 'cpu'
    task['env']['grasp']['cache'] = str(source)
    task['stage'] = 'teacher'
    set_seed(args.seed)
    env = AllegroDigitTabletop(task, 'cuda:0', -1, True)
    env.stage = 'grasps'  # disable auto-reset; retain every failure for diagnostics

    def rollout(mode, steps, joint=None, finger=None):
        install(env, states, targets)
        n = env.num_envs
        failed = torch.zeros(n, dtype=torch.bool)
        support, two, other = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        force, drift = torch.zeros(n), torch.zeros(n)
        actual_open, base_q = torch.zeros(n), torch.zeros(n)
        initial_contact = torch.zeros(n, dtype=torch.bool)
        regained = torch.zeros(n)
        forbidden, saturated, count = 0, 0., 0
        for step in range(steps):
            desired = targets.clone()
            if joint is not None:
                amount = 0 if step < 20 else min((step-19)/10, 1) if step < 40 else max(1-(step-39)/10, 0)
                desired[:, joint] = (desired[:, joint]-.06*amount).clamp(env.lower[joint], env.upper[joint])
            action = torch.zeros_like(env.actions) if mode == 'zero' else tracking_action(
                env.q, env.targets, desired, env.actions, env.cfg)
            saturated += (action.abs() >= 1-1e-6).float().sum().item()
            count += action.numel()
            env.step(action)
            pairs = object_contact_summary(env)
            forces = env.contact_forces[:, env.gel_ids].norm(dim=-1)
            object_force = env.object_gel_normal_force if env.object_contact_reader else forces
            effective = pairs['gels'] & (object_force > cfg.task.env.digit.contact_threshold)
            failed |= env.terminal['failed']
            support += pairs['table'].float()
            two += (effective.sum(-1) >= 2).float()
            other += pairs['other_hand'].float()
            force = torch.maximum(force, forces.max(-1).values)
            drift = torch.maximum(drift, (env.object_state[:, :2]-states[:, 16:18]).norm(dim=-1))
            forbidden += len(pairs['forbidden'])
            if joint is not None:
                if step == 19:
                    initial_contact = effective[:, finger].clone()
                    base_q = env.q[:, joint].clone()
                if 20 <= step < 40:
                    actual_open = torch.maximum(actual_open, base_q-env.q[:, joint])
                if step >= 70:
                    regained += (effective.sum(-1) >= 2).float()
        passed = (~failed & (support/steps >= .9) & (two/steps >= .9) &
                  (other/steps <= .1) & (force <= 2.0))
        if forbidden:
            passed[:] = False
        result = dict(mode=mode, seconds=steps*env.control_dt, failed=failed.tolist(),
            table_fraction=(support/steps).tolist(), two_object_gels_fraction=(two/steps).tolist(),
            non_gel_fraction=(other/steps).tolist(), max_force_N=force.tolist(),
            max_xy_drift_m=drift.tolist(), forbidden_contacts=forbidden,
            action_saturated_fraction=saturated/count, passed=passed.tolist())
        if joint is not None:
            result.update(initial_contact=initial_contact.tolist(), actual_open_rad=actual_open.tolist(),
                          recovered_two_contacts=(regained >= 9).tolist(),
                          retraction_success=(~failed & initial_contact & (actual_open >= .015) & (regained >= 9)).tolist())
        print(json.dumps(dict(mode=mode, passed=int(passed.sum()), retained=int((~failed).sum()))), flush=True)
        return result, passed

    try:
        zero, _ = rollout('zero', args.zero_steps)
        hold, selected = rollout('track_cached_targets', args.hold_steps)
        retract = []
        for finger, number in enumerate([1, 5, 9, 13]):
            result, _ = rollout('retract_'+str(number), 80, env.canonical_order.index(number), finger)
            retract.append(result)
        report = dict(input_cache=str(source), input_cache_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            action_reference=env.action_reference, seed=args.seed, count=len(states), accepted=int(selected.sum()),
            passed=bool(selected.any()), accepted_indices=selected.nonzero().flatten().tolist(),
            asset_sha256=env.asset_hash, collision_sha256=env.collision_hash,
            hand_shape_count=env.gym.get_asset_rigid_shape_count(env.hand_asset),
            criteria='Bounded tracking: no failure; >=90% table support and >=2 true object-gel contacts; <=10% non-gel support; max gel force <=2 N; no filtered pairs. Zero action and 0.06 rad single-finger withdrawal are separate diagnostics, not acceptance gates.',
            zero=zero, hold=hold, retractions=retract,
            joint_margin_rad=torch.minimum(states[:, :16]-env.lower, env.upper-states[:, :16]).min(-1).values.tolist())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n')
        if dest and selected.any():
            bank['states'] = {name: states[selected].clone()}
            bank['targets'] = {name: targets[selected].clone()}
            bank['quality_validation'] = dict(report=str(output), report_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                              action_reference=env.action_reference, controller='bounded_target_tracking_v1')
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bank, dest)
        print(json.dumps({k: report[k] for k in ['count', 'accepted', 'hand_shape_count', 'action_reference']}), flush=True)
        if dest and not selected.any():
            raise RuntimeError('No accepted starts; no output cache saved')
    finally:
        env.close()


if __name__ == '__main__':
    main()
