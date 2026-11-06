"""Run a checkpoint-free tabletop scanning controller and save auditable trajectories."""
import argparse
import hashlib
import json
import math
import signal
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/tabletop_scan.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--offset-mm', type=float, default=0., help='Random initial XY placement offset, evaluation only')
    parser.add_argument('--rotation-degrees', type=float)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not math.isfinite(args.offset_mm) or args.seconds <= 0 or args.num_envs < 1 or args.offset_mm < 0:
        parser.error('Require positive duration/env count and nonnegative placement offset')
    if args.rotation_degrees is not None and (not math.isfinite(args.rotation_degrees) or args.rotation_degrees == 0):
        parser.error('Rotation increment must be finite and nonzero')
    import isaacgym
    from isaacgym import gymapi, gymtorch, gymutil
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hora.tasks.allegro_digit_scan import AllegroDigitScan
    from hora.controllers.tabletop_scan import TabletopScanController
    from hora.utils.scan_kinematics import ScanKinematics, cartesian_joint_target
    from hora.utils.mount_geometry import rotation
    from hora.utils.manip_control import tracking_action

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    spec = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(args.config.parent/spec.base_config), {'task': spec.task}) if 'base_config' in spec else spec
    task = OmegaConf.to_container(cfg.task, resolve=True)
    task['stage'] = 'grasps'  # no cache required and NO automatic resets within the trajectory
    task['rl_device'] = 'cpu'
    task['env']['numEnvs'] = args.num_envs
    task['sim']['use_gpu_pipeline'] = False
    task['enableCameraSensors'] = False
    task.pop('record_scan_camera', None)
    scan = OmegaConf.to_container(spec.scan, resolve=True)
    if args.rotation_degrees is not None:
        scan['rotation_degrees'] = args.rotation_degrees
    output = (args.output or ROOT.parent/'runs/scan'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(dict(task=task, scan=scan)), output/'config.yaml')
    env = AllegroDigitScan(task, args.device, -1 if args.headless else args.graphics_device_id, args.headless)
    records = {k: [] for k in ('q', 'qdot', 'targets', 'action', 'phase', 'desired_gel_positions',
                              'gel_positions', 'gel_normals', 'desired_gel_normals', 'gel_load', 'object_gel_load', 'object_state',
                              'axis_angle', 'table_contact', 'other_hand_contact', 'post_object_state', 'post_axis_angle')}
    contact_points = []
    stopped = ''
    fk_error = 0.
    interrupted = [False]
    previous_sigint = signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    try:
        env.reset()
        transform = np.eye(4)
        transform[:3, :3] = rotation([0, 1, 0], np.pi/2) @ rotation([1, 0, 0], np.pi/2)
        transform[:3, 3] = env.table_cfg['hand_position']
        kin = ScanKinematics(ROOT/task['env']['asset']['handAsset'], transform)
        q0 = np.array(task['env']['grasp']['canonical_q'])
        pos, normals, _ = kin.evaluate(q0)
        center = np.array(scan['center'])
        # Select surrounding rays from the projected sensing normals, rather
        # than arbitrary canonical tip locations that may face past the object.
        radial = -normals.copy(); radial[:, 2] = 0
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        diameter = task['env']['objects'][0]['size'][0]
        desired_open = center + radial*(diameter/2+scan['opening_clearance'])
        desired_open[:, 2] = pos[:, 2]
        q_open = kin.fit(q0, desired_open, -radial, scan['normal_weight'])
        open_pos, actual_open_normals, _ = kin.evaluate(q_open)
        open_normals = -radial  # retain the intended horizontal normal, not IK residual tilt
        open_normal_error = np.rad2deg(np.arccos(np.clip(np.sum(actual_open_normals*open_normals, axis=1), -1, 1)))
        if open_normal_error.max() > scan.get('max_preload_normal_error_degrees', 15.):
            raise ValueError('Open-pose sensing normal exceeds the allowed error; revise reachable setup')
        if np.linalg.norm(open_pos-desired_open, axis=1).max() > .006:
            raise ValueError('Open-pose IK error exceeds 6 mm; revise object placement/hand reach before running')
        setup = dict(q_open=q_open.tolist(), desired_open=desired_open.tolist(), open_positions=open_pos.tolist(),
                     open_normals=open_normals.tolist(), actual_open_normals=actual_open_normals.tolist(),
                     normal_error_degrees=open_normal_error.tolist(), position_error_m=np.linalg.norm(open_pos-desired_open, axis=1).tolist(),
                     asset_sha256=env.asset_hash, collision_sha256=env.collision_hash,
                     collision_spec=env.collision_spec,
                     controller_inputs='URDF kinematics, joint state, unfiltered gel net-load magnitude; fixed placement centre',
                     privileged_evaluation_only='object state, object-specific contacts, object-frame contact samples',
                     tactile_rendering=False)
        setup['schema_version'] = 2
        setup['source_sha256'] = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
            'scripts/run_tabletop_scan.py', 'hora/controllers/tabletop_scan.py', 'hora/utils/scan_kinematics.py',
            'hora/utils/primitive_assets.py', 'hora/utils/manip_control.py', 'hora/tasks/allegro_digit_scan.py',
            'hora/tasks/allegro_digit_tabletop.py', 'hora/tasks/allegro_digit_two_axis.py', 'hora/tasks/base/vec_task.py')}
        (output/'setup.json').write_text(json.dumps(setup, indent=2)+'\n')
        order = env.canonical_order
        inverse = np.argsort(order)
        env.q.copy_(torch.tensor(q_open[order], dtype=torch.float32).repeat(args.num_envs, 1))
        env.qdot.zero_(); env.targets.copy_(env.q); env.initial_q.copy_(env.q)
        state = torch.zeros(args.num_envs, 13)
        state[:, :3] = torch.tensor(task['env']['grasp']['object_position'])
        state[:, :2] += torch.tensor(np.random.uniform(-args.offset_mm/1000, args.offset_mm/1000, (args.num_envs, 2)), dtype=torch.float32)
        state[:, 6] = 1
        env.root_states[env.object_indices] = state
        env.initial_position.copy_(state[:, :3]); env.prev_pose.copy_(state[:, :7])
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_states))
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env._refresh()
        controllers = [TabletopScanController(scan, open_pos, open_normals, center, env.control_dt, actual_open_normals) for _ in range(args.num_envs)]
        axes = gymutil.AxesGeometry(.04)
        steps = math.ceil(args.seconds/env.control_dt)
        for step in range(steps):
            if interrupted[0] and records['q']:
                stopped = 'user_interrupt'
                break
            load = env.contact_forces[:, env.gel_ids].norm(dim=-1).numpy().copy()
            targets, desired, poses, sensed_normals, target_normals = [], [], [], [], []
            for i, control in enumerate(controllers):
                q = env.q[i].numpy()[inverse]
                p, n, j = kin.evaluate(q)
                # Kinematics stay in each environment's local frame, matching configured placement.
                d, dn = control.update(p, load[i], n)
                weights = np.broadcast_to(scan.get('position_weights', [1., 1., 1.]), (4, 3)).copy()
                # Do not use vertical compliance to escape above the side surface
                # or approach the table. Restore full height correction at the band.
                outside_height = (p[:, 2] < scan.get('contact_height_min', -np.inf)) | (p[:, 2] > scan.get('contact_height_max', np.inf))
                weights[outside_height, 2] = 1.
                if control.phase in ('settle', 'reset'):
                    weights[:] = 1.
                target = cartesian_joint_target(q, p, n, j, d, dn, kin.lower, kin.upper,
                    scan['max_joint_step'], scan['normal_weight'], scan['ik_damping'], weights)
                targets.append(target[order]); desired.append(d); poses.append(p)
                sensed_normals.append(n); target_normals.append(dn)
            if any(c.phase == 'fault' for c in controllers):
                stopped = 'controller_fault'
                break
            action = tracking_action(env.q, env.targets, torch.tensor(np.array(targets), dtype=torch.float32), env.actions, env.cfg)
            phase = [TabletopScanController.PHASES.index(c.phase) for c in controllers]
            # Pre-action state, sensor readings and exact bounded action are aligned for future BC.
            for key, value in dict(q=env.q, qdot=env.qdot, targets=env.targets, action=action,
                phase=np.array(phase), desired_gel_positions=np.array(desired), gel_positions=np.array(poses),
                gel_normals=np.array(sensed_normals), desired_gel_normals=np.array(target_normals),
                gel_load=load, object_gel_load=env.object_gel_normal_force if step else torch.zeros(args.num_envs, 4),
                object_state=env.object_state, axis_angle=env.total_angle[:, 0]).items():
                records[key].append(value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value.copy())
            env.step(action)
            records['post_object_state'].append(env.object_state.numpy().copy())
            records['post_axis_angle'].append(env.total_angle[:, 0].numpy().copy())
            for i in range(args.num_envs):
                predicted, _, _ = kin.evaluate(env.q[i].numpy()[inverse])
                measured = env.bodies[i, env.gel_ids, :3].numpy()
                fk_error = max(fk_error, float(np.linalg.norm(predicted-measured, axis=1).max()))
            if fk_error > 0.001:
                raise RuntimeError('URDF/simulator gel coordinates disagree: %.6fm' % fk_error)
            supported, other = [], []
            for i, e in enumerate(env.envs):
                object_body = env.object_contact_reader.object_bodies[i]
                table_body = env.gym.get_actor_rigid_body_index(e, env.table_handles[i], 0, gymapi.DOMAIN_ENV)
                support, non_gel = False, False
                for contact in env.gym.get_env_rigid_contacts(e):
                    a, b = int(contact['body0']), int(contact['body1'])
                    if contact['lambda'] <= 0 or object_body not in (a, b):
                        continue
                    hand_body = b if a == object_body else a
                    if hand_body == table_body:
                        support = True
                    elif hand_body in env.gel_ids:
                        point = contact['localPos0'] if a == object_body else contact['localPos1']
                        contact_points.append([step+1, i, env.gel_ids.index(hand_body), point['x'], point['y'], point['z'], contact['lambda']])
                    elif 0 <= hand_body < env.gym.get_asset_rigid_body_count(env.hand_asset):
                        non_gel = True
                supported.append(support); other.append(non_gel)
            records['table_contact'].append(np.array(supported)); records['other_hand_contact'].append(np.array(other))
            if env.viewer:
                env.gym.clear_lines(env.viewer)
                for i, state_row in enumerate(env.object_state.tolist()):
                    pose = gymapi.Transform(gymapi.Vec3(*state_row[:3]), gymapi.Quat(*state_row[3:7]))
                    gymutil.draw_lines(axes, env.gym, env.viewer, env.envs[i], pose)
            if (step+1) % 100 == 0:
                print(json.dumps(dict(seconds=(step+1)*env.control_dt, phase=[c.phase for c in controllers],
                    cycles=[c.cycles for c in controllers], degrees=(env.total_angle[:, 0]*180/math.pi).tolist(),
                    gel_load=load.round(3).tolist())), flush=True)
            if env.terminal['failed'].any():
                stopped = 'workspace_failure'
                break
        arrays = {k: np.array(v) for k, v in records.items()}
        np.savez_compressed(output/'trajectory.npz', **arrays, contact_samples=np.array(contact_points).reshape(-1, 7),
                            time=np.arange(len(records['q']))*env.control_dt,
                            joint_names=np.array(env.joint_names), phase_names=np.array(TabletopScanController.PHASES))
        reports = []
        for i, c in enumerate(controllers):
            obj = arrays['object_state'][:, i]
            angles = arrays['axis_angle'][:, i]
            phase = arrays['phase'][:, i]
            post_angles = arrays['post_axis_angle'][:, i]
            rotation_mask = phase == TabletopScanController.PHASES.index('rotate')
            return_mask = np.isin(phase, [TabletopScanController.PHASES.index('release'), TabletopScanController.PHASES.index('reset')])
            two_contact = (arrays['object_gel_load'][:, i] > scan['contact_on']).sum(-1) >= 2
            four_contact = (arrays['object_gel_load'][:, i] > scan['contact_on']).sum(-1) == 4
            sensor_tilt = np.rad2deg(np.arcsin(np.clip(np.abs(arrays['gel_normals'][:, i, :, 2]), 0, 1)))
            tilt = np.arccos(np.clip(1-2*(obj[:, 3]**2+obj[:, 4]**2), -1, 1))
            pushes = [e for e in c.events if e['target'] == 'release' and e['reason'] == 'push_complete']
            cycle_metrics = []
            for e in c.events:
                if e['source'] != 'reset' or e['target'] != 'approach':
                    continue
                starts = [s for s in c.events if s['target'] == 'approach' and s['time'] < e['time']]
                if not starts:
                    continue
                start_step = max(0, round(starts[-1]['time']/env.control_dt)-1)
                end_step = min(len(angles)-1, round(e['time']/env.control_dt)-1)
                cycle_metrics.append(dict(cycle=e['cycle'], start_seconds=starts[-1]['time'],
                    end_seconds=e['time'], net_degrees=float((angles[end_step]-angles[start_step])*180/math.pi)))
            points = np.array([p[3:6] for p in contact_points if p[1] == i]).reshape(-1, 3)
            bins = np.unique(np.floor((np.arctan2(points[:, 1], points[:, 0])+np.pi)/(2*np.pi)*36).astype(int)) if len(points) else []
            reports.append(dict(cycles=c.cycles, completed_pushes=len(pushes), events=c.events, cycle_metrics=cycle_metrics,
                net_rotation_degrees=float(env.total_angle[i, 0]*180/math.pi),
                last_half_rotation_degrees=float((angles[-1]-angles[len(angles)//2])*180/math.pi),
                max_xy_drift_m=float(np.linalg.norm(obj[:, :2]-obj[0, :2], axis=1).max()),
                max_object_tilt_degrees=float(np.rad2deg(tilt).max()),
                rotate_two_object_gels_fraction=float(two_contact[rotation_mask].mean()) if rotation_mask.any() else None,
                rotate_four_object_gels_fraction=float(four_contact[rotation_mask].mean()) if rotation_mask.any() else None,
                rotate_gel_load_median_N=np.median(arrays['object_gel_load'][rotation_mask, i], axis=0).tolist() if rotation_mask.any() else None,
                rotate_sensor_tilt_p95_degrees=np.quantile(sensor_tilt[rotation_mask], .95, axis=0).tolist() if rotation_mask.any() else None,
                release_reset_net_degrees=float((post_angles-angles)[return_mask].sum()*180/math.pi),
                table_support_fraction=float(arrays['table_contact'][:, i].mean()),
                non_gel_object_fraction=float(arrays['other_hand_contact'][:, i].mean()),
                two_object_gels_fraction=float(two_contact.mean()),
                max_gel_force_N=float(arrays['gel_load'][:, i].max()),
                touched_azimuth_bins_of_36=len(bins),
                azimuth_metric='simulator contact locations; not observed shape reconstruction or full surface coverage',
                final_phase=c.phase, reason=c.reason))
        report = dict(seed=args.seed, control_dt=env.control_dt, recorded_steps=len(records['q']), max_kinematics_error_m=fk_error,
                      stop_reason=stopped or 'duration_complete', environments=reports)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Saved:', output, flush=True)
        print(json.dumps(dict(stop_reason=report['stop_reason'], environments=[{k: v for k, v in r.items()
            if k not in ('events', 'cycle_metrics')} for r in reports]), indent=2), flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        env.close()


if __name__ == '__main__':
    main()
