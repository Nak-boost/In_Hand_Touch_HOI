"""Palm-up two-axis rotation. No images, object pose, or force magnitude in student observations.

DIGIT gel links remain separate rigid bodies so housing contacts cannot masquerade as touch.
The rigid proxy and binary tactile model require calibration before real hardware transfer.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from .base.vec_task import VecTask
from .manipulation_math import (OBS_DIM, PRIV_DIM, build_observation, append_history,
                                rotation_increment, rotation_terms, joint_pose_deviation)
from hora.utils.hand_collision import (collision_spec, collision_signature, configure_asset_collisions,
                             verify_actor_collisions, require_collision_signature)
from hora.utils.asset_geometry import verify_geometry_manifest

ROOT = Path(__file__).resolve().parents[2]
FINGERS = ('index', 'middle', 'ring', 'thumb')


class AllegroDigitTwoAxis(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        self.cfg = config['env']['manipulation']
        self.stage = config.get('stage', 'teacher')
        self.max_episode_length = config['env']['episodeLength']
        self.up_axis = 'z'
        self.prop_hist_len = self.cfg['history_length']
        self.history_dim = OBS_DIM
        self.catalog = config['env']['objects']
        if not self.catalog or len({x['name'] for x in self.catalog}) != len(self.catalog):
            raise ValueError('Object names must be nonempty and unique')
        asset_path = ROOT / config['env']['asset']['handAsset']
        if not asset_path.is_file():
            raise FileNotFoundError('Missing hand asset; see docs/official_digit.md for the matching builder: %s' % asset_path)
        verify_geometry_manifest(asset_path)
        self.asset_hash = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        self.collision_spec = self._collision_spec(asset_path)
        self.collision_hash = collision_signature(self.collision_spec)
        self.cache_path = ROOT / config['env']['grasp']['cache']
        self.grasp_cache = None
        if self.stage not in ('grasps', 'smoke'):
            if not self.cache_path.is_file():
                raise FileNotFoundError('Generate DIGIT-specific grasps first with stage=grasps')
            self.grasp_cache = torch.load(self.cache_path, map_location='cpu')
            require_collision_signature(self.grasp_cache, self.collision_hash, 'Grasp cache')
            if self.grasp_cache['asset_sha256'] != self.asset_hash or self.grasp_cache['objects'] != self.catalog:
                raise ValueError('Grasp cache asset/catalog mismatch; regenerate grasps')
            for obj in self.catalog:
                states = self.grasp_cache['states'].get(obj['name'])
                if states is None or len(states) == 0 or states.shape[1] != 23 or not torch.isfinite(states).all():
                    raise ValueError('Invalid or empty grasp cache for '+obj['name'])
                targets = self.grasp_cache.get('targets', {}).get(obj['name'])
                if targets is not None and (targets.shape != (len(states), 16) or not torch.isfinite(targets).all()):
                    raise ValueError('Invalid paired grasp targets for '+obj['name'])
        super().__init__(config, sim_device, graphics_device_id, headless)
        if self.grasp_cache is not None and self.grasp_cache.get('joint_names') != self.joint_names:
            raise ValueError('Grasp cache joint order mismatch; regenerate with current asset/Isaac Gym')
        self.dt = self.sim_params.dt
        self.control_dt = self.dt * self.control_freq_inv
        self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(-1, 13)
        self.dof_states = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, 16, 2)
        self.bodies = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, -1, 13)
        self.contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs, -1, 3)
        self.q, self.qdot = self.dof_states[..., 0], self.dof_states[..., 1]
        self.targets = torch.zeros_like(self.q)
        self.initial_q = torch.zeros_like(self.q)
        self.actions = torch.zeros_like(self.q)
        self.last_actions = torch.zeros_like(self.q)
        self.torques = torch.zeros_like(self.q)
        self.initial_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.axis_options = torch.tensor(self.cfg['axes'], device=self.device, dtype=torch.float)
        if self.axis_options.ndim != 2 or self.axis_options.shape[1] != 3 or (self.axis_options.norm(dim=-1) < 1e-6).any():
            raise ValueError('Each commanded axis must be a nonzero 3-vector')
        self.axis_options /= self.axis_options.norm(dim=-1, keepdim=True)
        self.axis_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.axes = self.axis_options[self.axis_ids].clone()
        self.segment_angle = torch.zeros(self.num_envs, device=self.device)
        self.total_angle = torch.zeros(self.num_envs, len(self.axis_options), device=self.device)
        self.next_goal = torch.full((self.num_envs,), self.cfg['goal_increment'], device=self.device)
        self.goal_hits = torch.zeros(self.num_envs, device=self.device)
        latency = config['env']['digit']['latency_steps']
        if latency < 0:
            raise ValueError('latency_steps must be non-negative')
        self.contact_queue = torch.zeros(self.num_envs, latency+1, 4, device=self.device)
        self.raw_contacts = torch.zeros(self.num_envs, 4, device=self.device)
        self.raw_housing = torch.zeros_like(self.raw_contacts)
        self.kp = torch.ones_like(self.q)*config['env']['controller']['pgain']
        self.kd = torch.ones_like(self.q)*config['env']['controller']['dgain']
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], gymapi.Vec3(0.5, 0.4, 0.95), gymapi.Vec3(0, 0, 0.6))
        self._refresh()

    def _allocate_task_buffer(self, num_envs):
        self.priv_info_buf = torch.zeros(num_envs, PRIV_DIM, device=self.device)
        self.proprio_hist_buf = torch.zeros(num_envs, self.prop_hist_len, OBS_DIM, device=self.device)

    def _collision_spec(self, asset_path):
        return collision_spec(asset_path)

    def _create_scene_assets(self):
        pass

    def _create_scene_actor(self, env, env_id):
        pass

    def _hand_pose(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0.5)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -np.pi/2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/2)
        return pose

    def _set_actor_friction(self, env, hand, obj, friction):
        for actor in [hand, obj]:
            shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
            for shape in shapes:
                shape.friction = friction
                shape.restitution = 0
            self.gym.set_actor_rigid_shape_properties(env, actor, shapes)

    def _create_envs(self, num_envs, spacing, num_per_row):
        if num_envs < len(self.catalog):
            raise ValueError('num_envs must cover every object in the catalog')
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane)
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        options.collapse_fixed_joints = False  # essential: separate gel vs housing contact tensors
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        self.hand_asset = self.gym.load_asset(self.sim, str(ROOT), self.config['env']['asset']['handAsset'], options)
        configure_asset_collisions(self.gym, self.hand_asset, self.collision_spec)
        if self.gym.get_asset_dof_count(self.hand_asset) != 16:
            raise ValueError('Expected a 16-DOF Allegro hand')
        self.joint_names = self.gym.get_asset_dof_names(self.hand_asset)
        self.canonical_order = [int(name.split('_')[1].split('.')[0]) for name in self.joint_names]
        print('DOF order:', self.joint_names, flush=True)
        body_names = self.gym.get_asset_rigid_body_names(self.hand_asset)
        self.gel_ids = [body_names.index('digit_'+f+'_gel') for f in FINGERS]
        self.housing_ids = [body_names.index('digit_'+f+'_housing') for f in FINGERS]
        props = self.gym.get_asset_dof_properties(self.hand_asset)
        self.lower = torch.tensor(props['lower'].copy(), device=self.device)
        self.upper = torch.tensor(props['upper'].copy(), device=self.device)
        self.effort = torch.tensor(props['effort'].copy(), device=self.device).clamp(max=0.5)
        props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        props['stiffness'].fill(0)
        props['damping'].fill(0)
        props['armature'].fill(0.001)
        props['friction'].fill(0.01)
        assets = []
        for obj in self.catalog:
            size = obj['size']
            opt = gymapi.AssetOptions()
            opt.density = 500
            if obj['shape'] == 'sphere':
                asset = self.gym.create_sphere(self.sim, size[0]/2, opt)
            elif obj['shape'] == 'box':
                asset = self.gym.create_box(self.sim, *size, opt)
            elif obj['shape'] == 'capsule':
                # PhysX capsule local axis is X; Gym's width argument is half cylinder length.
                asset = self.gym.create_capsule(self.sim, size[1]/2, max(0, (size[0]-size[1])/2), opt)
            else:
                raise ValueError('Unsupported primitive: '+obj['shape'])
            assets.append(asset)
        self._create_scene_assets()
        hand_pose = self._hand_pose()
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(*self.config['env']['grasp']['object_position'])
        self.envs, hand_indices, object_indices = [], [], []
        domains, object_ids = [], []
        domain = self.config['env']['domain']
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), num_per_row)
            # -1 preserves the asset's explicit per-shape masks. Never use a
            # shared nonzero mask, which disables every hand/hand contact.
            hand = self.gym.create_actor(env, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env, hand, props)
            oid = i % len(assets)
            obj = self.gym.create_actor(env, assets[oid], obj_pose, 'object', i, 0, 0)
            friction = float(np.random.uniform(*domain['friction']))
            self._set_actor_friction(env, hand, obj, friction)
            self._create_scene_actor(env, i)
            verify_actor_collisions(self.gym, env, hand, self.collision_spec)
            mass = float(np.random.uniform(*domain['mass']))
            com = np.random.uniform(-domain['com'], domain['com'], 3)
            rb = self.gym.get_actor_rigid_body_properties(env, obj)
            ratio = mass / rb[0].mass
            for axis in ('x', 'y', 'z'):
                row = getattr(rb[0].inertia, axis)
                row.x *= ratio; row.y *= ratio; row.z *= ratio
            rb[0].mass = mass
            rb[0].com = gymapi.Vec3(*com)
            self.gym.set_actor_rigid_body_properties(env, obj, rb, False)
            domains.append(self.catalog[oid]['size'] + [mass, friction] + com.tolist())
            object_ids.append(oid)
            self.envs.append(env)
            hand_indices.append(self.gym.get_actor_index(env, hand, gymapi.DOMAIN_SIM))
            object_indices.append(self.gym.get_actor_index(env, obj, gymapi.DOMAIN_SIM))
        self.hand_indices = torch.tensor(hand_indices, device=self.device, dtype=torch.long)
        self.object_indices = torch.tensor(object_indices, device=self.device, dtype=torch.long)
        self.object_ids = torch.tensor(object_ids, device=self.device, dtype=torch.long)
        self.domain_params = torch.tensor(domains, device=self.device, dtype=torch.float)

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_state = self.root_states[self.object_indices]

    def _reset_task_state(self, ids):
        pass

    def _failure_conditions(self, gel, displacement, finite):
        return ((self.object_state[:, 2] < self.cfg['min_height']) |
                (displacement > self.cfg['max_displacement']) | ~finite)

    def _task_extras(self):
        return {}

    def reset_idx(self, ids):
        if len(ids) == 0:
            return
        grasp = self.config['env']['grasp']
        poses = torch.zeros(len(ids), 23, device=self.device)
        reset_targets = torch.zeros(len(ids), 16, device=self.device)
        if self.grasp_cache is None:
            # Config is numeric joint_0..15 order; PhysX may reorder the kinematic tree.
            poses[:, :16] = torch.tensor(grasp['canonical_q'], device=self.device)[self.canonical_order]
            poses[:, :16] += (torch.rand(len(ids), 16, device=self.device)*2-1)*grasp['q_noise']
            poses[:, 16:19] = torch.tensor(grasp['object_position'], device=self.device)
            poses[:, 16:18] += (torch.rand(len(ids), 2, device=self.device)*2-1)*0.004
            poses[:, 22] = 1
            reset_targets.copy_(poses[:, :16])
        else:
            for oid, obj in enumerate(self.catalog):
                mask = self.object_ids[ids] == oid
                count = int(mask.sum())
                if count:
                    bank = self.grasp_cache['states'][obj['name']]
                    sample_ids = torch.randint(len(bank), (count,))
                    sampled = bank[sample_ids]
                    poses[mask] = sampled.to(self.device)
                    target_bank = self.grasp_cache.get('targets', {}).get(obj['name'], bank[:, :16])
                    reset_targets[mask] = target_bank[sample_ids].to(self.device)
        poses[:, :16] = torch.max(torch.min(poses[:, :16], self.upper), self.lower)
        self.q[ids] = poses[:, :16]
        self.initial_q[ids] = poses[:, :16]
        self.qdot[ids] = 0
        self.targets[ids] = torch.max(torch.min(reset_targets, self.upper), self.lower)
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.root_states[self.object_indices[ids], :7] = poses[:, 16:23]
        self.root_states[self.object_indices[ids], 7:] = 0
        self.initial_position[ids] = poses[:, 16:19]
        self.prev_pose[ids] = poses[:, 16:23]
        self.contact_queue[ids] = 0
        self.raw_contacts[ids] = 0
        self.raw_housing[ids] = 0
        gain_range = self.config['env']['domain']['gain_scale']
        scale = torch.rand(len(ids), 16, device=self.device)*(gain_range[1]-gain_range[0])+gain_range[0]
        self.kp[ids] = self.p_gain*scale
        self.kd[ids] = self.d_gain*scale
        self.axis_ids[ids] = torch.randint(len(self.axis_options), (len(ids),), device=self.device) if self.cfg['random_initial_axis'] else 0
        self.axes[ids] = self.axis_options[self.axis_ids[ids]]
        self.segment_angle[ids] = 0
        self.total_angle[ids] = 0
        self.next_goal[ids] = self.cfg['goal_increment']
        self.goal_hits[ids] = 0
        self.progress_buf[ids] = 0
        self.proprio_hist_buf[ids] = 0
        self.at_reset_buf[ids] = 1
        self._reset_task_state(ids)
        hi, oi = self.hand_indices[ids].to(torch.int32), self.object_indices[ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(hi), len(hi))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(oi), len(oi))

    def pre_physics_step(self, actions):
        self.last_actions.copy_(self.actions)
        alpha = self.cfg['action_smoothing']
        self.actions.copy_(alpha*actions.to(self.device)+(1-alpha)*self.actions)
        requested = self.cfg['action_scale']*self.actions
        previous_targets = self.targets.clone()
        self.targets.add_(requested)
        self.targets.copy_(torch.max(torch.min(self.targets, self.upper), self.lower))
        self.target_increment_lost = (requested-(self.targets-previous_targets)).abs().sum(-1)/requested.abs().sum(-1).clamp_min(1e-8)
        self.prev_pose.copy_(self.object_state[:, :7])

    def update_low_level_control(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        torque = self.kp*(self.targets-self.q)-self.kd*self.qdot
        self.torques.copy_(torch.max(torch.min(torque, self.effort), -self.effort))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def _observe(self):
        threshold = self.config['env']['digit']['contact_threshold']
        self.raw_contacts.copy_((self.contact_forces[:, self.gel_ids].norm(dim=-1) > threshold).float())
        self.raw_housing.copy_((self.contact_forces[:, self.housing_ids].norm(dim=-1) > threshold).float())
        # No contact coordinates or force magnitudes are exposed to the actor.
        sensed = self.raw_contacts * (torch.rand_like(self.raw_contacts) >= self.config['env']['digit']['dropout'])
        self.contact_queue.copy_(torch.cat((self.contact_queue[:, 1:], sensed[:, None]), 1))
        self.contact_queue[self.at_reset_buf.bool()] = 0
        contacts = self.contact_queue[:, 0].clone()
        at_reset = self.at_reset_buf.bool()
        contacts[at_reset] = 0  # contact tensor may still describe the pre-reset state
        q = self.q+(2*torch.rand_like(self.q)-1)*self.cfg['joint_noise']
        vel = self.qdot+torch.randn_like(self.qdot)*self.cfg['velocity_noise']
        frame = build_observation(q, vel, self.targets, self.actions, contacts, self.axes, self.lower, self.upper)
        self.obs_buf.copy_(frame)
        self.proprio_hist_buf.copy_(append_history(self.proprio_hist_buf, frame, at_reset))
        self.at_reset_buf.zero_()
        # privileged state is kept in a separate key; student code never reads it.
        self.priv_info_buf.copy_(torch.cat((self.object_state[:, :3]-self.initial_position,
                                           self.object_state[:, 3:13], self.domain_params,
                                           self.segment_angle[:, None], self.next_goal[:, None]), -1))

    def post_physics_step(self):
        if self.device == 'cpu':
            # CPU body-pair contact queries require the last substep to finish.
            self.gym.fetch_results(self.sim, True)
        self.progress_buf += 1
        self._refresh()
        delta = rotation_increment(self.object_state[:, 3:7], self.prev_pose[:, 3:7])
        omega = delta/self.control_dt
        rot, off, fast, along = rotation_terms(omega, self.axes, self.cfg['target_speed'])
        angle = (delta*self.axes).sum(-1)
        self.segment_angle += angle
        self.total_angle.scatter_add_(1, self.axis_ids[:, None], angle[:, None])
        tolerance = self.cfg['student_goal_tolerance'] if self.stage == 'student' else self.cfg['goal_tolerance']
        hit = self.segment_angle >= self.next_goal-tolerance
        self.next_goal += hit*self.cfg['goal_increment']
        self.goal_hits += hit
        gel = self.contact_forces[:, self.gel_ids].norm(dim=-1) > self.config['env']['digit']['contact_threshold']
        housing = self.contact_forces[:, self.housing_ids].norm(dim=-1) > self.config['env']['digit']['contact_threshold']
        displacement = (self.object_state[:, :3]-self.initial_position).norm(dim=-1)
        pose_penalty = self.cfg.get('pose_weight', 0.0)*joint_pose_deviation(self.q, self.initial_q)
        finite = torch.isfinite(self.object_state).all(-1) & torch.isfinite(self.q).all(-1)
        failed = self._failure_conditions(gel, displacement, finite)
        timeout = self.progress_buf >= self.max_episode_length
        reward = (self.cfg['rotation_weight']*rot-self.cfg['off_axis_weight']*off
                  -self.cfg['overspeed_weight']*fast+self.cfg['goal_weight']*hit
                  +self.cfg['contact_weight']*(gel.sum(-1)>=2)
                  -self.cfg['housing_weight']*housing.float().mean(-1)
                  -self.cfg['position_weight']*displacement
                  -self.cfg['linear_velocity_weight']*self.object_state[:, 7:10].norm(dim=-1)
                  -self.cfg['action_rate_weight']*(self.actions-self.last_actions).square().sum(-1)
                  -self.cfg['torque_weight']*self.torques.square().sum(-1)
                  -self.cfg['drop_penalty']*failed-pose_penalty)
        self.rew_buf.copy_(torch.nan_to_num(reward, nan=-self.cfg['drop_penalty']))
        self.reset_buf.copy_((failed | timeout).long())
        self.extras = {'rotation/speed': along.mean(), 'rotation/off_axis': off.mean(),
                       'contact/gel_count': gel.float().sum(-1).mean(),
                       'contact/housing_count': housing.float().sum(-1).mean(),
                       'episode/failed': failed.float().mean(),
                       'episode/time_limit': timeout.float().mean()}
        self.extras.update({'reward/pose_penalty': -pose_penalty.mean(),
                            'control/target_increment_lost': self.target_increment_lost.mean(),
                            'control/target_at_limit': ((self.targets-self.lower<.001)|(self.upper-self.targets<.001)).float().mean(),
                            'control/torque_at_limit': (self.torques.abs()>=self.effort-1e-5).float().mean()})
        self.extras.update(self._task_extras())
        # Terminal metrics are captured before auto-reset; evaluator consumes these separately.
        self.terminal = {'done': self.reset_buf.bool().clone(), 'failed': failed.clone(),
                         'angles': self.total_angle.clone(), 'length': self.progress_buf.clone()}
        if self.stage != 'grasps':
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))
        switch = self.cfg['switch_steps']
        if switch > 0:
            ids = ((self.progress_buf > 0) & (self.progress_buf % switch == 0)).nonzero(as_tuple=False).squeeze(-1)
            self.axis_ids[ids] = (self.axis_ids[ids]+1) % len(self.axis_options)
            self.axes[ids] = self.axis_options[self.axis_ids[ids]]
            self.segment_angle[ids] = 0
            self.next_goal[ids] = self.cfg['goal_increment']
        self._refresh()
        self._observe()

    def _observations(self):
        return {'obs': self.obs_buf.to(self.rl_device), 'priv_info': self.priv_info_buf.to(self.rl_device),
                'proprio_hist': self.proprio_hist_buf.to(self.rl_device)}

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()
        self._observe()
        return self._observations()

    def step(self, actions):
        _, reward, done, info = super().step(actions)
        return self._observations(), reward, done, info

    def generate_grasps(self):
        if self.sim_params.use_gpu_pipeline:
            raise ValueError('Grasp validation needs body-pair contacts: use pipeline=cpu rl_device=cpu (GPU PhysX is supported)')
        cfg = self.config['env']['grasp']
        object_bodies = [self.gym.get_actor_rigid_body_index(env, 1, 0, gymapi.DOMAIN_ENV)
                         for env in self.envs]
        banks = {obj['name']: [] for obj in self.catalog}
        target_banks = {obj['name']: [] for obj in self.catalog}
        counts = {name: 0 for name in banks}
        for batch in range(cfg['max_batches']):
            self.reset()
            valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for t in range(cfg['candidate_steps']):
                self.step(torch.zeros_like(self.actions))
                if t >= cfg['warmup_steps']:
                    # Net force alone cannot identify the other body now that
                    # fingers can collide. Use real PhysX pairs for cache filtering.
                    object_contacts = torch.zeros_like(self.raw_contacts, dtype=torch.bool)
                    for i, env in enumerate(self.envs):
                        for contact in self.gym.get_env_rigid_contacts(env):
                            a, b = int(contact['body0']), int(contact['body1'])
                            object_body = object_bodies[i]
                            other = b if a == object_body else a if b == object_body else -1
                            if other in self.gel_ids and contact['lambda'] > 0:
                                object_contacts[i, self.gel_ids.index(other)] = True
                    valid &= (((object_contacts & self.raw_contacts.bool()).sum(-1) >= 2) & ~self.terminal['failed']
                              & (self.object_state[:, 7:10].norm(dim=-1) < 0.1)
                              & (self.object_state[:, 10:13].norm(dim=-1) < 1.0))
            states = torch.cat((self.q, self.object_state[:, :7]), -1).detach().cpu()
            for oid, obj in enumerate(self.catalog):
                selected_mask = (valid & (self.object_ids == oid)).cpu()
                selected = states[selected_mask]
                banks[obj['name']].append(selected)
                target_banks[obj['name']].append(self.targets.detach().cpu()[selected_mask].clone())
                counts[obj['name']] += len(selected)
            print('grasp batch', batch+1, counts, flush=True)
            if min(counts.values()) >= cfg['required_per_object']:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                if self.cache_path.exists():
                    raise FileExistsError('Refusing to overwrite existing grasp cache: '+str(self.cache_path))
                torch.save({'asset_sha256': self.asset_hash, 'collision_sha256': self.collision_hash,
                            'grasp_contact_validation': 'physx_body_pairs_v1',
                            'objects': self.catalog,
                            'joint_names': self.joint_names,
                            'targets': {k: torch.cat(v)[:cfg['required_per_object']] for k, v in target_banks.items()},
                            'states': {k: torch.cat(v)[:cfg['required_per_object']] for k, v in banks.items()}}, self.cache_path)
                return counts
        raise RuntimeError('Not enough stable gel-contact grasps. Inspect geometry/initial pose; no cache saved. Counts: '+json.dumps(counts))

    def close(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
