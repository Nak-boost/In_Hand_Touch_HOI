"""Palm-down tabletop rotation with the user-authorized gel/table collision filter."""
import math
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from .allegro_digit_two_axis import AllegroDigitTwoAxis
from .tabletop_math import tabletop_collision_spec, tabletop_failures


class AllegroDigitTabletop(AllegroDigitTwoAxis):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.table_cfg = config['env']['tabletop']
        super().__init__(config, sim_device, graphics_device_id, headless)
        self.missing_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.workspace_center = torch.tensor(self.table_cfg['workspace_center'], device=self.device)
        self.missing_limit = max(1, math.ceil(self.table_cfg['contact_loss_seconds'] / self.control_dt - 1e-6))
        self.outside_workspace = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_lost = torch.zeros_like(self.outside_workspace)
        if self.viewer:
            center = self.table_cfg['workspace_center']
            z = self.table_cfg['surface_height']
            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], gymapi.Vec3(.4, -.5, z+.4),
                                          gymapi.Vec3(center[0], center[1], z+.05))

    def _collision_spec(self, asset_path):
        return tabletop_collision_spec(super()._collision_spec(asset_path), self.table_cfg)

    def _hand_pose(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.table_cfg['hand_position'])
        # Local palm +X points downward; local finger extension remains horizontal.
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi/2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/2)
        return pose

    def _create_scene_assets(self):
        opt = gymapi.AssetOptions()
        opt.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, *self.table_cfg['size'], opt)
        shapes = self.gym.get_asset_rigid_shape_properties(self.table_asset)
        for shape in shapes:
            shape.filter = self.collision_spec['table_filter']
            shape.friction = self.table_cfg['friction']
            shape.restitution = 0
        self.gym.set_asset_rigid_shape_properties(self.table_asset, shapes)
        self.table_handles = []

    def _create_scene_actor(self, env, env_id):
        pose = gymapi.Transform()
        xy = self.table_cfg['workspace_center']
        pose.p = gymapi.Vec3(xy[0], xy[1], self.table_cfg['surface_height'] - self.table_cfg['size'][2]/2)
        actor = self.gym.create_actor(env, self.table_asset, pose, 'table', env_id, -1, 0)
        self.table_handles.append(actor)
        shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
        if any(s.filter != self.collision_spec['table_filter'] for s in shapes):
            raise RuntimeError('Table collision mask was not preserved')

    def _set_actor_friction(self, env, hand, obj, friction):
        shapes = self.gym.get_actor_rigid_shape_properties(env, hand)
        ranges = self.gym.get_asset_rigid_body_shape_indices(self.hand_asset)
        for body, span in enumerate(ranges):
            for k in range(span.start, span.start + span.count):
                shapes[k].friction = self.table_cfg['gel_friction'] if body in self.gel_ids else self.table_cfg['hand_friction']
                shapes[k].restitution = 0
        self.gym.set_actor_rigid_shape_properties(env, hand, shapes)
        shapes = self.gym.get_actor_rigid_shape_properties(env, obj)
        for shape in shapes:
            shape.friction = friction
            shape.restitution = 0
        self.gym.set_actor_rigid_shape_properties(env, obj, shapes)

    def _reset_task_state(self, ids):
        self.missing_steps[ids] = 0

    def _failure_conditions(self, gel, displacement, finite):
        if not finite.all():
            # Numerical corruption is an error, not a third task failure criterion.
            raise FloatingPointError('Nonfinite tabletop simulation state')
        failed, self.missing_steps, self.outside_workspace, self.contact_lost = tabletop_failures(
            self.object_state[:, :3], gel, self.missing_steps, self.workspace_center,
            self.table_cfg['workspace_radius'], self.missing_limit)
        return failed

    def _task_extras(self):
        return {'episode/outside_workspace': self.outside_workspace.float().mean(),
                'episode/contact_lost': self.contact_lost.float().mean(),
                'contact/missing_seconds': self.missing_steps.float().mean()*self.control_dt}

    def generate_grasps(self):
        """Slow closing from a slightly open pose; accept actual light object contacts."""
        if self.sim_params.use_gpu_pipeline:
            raise ValueError('Generate tabletop starts with pipeline=cpu rl_device=cpu for contact-pair validation')
        from hora.utils.tabletop_contacts import object_contact_summary
        cfg = self.config['env']['grasp']
        if self.cache_path.exists():
            raise FileExistsError(self.cache_path)
        banks = {obj['name']: [] for obj in self.catalog}
        target_banks = {name: [] for name in banks}
        counts = {name: 0 for name in banks}
        close_steps = self.table_cfg['initialization']['close_steps']
        hold_steps = self.table_cfg['initialization']['hold_steps']
        for batch in range(cfg['max_batches']):
            self.reset()
            closed = self.targets.clone()
            preload = torch.tensor(cfg.get('preload_delta', [0.]*16), device=self.device)[self.canonical_order]
            closed = torch.max(torch.min(closed + preload, self.upper), self.lower)
            opened = closed.clone()
            for number in [1, 5, 9, 13]:
                j = self.canonical_order.index(number)
                opened[:, j] = (opened[:, j] - self.table_cfg['initialization']['opening_delta']).clamp(self.lower[j], self.upper[j])
            self.q.copy_(opened); self.qdot.zero_(); self.targets.copy_(opened)
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
            good = torch.zeros(self.num_envs, device=self.device)
            valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for step in range(close_steps + hold_steps):
                alpha = min(1., (step + 1)/close_steps)
                self.targets.copy_(opened + alpha*(closed-opened))
                self.step(torch.zeros_like(self.actions))
                if step >= close_steps:
                    contacts = object_contact_summary(self)
                    gel_force = self.contact_forces[:, self.gel_ids].norm(dim=-1)
                    effective = contacts['gels'] & (gel_force > self.table_cfg['initialization']['min_gel_force'])
                    good += ((effective.sum(-1) >= 2) & contacts['table'] & ~contacts['other_hand']).float()
                    valid &= (~self.terminal['failed'] &
                              (gel_force.max(-1).values <= self.table_cfg['initialization']['max_gel_force']) &
                              (self.object_state[:, 7:10].norm(dim=-1) < .05))
            valid &= good/hold_steps >= .9
            valid &= (effective.sum(-1) >= 2) & contacts['table'] & ~contacts['other_hand']
            states = torch.cat((self.q, self.object_state[:, :7]), -1).detach().cpu()
            for oid, obj in enumerate(self.catalog):
                keep = (valid & (self.object_ids == oid)).cpu()
                banks[obj['name']].append(states[keep])
                target_banks[obj['name']].append(self.targets.detach().cpu()[keep].clone())
                counts[obj['name']] += int(keep.sum())
            print('tabletop light-contact batch', batch+1, counts, flush=True)
            if min(counts.values()) >= cfg['required_per_object']:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'asset_sha256': self.asset_hash, 'collision_sha256': self.collision_hash,
                            'objects': self.catalog, 'joint_names': self.joint_names,
                            'grasp_contact_validation': 'tabletop_light_object_pairs_v1',
                            'scene': self.table_cfg,
                            'states': {k: torch.cat(v)[:cfg['required_per_object']] for k,v in banks.items()},
                            'targets': {k: torch.cat(v)[:cfg['required_per_object']] for k,v in target_banks.items()}}, self.cache_path)
                return counts
        raise RuntimeError('No sufficient light tabletop starts; inspect hand height and closing pose: '+str(counts))
