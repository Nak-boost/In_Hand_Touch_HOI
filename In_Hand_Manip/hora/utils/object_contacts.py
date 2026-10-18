"""Object-specific gel contact loads for reward/termination, never actor inputs.

Isaac Gym Preview 4 exposes pair identities through CPU contact queries, not
the GPU net-force tensor. PhysX itself may still run on GPU with a CPU pipeline.
"""
import numpy as np
import torch
import time


def validate_object_contact_filter(enabled, sim_config):
    if not isinstance(enabled, bool):
        raise ValueError('object_contact_filter must be a boolean')
    if not enabled:
        return
    if sim_config.get('use_gpu_pipeline', False):
        raise ValueError('object_contact_filter=true requires pipeline=cpu in Isaac Gym Preview 4. '
                         'Keep sim_device=cuda:0 and rl_device=cuda:0 for GPU physics/PPO; '
                         'the CPU contact API cannot run in the GPU tensor pipeline. '
                         'No pipeline switch is made automatically.')
    if sim_config.get('physx', {}).get('contact_collection', 2) == 0:
        raise ValueError('object_contact_filter requires PhysX contact_collection > 0')


def gel_object_normal_load(contacts, object_body, gel_lookup):
    """Sum positive normal loads (N) for each gel-object pair in one environment.

    RigidContact.lambda is documented as contact force magnitude in Preview 4;
    do not divide it by dt. This is compressive normal load, not the magnitude
    of a net vector including shear. Multiple manifold points add to the load.
    Identity is checked before aggregation, so self/table/housing loads cannot
    make a weak object contact cross the threshold.
    """
    loads = np.zeros(4, dtype=np.float32)
    if len(contacts) == 0:
        return loads
    a, b = contacts['body0'], contacts['body1']
    selected = (a == object_body) | (b == object_body)
    other = np.where(a[selected] == object_body, b[selected], a[selected])
    valid = (other >= 0) & (other < len(gel_lookup))
    slots = gel_lookup[other[valid]]
    force = contacts['lambda'][selected][valid]
    gels = slots >= 0
    slots, force = slots[gels], force[gels]
    if not np.isfinite(force).all():
        raise FloatingPointError('Nonfinite gel-object contact force')
    positive = force > 0
    loads[:] = np.bincount(slots[positive], weights=force[positive], minlength=4)
    return loads


class GelObjectContactReader:
    """One pair query per environment/control step, shared by reward and failure."""
    def __init__(self, env):
        from isaacgym import gymapi
        validate_object_contact_filter(True, env.config['sim'])
        if env.device != 'cpu':
            raise ValueError('Pair contact reader requires CPU state tensors')
        n_bodies = env.gym.get_asset_rigid_body_count(env.hand_asset)
        self.gel_lookup = np.full(n_bodies, -1, dtype=np.int64)
        self.gel_lookup[env.gel_ids] = np.arange(4)
        # Object actor is created after the hand; works for tabletop and in-hand.
        self.object_bodies = [env.gym.get_actor_rigid_body_index(e, 1, 0, gymapi.DOMAIN_ENV)
                              for e in env.envs]
        self.loads = np.zeros((env.num_envs, 4), dtype=np.float32)
        self.tensor = torch.from_numpy(self.loads)
        self.last_wall_ms = 0.0
        self.last_cpu_ms = 0.0

    def refresh(self, env):
        # Caller fetches the completed final physics step before querying. Do not
        # retain Gym's borrowed contact arrays across calls or simulation steps.
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        for i, e in enumerate(env.envs):
            contacts = env.gym.get_env_rigid_contacts(e)
            self.loads[i] = gel_object_normal_load(contacts, self.object_bodies[i], self.gel_lookup)
        self.last_cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
        self.last_wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        return self.tensor
