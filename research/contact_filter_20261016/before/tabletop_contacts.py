"""CPU-contact API diagnostics. Never supplies privileged contact identity to the actor."""
import torch
from isaacgym import gymapi


def object_contact_summary(env):
    gels = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
    table = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    other = torch.zeros_like(table)
    forbidden = []
    gel_set = set(env.gel_ids)
    n_hand = env.gym.get_asset_rigid_body_count(env.hand_asset)
    for i, e in enumerate(env.envs):
        ob = env.gym.get_actor_rigid_body_index(e, 1, 0, gymapi.DOMAIN_ENV)
        tb = env.gym.get_actor_rigid_body_index(e, env.table_handles[i], 0, gymapi.DOMAIN_ENV)
        for c in env.gym.get_env_rigid_contacts(e):
            if c['lambda'] <= 0:
                continue
            a,b = int(c['body0']),int(c['body1'])
            if ((a in gel_set and b == tb) or (b in gel_set and a == tb) or
                    (a in gel_set and b in gel_set and a != b)):
                forbidden.append([i,a,b])
            body = b if a == ob else a if b == ob else -1
            if body in gel_set: gels[i, env.gel_ids.index(body)] = True
            elif body == tb: table[i] = True
            elif 0 <= body < n_hand: other[i] = True
    return dict(gels=gels, table=table, other_hand=other, forbidden=forbidden)
