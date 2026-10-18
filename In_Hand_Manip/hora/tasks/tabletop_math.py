"""Pure helpers for the tabletop experiment; positions are in the environment frame."""
import copy
import itertools
import torch


def tabletop_collision_spec(base, table_config):
    spec = copy.deepcopy(base)
    used = 0
    for mask in spec['body_masks'].values():
        used |= mask
    bit = 1 << used.bit_length()
    if bit >= (1 << 31):
        raise ValueError('No safe unused collision bit for the tabletop experiment')
    gels = sorted(n for n in spec['body_masks'] if n.startswith('digit_') and n.endswith('_gel'))
    if len(gels) != 4:
        raise ValueError('Expected four independent DIGIT gel bodies')
    for name in gels:
        spec['body_masks'][name] |= bit
    spec['excluded_pairs'] = sorted(set(map(tuple, spec['excluded_pairs'])) | set(itertools.combinations(gels, 2)))
    spec.update(version='tabletop_shared_gels_table_v1', table_filter=bit,
                external_excluded_pairs=[['table', name] for name in gels],
                scene=copy.deepcopy(table_config))
    return spec


def tabletop_failures(position, gel_contact, previous_missing, center, radius, missing_limit):
    """No orientation, height, speed, or no-progress failure in this experiment."""
    missing = torch.where(gel_contact.any(-1), torch.zeros_like(previous_missing), previous_missing + 1)
    outside = (position[:, :2] - center).square().sum(-1) > radius * radius
    lost = missing >= missing_limit
    return outside | lost, missing, outside, lost
