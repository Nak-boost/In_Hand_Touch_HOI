"""Explicit pair exclusions for the separate Allegro/DIGIT rigid bodies.

Isaac Gym shapes collide iff their filter masks have no common bit. A unique
bit per excluded pair avoids accidentally disabling all collisions on a finger.
"""
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET


def collision_spec(asset_path):
    robot = ET.parse(asset_path).getroot()
    links = {link.get('name') for link in robot.findall('link')}
    parent = {name: name for name in links}

    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name

    excluded = set()
    for joint in robot.findall('joint'):
        a, b = joint.find('parent').get('link'), joint.find('child').get('link')
        excluded.add(tuple(sorted((a, b))))
        if joint.get('type') == 'fixed':
            parent[root(a)] = root(b)
    # A distal link, its housing and gel form one fixed assembly. Include the
    # distal/gel pair as well as the two immediate fixed-joint pairs.
    for a, b in itertools.combinations(sorted(links), 2):
        if root(a) == root(b):
            excluded.add((a, b))
    pairs = sorted(excluded)
    if len(pairs) > 31:
        raise ValueError('Collision exclusions exceed the supported signed 32-bit mask; redesign explicitly')
    masks = {name: 0 for name in sorted(links)}
    for bit, (a, b) in enumerate(pairs):
        masks[a] |= 1 << bit
        masks[b] |= 1 << bit
    return dict(version='adjacent_and_fixed_assembly_v1', excluded_pairs=pairs, body_masks=masks)


def collision_signature(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def configure_asset_collisions(gym, asset, spec):
    names = gym.get_asset_rigid_body_names(asset)
    ranges = gym.get_asset_rigid_body_shape_indices(asset)
    shapes = gym.get_asset_rigid_shape_properties(asset)
    if set(names) != set(spec['body_masks']):
        raise ValueError('Loaded rigid bodies differ from collision specification')
    for name, shape_range in zip(names, ranges):
        for index in range(shape_range.start, shape_range.start + shape_range.count):
            shapes[index].filter = spec['body_masks'][name]
    if not gym.set_asset_rigid_shape_properties(asset, shapes):
        raise RuntimeError('Failed to apply hand collision filters')


def verify_actor_collisions(gym, env, actor, spec):
    names = gym.get_actor_rigid_body_names(env, actor)
    ranges = gym.get_actor_rigid_body_shape_indices(env, actor)
    shapes = gym.get_actor_rigid_shape_properties(env, actor)
    for name, shape_range in zip(names, ranges):
        for index in range(shape_range.start, shape_range.start + shape_range.count):
            if shapes[index].filter != spec['body_masks'][name]:
                raise RuntimeError('Actor creation overwrote the collision mask for '+name)


def require_collision_signature(record, expected, description):
    if record.get('collision_sha256') != expected:
        raise ValueError(description+' collision settings mismatch; regenerate data/train with current collisions')
