"""URDF forward kinematics and exact box-overlap checks for assembly inspection.

No simulator, policy, grasp cache or optimization. All current hand collision
shapes are boxes; fail explicitly if another geometry is introduced.
"""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

FINGERS = dict(index=3, middle=7, ring=11, thumb=15)
# Diagnostic postures in numeric joint_0..15 order, NOT validated grasp states.
POSTURES = {
    'open': [0, .05, .05, .05] * 3 + [.3, .05, .05, .05],
    'half_closed': [0, .5, .35, .25] * 3 + [.9, .3, .4, .3],
    'opposition': [.08, 1.1, .4, .35, 0, 1.1, .4, .35, 0, 1.1, .4, .35, 1.1, 1, .8, .4],
}


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


def origin(element):
    result = np.eye(4)
    if element is not None:
        r, p, y = map(float, element.get('rpy', '0 0 0').split())
        result[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
        result[:3, 3] = list(map(float, element.get('xyz', '0 0 0').split()))
    return result


def forward_kinematics(asset, joints):
    robot = ET.parse(asset).getroot()
    transforms = {'base_link': np.eye(4)}
    remaining = robot.findall('joint')
    while remaining:
        ready = [j for j in remaining if j.find('parent').get('link') in transforms]
        if not ready:
            raise ValueError('Disconnected or cyclic URDF')
        for joint in ready:
            transform = origin(joint.find('origin'))
            if joint.get('type') == 'revolute':
                number = int(joint.get('name').split('_')[1].split('.')[0])
                value = joints[number]
                limit = joint.find('limit')
                if not float(limit.get('lower')) <= value <= float(limit.get('upper')):
                    raise ValueError('Diagnostic posture exceeds limit: '+joint.get('name'))
                turn = np.eye(4)
                turn[:3, :3] = rotation(list(map(float, joint.find('axis').get('xyz').split())), value)
                transform = transform @ turn
            elif joint.get('type') != 'fixed':
                raise ValueError('Unsupported joint type')
            transforms[joint.find('child').get('link')] = transforms[joint.find('parent').get('link')] @ transform
            remaining.remove(joint)
    return robot, transforms


def boxes(robot, transforms):
    result = []
    for link in robot.findall('link'):
        for collision in link.findall('collision'):
            box = collision.find('geometry/box')
            if box is None:
                raise ValueError('Inspection requires box collision geometry: '+link.get('name'))
            pose = transforms[link.get('name')] @ origin(collision.find('origin'))
            result.append((link.get('name'), pose, np.fromstring(box.get('size'), sep=' ')))
    return result


def box_overlap(a, b, tolerance=1e-7):
    """Separating-axis test; return minimum projection overlap, or zero.

    Positive values mean geometric interpenetration, not contact force.
    Touching faces are not counted as volumetric overlap.
    """
    _, pa, sa = a
    _, pb, sb = b
    ra, rb = pa[:3, :3], pb[:3, :3]
    displacement = pb[:3, 3]-pa[:3, 3]
    axes = [*ra.T, *rb.T] + [np.cross(x, y) for x in ra.T for y in rb.T]
    minimum = float('inf')
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis/norm
        depth = np.abs(ra.T @ axis) @ (sa/2) + np.abs(rb.T @ axis) @ (sb/2) - abs(displacement @ axis)
        if depth <= tolerance:
            return 0.
        minimum = min(minimum, float(depth))
    return minimum


def inspect(asset):
    from hora.utils.hand_collision import collision_spec
    excluded = set(map(tuple, collision_spec(asset)['excluded_pairs']))
    report = {'asset': str(Path(asset).name), 'postures': {}}
    for name, q in POSTURES.items():
        robot, transforms = forward_kinematics(asset, q)
        assembly = {}
        for finger, index in FINGERS.items():
            distal = transforms['link_%d.0' % index]
            gel = transforms['digit_'+finger+'_gel']
            housing = transforms['digit_'+finger+'_housing']
            size = np.fromstring(robot.find("link[@name='digit_%s_gel']/collision/geometry/box" % finger).get('size'), sep=' ')
            surface = gel[:3, 3] + gel[:3, 0]*size[0]/2
            local = distal[:3, :3].T @ (surface-distal[:3, 3])
            angle = np.degrees(np.arccos(np.clip(distal[:3, 0] @ gel[:3, 0], -1, 1)))
            longitudinal = np.degrees(np.arccos(np.clip(distal[:3, 2] @ housing[:3, 2], -1, 1)))
            assembly[finger] = {'normal_deviation_deg': float(angle),
                               'long_axis_deviation_deg': float(longitudinal),
                               'front_plane_offset_mm': float((local[0]-.0098)*1000)}
        overlaps, internal = [], []
        for a, b in itertools.combinations(boxes(robot, transforms), 2):
            pair = tuple(sorted((a[0], b[0])))
            depth = box_overlap(a, b)
            if depth:
                entry = {'bodies': list(pair), 'min_projection_overlap_mm': depth*1000}
                (internal if pair in excluded else overlaps).append(entry)
        report['postures'][name] = {'joint_angles': q, 'alignment': assembly,
                                    'enabled_collision_overlaps': overlaps,
                                    'excluded_assembly_overlaps': internal}
    return report
