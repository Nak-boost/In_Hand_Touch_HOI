"""Cached URDF kinematics for Cartesian DIGIT control; no object state inputs."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from .mount_geometry import origin, rotation

FINGERS = ('index', 'middle', 'ring', 'thumb')


class ScanKinematics:
    def __init__(self, asset, hand_transform):
        robot = ET.parse(asset).getroot()
        parent_joint = {j.find('child').get('link'): j for j in robot.findall('joint')}
        self.chains = []
        self.lower, self.upper = np.zeros(16), np.zeros(16)
        self.base = np.asarray(hand_transform)
        for finger in FINGERS:
            chain, link = [], 'digit_'+finger+'_gel'
            while link != 'base_link':
                j = parent_joint[link]
                index = int(j.get('name').split('_')[1].split('.')[0]) if j.get('type') == 'revolute' else None
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if index is not None else None
                if index is not None:
                    self.lower[index] = float(j.find('limit').get('lower'))
                    self.upper[index] = float(j.find('limit').get('upper'))
                chain.append((origin(j.find('origin')), index, axis))
                link = j.find('parent').get('link')
            self.chains.append(chain[::-1])

    def evaluate(self, q):
        positions, normals, jacobians = [], [], []
        for chain in self.chains:
            pose = self.base.copy()
            axes = []
            for fixed, index, axis in chain:
                pose = pose @ fixed
                if index is not None:
                    axes.append((index, pose[:3, 3].copy(), pose[:3, :3] @ axis))
                    turn = np.eye(4)
                    turn[:3, :3] = rotation(axis, q[index])
                    pose = pose @ turn
            p, n = pose[:3, 3], pose[:3, 0]
            jac = np.zeros((6, 16))
            for index, pivot, axis in axes:
                jac[:3, index] = np.cross(axis, p-pivot)
                jac[3:, index] = axis
            positions.append(p); normals.append(n); jacobians.append(jac)
        return np.array(positions), np.array(normals), np.array(jacobians)

    def fit(self, q, positions, normals, normal_weight=.015):
        def residual(x):
            p, n, _ = self.evaluate(x)
            return np.r_[(p-positions).ravel(), normal_weight*(n-normals).ravel(), .0001*(x-q)]
        result = least_squares(residual, np.clip(q, self.lower+1e-5, self.upper-1e-5),
                               bounds=(self.lower+1e-6, self.upper-1e-6), max_nfev=150)
        return result.x


def cartesian_joint_target(q, positions, normals, jacobians, desired, desired_normals,
                           lower, upper, max_step=.026, normal_weight=.015, damping=.003):
    """Damped least squares, including the gel normal, respecting every joint limit."""
    j = jacobians.copy()
    j[:, 3:] *= normal_weight
    error = np.concatenate((desired-positions, normal_weight*np.cross(normals, desired_normals)), axis=1)
    a = j.reshape(-1, 16)
    dq = np.linalg.solve(a.T @ a + damping**2*np.eye(16), a.T @ error.ravel())
    dq = np.clip(dq, -max_step, max_step)
    return np.clip(q+dq, lower, upper)
