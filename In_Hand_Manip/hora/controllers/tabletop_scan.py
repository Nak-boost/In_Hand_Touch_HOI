"""Contact-adaptive cyclic scanning. Inputs are proprioception and tactile load only.

The fixed workspace centre is a placement assumption, not live object tracking.
Simulated load is an ideal sensor proxy; real DIGIT needs a calibrated estimator.
"""
import numpy as np
from hora.utils.mount_geometry import rotation


class TabletopScanController:
    PHASES = ('settle', 'approach', 'hold', 'rotate', 'release', 'reset', 'fault')

    def __init__(self, config, open_positions, open_normals, center, dt, rest_normals=None):
        self.cfg = config
        self.open = np.asarray(open_positions).copy()
        self.open_normals = np.asarray(open_normals).copy()
        self.center = np.asarray(center).copy()
        self.dt = dt
        radial = self.open-self.center
        radial[:, 2] = 0
        self.radial = radial/np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        self.desired = self.open.copy()
        self.rest_normals = self.open_normals.copy() if rest_normals is None else np.asarray(rest_normals).copy()
        self.normals = self.rest_normals.copy()
        self.phase = 'settle'
        self.age = 0.
        self.stable = 0.
        self.clear = 0.
        self.cycles = 0
        self.reason = ''
        self.filtered = np.zeros(4)
        self.touch = np.zeros(4, dtype=bool)
        self.events = []
        self.clock = 0.
        self.rotation_elapsed = 0.
        self.lost_seconds = 0.
        self.preloaded_seconds = 0.

    def transition(self, phase, reason=''):
        self.events.append(dict(time=self.clock, source=self.phase, target=phase,
                                cycle=self.cycles, reason=reason))
        self.phase, self.age, self.stable = phase, 0., 0.
        self.reason = reason
        self.preloaded_seconds = 0.
        if phase == 'approach':
            self.normals = self.open_normals.copy()
        if phase == 'rotate':
            self.rotation_elapsed = self.lost_seconds = 0.

    def update(self, positions, load, measured_normals=None):
        c = self.cfg
        self.clock += self.dt
        self.age += self.dt
        self.filtered = c['force_filter']*np.asarray(load)+(1-c['force_filter'])*self.filtered
        self.touch = np.where(self.touch, self.filtered > c['contact_off'], self.filtered > c['contact_on'])
        normals = self.normals if measured_normals is None else np.asarray(measured_normals)
        if not np.isfinite(positions).all() or not np.isfinite(load).all() or not np.isfinite(normals).all():
            self.transition('fault', 'nonfinite_sensor')
        elif np.max(load) > c['abort_force'] and self.phase not in ('release', 'reset', 'fault'):
            self.transition('release', 'overload')
        self.stable = self.stable+self.dt if self.touch.sum() >= c['required_contacts'] else 0.
        self.clear = self.clear+self.dt if not self.touch.any() else 0.
        # Vertical sensing plane means a horizontal outward sensing normal.
        vertical_ok = np.abs(normals[:, 2]) <= np.sin(np.deg2rad(c.get('max_preload_tilt_degrees', 12.)))
        aligned = np.sum(normals*self.open_normals, axis=1) >= np.cos(np.deg2rad(c.get('max_preload_normal_error_degrees', 15.)))
        loaded = self.touch & (self.filtered >= c.get('minimum_hold_force', .3)) & vertical_ok & aligned
        self.preloaded_seconds = self.preloaded_seconds+self.dt if loaded.sum() >= c['required_contacts'] else 0.
        if self.phase == 'settle':
            if self.age >= c['settle_seconds']:
                if self.touch.any():
                    self.transition('fault', 'initial_pose_in_contact')
                else:
                    self.transition('approach')
        elif self.phase in ('approach', 'hold'):
            inward_speed = np.where(self.touch,
                np.clip(c['force_gain']*(c['hold_force']-self.filtered), -c['approach_speed'], c['approach_speed']),
                c['approach_speed'])
            self.desired -= self.radial*inward_speed[:, None]*self.dt
            if np.max(np.linalg.norm(self.desired-self.open, axis=1)) > c['max_closure']:
                self.transition('release', 'closure_limit')
            elif self.phase == 'approach' and self.stable >= c['contact_dwell']:
                self.transition('hold')
            elif self.phase == 'hold' and self.preloaded_seconds >= c['hold_seconds']:
                self.anchor = self.desired.copy()
                self.anchor_normals = self.normals.copy()
                self.radial_offset = np.zeros(4)
                self.transition('rotate')
            elif self.age >= c['approach_timeout']:
                self.transition('release', 'contact_timeout')
        elif self.phase == 'rotate':
            supported = self.touch & (self.filtered >= c.get('minimum_rotation_force', c['contact_on']))
            enough_contact = supported.sum() >= c['minimum_rotation_contacts']
            self.lost_seconds = 0. if enough_contact else self.lost_seconds+self.dt
            # Pause tangential progression while the load servo restores contact.
            if enough_contact:
                self.rotation_elapsed += self.dt
            angle = np.deg2rad(c['rotation_degrees'])*min(1., self.rotation_elapsed/c['rotate_seconds'])
            r = rotation([0, 0, 1], angle)
            directions = self.radial @ r.T
            self.radial_offset += np.clip(c['force_gain']*(self.filtered-c['hold_force']),
                                         -c['approach_speed'], c['approach_speed'])*self.dt
            self.radial_offset = np.clip(self.radial_offset, -c['max_rotation_compliance'], c['max_rotation_compliance'])
            self.desired = (self.anchor-self.center) @ r.T+self.center+directions*self.radial_offset[:, None]
            self.normals = self.anchor_normals @ r.T
            if self.rotation_elapsed >= c['rotate_seconds']:
                self.transition('release', 'push_complete')
            elif self.lost_seconds >= c['loss_grace_seconds']:
                self.transition('release', 'lost_contact')
        elif self.phase == 'release':
            radial = positions-self.center
            radial[:, 2] = 0
            radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
            self.desired += radial*c['release_speed']*self.dt
            if self.age >= c['release_min_seconds'] and self.clear >= c['release_dwell']:
                self.transition('reset')
            elif self.age >= c['release_timeout']:
                self.transition('fault', 'release_timeout')
        elif self.phase == 'reset':
            error = self.open-self.desired
            distance = np.linalg.norm(error, axis=1, keepdims=True)
            self.desired += error*np.minimum(1., c['reset_speed']*self.dt/np.maximum(distance, 1e-9))
            self.normals = self.rest_normals.copy()
            if self.touch.any():
                self.transition('release', 'reset_contact')
            elif np.max(np.linalg.norm(positions-self.open, axis=1)) < c['reset_tolerance']:
                self.cycles += 1
                self.transition('approach')
            elif self.age >= c['reset_timeout']:
                self.transition('fault', 'reset_timeout')
        return self.desired.copy(), self.normals.copy()
