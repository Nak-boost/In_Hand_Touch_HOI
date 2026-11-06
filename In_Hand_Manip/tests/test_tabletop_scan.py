"""Feedback transitions and kinematic correctness; independent of Isaac Gym."""
from pathlib import Path
import numpy as np
import pytest
from omegaconf import OmegaConf
from hora.controllers.tabletop_scan import TabletopScanController
from hora.utils.scan_kinematics import ScanKinematics, cartesian_joint_target
from hora.utils.primitive_assets import cylinder_asset
from hora.utils.mount_geometry import rotation

ROOT = Path(__file__).resolve().parents[1]


def controller():
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/tabletop_scan.yaml').scan)
    cfg['force_filter'] = 1.
    p = np.array([[.04, 0, .54], [0, -.04, .54], [-.04, 0, .54], [0, .04, .54]])
    n = -(p-np.array([0, 0, .54]))/.04
    return TabletopScanController(cfg, p, n, [0, 0, .54], .05), p


def test_requires_contact_dwell_before_rotation():
    control, p = controller()
    control.transition('approach')
    for _ in range(20):
        control.update(p, [0.5, 0.5, 0.5, 0])
    assert control.phase == 'approach'
    control.update(p, [0.5]*4)
    assert control.phase == 'approach'
    for _ in range(12):
        control.update(p, [0.5]*4)
    assert control.phase == 'rotate'


def test_release_requires_actual_clearance_before_reset():
    control, p = controller()
    control.transition('release')
    for _ in range(15):
        control.update(p, [.2, 0, 0, 0])
    assert control.phase == 'release'
    control.update(p, [0]*4)
    assert control.phase == 'release'
    for _ in range(10):
        control.update(p, [0]*4)
        if control.phase == 'reset':
            break
    assert control.phase == 'reset'
    control.update(p, [.3, 0, 0, 0])
    assert control.phase == 'release'
    assert control.reason == 'reset_contact'


def test_load_feedback_opens_overloaded_finger_and_aborts_extreme_load():
    control, p = controller()
    control.transition('hold')
    before = control.desired.copy()
    control.update(p, [1., .3, .5, .5])
    travel = np.sum((control.desired-before)*control.radial, axis=1)
    assert travel[0] > 0 and travel[1] < 0
    control.update(p, [4., .3, .5, .5])
    assert control.phase == 'release' and control.reason == 'overload'


def test_jacobian_matches_finite_difference_and_ik_reduces_pose_error():
    transform = np.eye(4)
    transform[:3, :3] = rotation([0, 1, 0], np.pi/2) @ rotation([1, 0, 0], np.pi/2)
    kin = ScanKinematics(ROOT/'assets/allegro/allegro_digit_compact_inward90.urdf', transform)
    q = np.array([.05, 1.1, .4, .3]*3+[1.1, .7, .4, .2])
    p, n, jac = kin.evaluate(q)
    for j in range(16):
        changed = q.copy(); changed[j] += 1e-6
        p2, n2, _ = kin.evaluate(changed)
        np.testing.assert_allclose((p2-p)/1e-6, jac[:, :3, j], atol=2e-7)
        np.testing.assert_allclose((n2-n)/1e-6, np.cross(jac[:, 3:, j], n), atol=1e-6)
    reachable = q+np.linspace(-.01, .01, 16)
    desired, dn, _ = kin.evaluate(reachable)
    target = cartesian_joint_target(q, p, n, jac, desired, dn, kin.lower, kin.upper)
    new, _, _ = kin.evaluate(target)
    assert np.linalg.norm(new-desired) < np.linalg.norm(p-desired)
    assert np.all(target >= kin.lower) and np.all(target <= kin.upper)


def test_cylinder_asset_is_reproducible_and_rejects_invalid_size(tmp_path):
    import xml.etree.ElementTree as ET
    path = cylinder_asset(tmp_path, [.06, .06, .07])
    text = path.read_text()
    assert path.name.count('.') == 1
    assert ET.fromstring(text).find('.//cylinder').get('radius') == '0.03'
    assert cylinder_asset(tmp_path, [.06, .06, .07]).read_text() == text
    with pytest.raises(ValueError):
        cylinder_asset(tmp_path, [.06, .05, .07])


def test_touch_threshold_alone_does_not_authorize_rotation():
    control, p = controller()
    control.transition('approach')
    for _ in range(30):
        control.update(p, [.18]*4)
    assert control.touch.all()
    assert control.phase == 'hold'  # touched, but not enough preload
    for _ in range(7):
        control.update(p, [.5]*4)
    assert control.phase == 'rotate'


def test_loaded_but_tilted_sensor_does_not_authorize_rotation():
    control, p = controller()
    control.transition('hold')
    normals = control.open_normals.copy()
    normals[0] = [-np.cos(np.pi/6), 0, np.sin(np.pi/6)]
    for _ in range(20):
        control.update(p, [.5]*4, normals)
    assert control.phase == 'hold'
    for _ in range(7):
        control.update(p, [.5]*4, control.open_normals)
    assert control.phase == 'rotate'


def test_rotation_pauses_for_contact_recovery_and_counts_loss_duration():
    control, p = controller()
    control.anchor = p.copy()
    control.anchor_normals = control.open_normals.copy()
    control.radial_offset = np.zeros(4)
    control.transition('rotate')
    for _ in range(14):
        control.update(p, [.5]*4)
    previous = control.rotation_elapsed
    for _ in range(3):
        control.update(p, [.5, .5, .5, 0])
    assert control.phase == 'rotate'  # grace is consecutive loss, not age of phase
    assert control.rotation_elapsed == previous
    control.update(p, [.5]*4)
    assert control.rotation_elapsed > previous and control.lost_seconds == 0
    for _ in range(14):
        control.update(p, [.5, .5, .5, 0])
        if control.phase != 'rotate':
            break
    assert control.phase == 'release' and control.reason == 'lost_contact'


def test_normal_constraint_does_not_penalize_free_axial_rotation():
    q = np.zeros(16)
    positions = np.zeros((4, 3))
    normals = np.tile([1., 0., 0.], (4, 1))
    j = np.zeros((4, 6, 16))
    j[0, 0, 0] = .1
    desired = positions.copy(); desired[0, 0] = .001
    without_spin = cartesian_joint_target(q, positions, normals, j, desired, normals, -np.ones(16), np.ones(16))
    j[0, 3, 0] = 10.  # rotation about sensing normal cannot change that normal
    with_spin = cartesian_joint_target(q, positions, normals, j, desired, normals, -np.ones(16), np.ones(16))
    np.testing.assert_allclose(with_spin, without_spin, atol=1e-12)
