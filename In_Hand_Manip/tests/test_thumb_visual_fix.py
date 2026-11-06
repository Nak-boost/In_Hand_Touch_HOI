"""Check assembly continuity separately from collision nonpenetration."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hora.utils.mount_geometry import origin, forward_kinematics, POSTURES
from hora.utils.hand_collision import collision_spec
from hora.utils.asset_geometry import verify_geometry_manifest
from scripts.build_thumb_visual_fix import semantics, build

A = Path(__file__).resolve().parents[1]/'assets/allegro'
OLD = A/'allegro_digit_compact_inward90.urdf'
FIX = A/'allegro_digit_compact_inward90_thumb_visual_fixed.urdf'


def test_only_thumb_visual_origin_changes_and_all_physics_is_preserved():
    old, fixed = [ET.parse(p).getroot() for p in [OLD, FIX]]
    old.find("link[@name='link_12.0']/visual/origin").set('rpy', '0 0 0')
    assert semantics(old) == semantics(fixed)
    assert collision_spec(OLD) == collision_spec(FIX)
    verify_geometry_manifest(FIX)
    for q in POSTURES.values():
        _, left = forward_kinematics(OLD, q)
        _, right = forward_kinematics(FIX, q)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])


def test_thumb_visual_reaches_child_joint_instead_of_leaving_a_31mm_gap():
    trimesh = pytest.importorskip('trimesh')
    meshes = [trimesh.load(A/'meshes/allegro'/n, force='mesh') for n in
              ['link_12.0_right.obj', 'link_13.0.obj']]
    intervals = []
    for path in [OLD, FIX]:
        root = ET.parse(path).getroot()
        parent, child = [m.copy() for m in meshes]
        parent.apply_transform(origin(root.find("link[@name='link_12.0']/visual/origin")))
        child.apply_transform(origin(root.find("joint[@name='joint_13.0']/origin")))
        intervals.append(child.bounds[0, 2] - parent.bounds[1, 2])
    assert intervals[0] > .031  # Confirm the historical defect is actually detected.
    assert intervals[1] < 0     # Mesh extents now reach the joint interface.
    internal = ET.parse(A/'allegro_internal.urdf').getroot()
    np.testing.assert_allclose(origin(ET.parse(FIX).getroot().find("link[@name='link_12.0']/visual/origin")),
                               origin(internal.find("link[@name='link_12.0']/visual/origin")))


def test_builder_refuses_to_overwrite_historical_asset():
    with pytest.raises(FileExistsError):
        build(OLD, OLD)


def test_builder_rejects_85_shape_physical_error_as_visual_only(tmp_path):
    # Mock-free guard exercised in the real asset directory, before any write.
    destination = A/'unused_thumb_fix_test.urdf'
    assert not destination.exists()
    with pytest.raises(ValueError, match='physical revalidation'):
        build(A/'allegro_digit_official_inward90.urdf', destination)
    assert not destination.exists()
