"""Source geometry, mounting, collision and stale-mesh regression checks."""
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hora.utils.asset_geometry import verify_geometry_manifest
from hora.utils.hand_collision import collision_spec

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/allegro'
PACKAGE = ASSETS / 'official_digit'
ASSET = ASSETS / 'allegro_digit_official.urdf'


def test_published_mounts_and_original_actuated_kinematics():
    robot = ET.parse(ASSET).getroot()
    original = ET.parse(ASSETS / 'allegro.urdf').getroot()
    published = ET.parse(PACKAGE / 'source/tacto_left_digit.urdf').getroot()
    assert len(robot.findall('link')) == 25
    active = [j for j in robot.findall('joint') if j.get('type') == 'revolute']
    assert len(active) == 16
    def semantics(element):
        return element.tag, element.attrib, (element.text or '').strip(), [semantics(c) for c in element]
    for joint in active:
        reference = original.find("joint[@name='%s']" % joint.get('name'))
        assert semantics(joint) == semantics(reference)
    for finger, index in [('index', 3), ('middle', 7), ('ring', 11), ('thumb', 15)]:
        mount = robot.find("joint[@name='digit_%s_housing_mount']/origin" % finger)
        reference = published.find("joint[@name='joint_%d.0_tip']/origin" % index)
        for key in ['xyz', 'rpy']:
            assert np.allclose(np.fromstring(mount.get(key), sep=' '),
                               np.fromstring(reference.get(key), sep=' '))
        assert np.allclose(np.fromstring(mount.get('rpy'), sep=' '), 0)
    assert collision_spec(ASSET) == collision_spec(ASSETS / 'allegro_digit.urdf')


def test_all_referenced_meshes_are_hash_protected_and_tampering_is_rejected(tmp_path):
    verify_geometry_manifest(ASSET)
    robot = ET.parse(ASSET).getroot()
    manifest = json.loads((PACKAGE / 'mesh_manifest.json').read_text())
    protected = {PACKAGE / p for p in manifest['files_sha256']}
    protected.update(ASSETS / p for p in manifest['hand_visual_sha256'])
    assert all(ASSETS / mesh.get('filename') in protected for mesh in robot.iter('mesh'))
    shutil.copytree(ASSETS, tmp_path / 'allegro')
    copy = tmp_path / 'allegro'
    with (copy / 'official_digit/visual/digit_gel.obj').open('a') as out:
        out.write('\n# Changed external mesh\n')
    with pytest.raises(ValueError, match='Geometry mesh changed'):
        verify_geometry_manifest(copy / ASSET.name)


def test_visual_partition_preserves_published_surface_and_gel_coordinate_frame():
    trimesh = pytest.importorskip('trimesh')
    source = trimesh.load(PACKAGE / 'source/tacto_digit.stl', force='mesh')
    gel = trimesh.load(PACKAGE / 'visual/digit_gel.obj', force='mesh')
    housing = trimesh.load(PACKAGE / 'visual/digit_housing.obj', force='mesh')
    # Map exported vertices back to the published vertices within OBJ precision,
    # then compare triangles with multiplicities, so no surface is omitted.
    from scipy.spatial import cKDTree
    tree = cKDTree(source.vertices)
    def triangles(mesh):
        distance, vertex_ids = tree.query(mesh.vertices)
        assert distance.max() < 1e-8
        return sorted(tuple(sorted(tri)) for tri in vertex_ids[mesh.faces])
    assert triangles(source) == triangles(trimesh.util.concatenate([gel, housing]))
    assert .004 < gel.extents[0] < .0042  # Curved solid, not a planar patch.
    assert len(np.unique(np.round(gel.vertices[:, 0], 6))) > 20
    robot = ET.parse(ASSET).getroot()
    for finger in ['index', 'middle', 'ring', 'thumb']:
        mount = robot.find("joint[@name='digit_%s_gel_mount']/origin" % finger)
        visual = robot.find("link[@name='digit_%s_gel']/visual/origin" % finger)
        assert np.allclose(np.fromstring(mount.get('xyz'), sep=' ') +
                           np.fromstring(visual.get('xyz'), sep=' '), 0, atol=1e-12)


def test_collision_hulls_meet_physx_limits_and_gel_is_exposed():
    trimesh = pytest.importorskip('trimesh')
    manifest = json.loads((PACKAGE / 'mesh_manifest.json').read_text())
    for files in manifest['collision'].values():
        for file in files:
            mesh = trimesh.load(ASSETS / file, force='mesh')
            assert mesh.is_convex and mesh.is_watertight
            assert len(mesh.vertices) <= 64
            assert mesh.volume > 0
    gel = trimesh.load(ASSETS / manifest['collision']['digit_gel'][0], force='mesh')
    housing = trimesh.load(ASSETS / manifest['collision']['digit_housing'][0], force='mesh')
    # The front gel must remain outside the rigid housing collision envelope.
    assert gel.bounds[1, 0] - housing.bounds[1, 0] > .002
