"""Collision simplification must preserve assembly, sensing and exclusion rules."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from hora.utils.asset_geometry import verify_geometry_manifest
from hora.utils.hand_collision import collision_spec
from hora.utils.mesh_inspection import inspect_mesh_asset
A=Path(__file__).resolve().parents[1]/'assets/allegro'

def semantics(x):
    return x.tag,x.attrib,(x.text or '').strip(),[semantics(c) for c in x]

def test_compact_only_changes_hand_collision_geometry():
    original=ET.parse(A/'allegro_digit_official.urdf').getroot()
    compact=ET.parse(A/'allegro_digit_compact.urdf').getroot()
    assert len(compact.findall('.//collision'))==27
    assert len(compact.findall("link[@name='base_link']/collision"))==3
    for tree in [original,compact]:
        tree.attrib.clear()
        for link in tree.findall('link'):
            if not link.get('name').startswith('digit_'):
                for c in list(link.findall('collision')):link.remove(c)
    assert semantics(original)==semantics(compact)
    assert collision_spec(A/'allegro_digit_compact.urdf')==collision_spec(A/'allegro_digit_official.urdf')
    verify_geometry_manifest(A/'allegro_digit_compact.urdf')

def test_vendor_baseline_uses_vendor_collision_elements():
    reference=ET.parse(A/'official_digit/source/vendor_right.urdf').getroot()
    robot=ET.parse(A/'allegro_digit_vendor.urdf').getroot()
    for link in robot.findall('link'):
        name=link.get('name')
        if name.startswith('digit_'):continue
        source=reference.find("link[@name='%s']"%('palm_link' if name=='base_link' else name))
        assert list(map(semantics,link.findall('collision')))==list(map(semantics,source.findall('collision')))

def test_positive_distance_query_detects_known_overlap_and_compact_clears_poses():
    pytest.importorskip('pybullet')
    vendor=inspect_mesh_asset(A/'allegro_digit_vendor.urdf')
    assert any(x['bodies']==['base_link','link_13.0'] and x['penetration_mm']>4 for x in vendor['postures']['open']['enabled_collision_overlaps'])
    compact=inspect_mesh_asset(A/'allegro_digit_compact.urdf')
    assert all(not x['enabled_collision_overlaps'] for x in compact['postures'].values())

def test_compact_hulls_fit_physx_vertex_policy():
    trimesh=pytest.importorskip('trimesh')
    root=ET.parse(A/'allegro_digit_compact.urdf').getroot()
    for mesh in root.findall('.//collision/geometry/mesh'):
        hull=trimesh.load(A/mesh.get('filename'),force='mesh')
        assert hull.is_convex and hull.is_watertight and len(hull.vertices)<=64
