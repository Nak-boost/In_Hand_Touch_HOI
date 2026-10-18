"""Mount rotation changes the whole sensor, without changing physical assumptions."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from hora.utils.asset_geometry import verify_geometry_manifest
from hora.utils.hand_collision import collision_spec
from hora.utils.mount_geometry import forward_kinematics, POSTURES, FINGERS

ASSETS = Path(__file__).resolve().parents[1]/'assets/allegro'


def semantics(x):
    return x.tag, x.attrib, (x.text or '').strip(), [semantics(c) for c in x]


@pytest.mark.parametrize('variant,shapes', [('compact', 27), ('official', 85)])
def test_only_two_fixed_mounts_change_and_old_cache_identity_is_invalidated(variant, shapes):
    source = ASSETS/('allegro_digit_'+variant+'.urdf')
    target = ASSETS/('allegro_digit_'+variant+'_inward90.urdf')
    a, b = ET.parse(source).getroot(), ET.parse(target).getroot()
    assert len(b.findall('.//collision')) == shapes
    assert len(b.findall('link')) == 25
    assert sum(j.get('type') == 'revolute' for j in b.findall('joint')) == 16
    assert hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest()
    assert collision_spec(source) == collision_spec(target)
    verify_geometry_manifest(target)
    b.attrib = a.attrib.copy()
    for finger, angle in [('index', -np.pi/2), ('ring', np.pi/2)]:
        mount = b.find("joint[@name='digit_%s_housing_mount']/origin" % finger)
        np.testing.assert_allclose(np.fromstring(mount.get('rpy'), sep=' '), [0,0,angle], atol=1e-12)
        mount.set('rpy', a.find("joint[@name='digit_%s_housing_mount']/origin" % finger).get('rpy'))
    # Includes all visual/collision origins, inertias, active joints and limits.
    assert semantics(a) == semantics(b)


def test_gel_normal_and_housing_rotate_together_in_all_postures():
    for q in POSTURES.values():
        _, transforms = forward_kinematics(ASSETS/'allegro_digit_compact_inward90.urdf', q)
        for finger, distal in FINGERS.items():
            parent = transforms['link_%d.0' % distal]
            gel = transforms['digit_'+finger+'_gel']
            housing = transforms['digit_'+finger+'_housing']
            expected = {'index': [0,-1,0], 'ring': [0,1,0]}.get(finger, [1,0,0])
            np.testing.assert_allclose(parent[:3,:3].T @ gel[:3,0], expected, atol=1e-12)
            np.testing.assert_allclose(housing[:3,:3], gel[:3,:3], atol=1e-12)
            np.testing.assert_allclose(parent[:3,2], housing[:3,2], atol=1e-12)


def test_task_configs_use_new_assets_and_separate_cache_names_but_keep_historical_resumes():
    from hydra import compose, initialize_config_dir
    configs = ASSETS.parents[1]/'configs'
    with initialize_config_dir(version_base='1.3', config_dir=str(configs)):
        for path in sorted(configs.glob('manip*.yaml')):
            cfg = compose(config_name=path.stem)
            asset = cfg.task.env.asset.handAsset
            cache = cfg.task.env.grasp.cache
            if '_resume20261016' in path.stem:
                assert asset.endswith('allegro_digit_compact.urdf')
                assert 'inward90' not in cache
            else:
                assert asset.endswith('_inward90.urdf'), path.stem
                assert 'inward90' in cache and cache.endswith('.pt'), path.stem
