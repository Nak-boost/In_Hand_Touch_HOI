"""Assembly invariants independent of the simulator or training outcome."""
from pathlib import Path
import numpy as np

from hora.utils.mount_geometry import inspect, forward_kinematics, POSTURES, box_overlap
from hora.utils.hand_collision import collision_spec

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'assets/allegro/allegro_digit_aligned.urdf'


def test_sensor_normal_and_long_axis_follow_distal_in_all_diagnostic_poses():
    report=inspect(ASSET)
    for pose in report['postures'].values():
        for alignment in pose['alignment'].values():
            assert abs(alignment['normal_deviation_deg']) < 1e-5
            assert abs(alignment['long_axis_deviation_deg']) < 1e-5
            assert abs(alignment['front_plane_offset_mm']) < 1e-8
        # Check the added assembly, without hiding unrelated palm/thumb overlaps.
        assert not [c for c in pose['enabled_collision_overlaps']
                    if any(body.startswith('digit_') for body in c['bodies'])]


def test_sensor_abuts_distal_and_keeps_physical_thickness_and_existing_collision_rules():
    robot, transforms=forward_kinematics(ASSET, POSTURES['open'])
    assert sum(j.get('type')=='revolute' for j in robot.findall('joint')) == 16
    for finger,index in [('index',3),('middle',7),('ring',11),('thumb',15)]:
        parent=transforms['link_%d.0'%index]
        housing=transforms['digit_'+finger+'_housing']
        local=np.linalg.inv(parent) @ housing
        size=np.fromstring(robot.find("link[@name='digit_%s_housing']/collision/geometry/box"%finger).get('size'),sep=' ')
        assert np.isclose(size[0],.016)  # Housing isn't a zero-thickness touch patch.
        distal=robot.find("link[@name='link_%d.0']/collision"%index)
        end=float(distal.find('origin').get('xyz').split()[2])+float(distal.find('geometry/box').get('size').split()[2])/2
        assert np.isclose(local[2,3]-size[2]/2,end)
        assert robot.find("link[@name='link_%d.0_tip']"%index) is None
    legacy=ROOT/'assets/allegro/allegro_digit.urdf'
    assert collision_spec(ASSET)==collision_spec(legacy)
    assert ASSET.read_bytes()!=legacy.read_bytes()  # Old checkpoint hash must differ.


def test_box_overlap_distinguishes_touch_separation_and_rotated_penetration():
    a=('a',np.eye(4),np.ones(3))
    pose=np.eye(4);pose[0,3]=1
    assert box_overlap(a,('b',pose,np.ones(3)))==0
    pose[0,3]=1.1
    assert box_overlap(a,('b',pose,np.ones(3)))==0
    from hora.utils.mount_geometry import rotation
    pose[:3,:3]=rotation([0,0,1],np.pi/4)
    assert box_overlap(a,('b',pose,np.ones(3)))>0
