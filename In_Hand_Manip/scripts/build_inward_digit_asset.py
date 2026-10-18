"""Rotate the two outer DIGIT assemblies; preserve geometry and all dynamics."""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).resolve().parents[1]


def build(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg['mount_profile'] != 'inward90_right':
        raise ValueError('Expected right-hand inward90 profile')
    if cfg['yaw_degrees'] != dict(index=-90, middle=0, ring=90, thumb=0):
        raise ValueError('This profile requires index -90, ring +90, middle/thumb 0 degrees')
    from hora.utils.asset_geometry import verify_geometry_manifest
    from hora.utils.hand_collision import collision_spec
    outputs = []
    for pair in cfg['assets']:
        source, output = ROOT/pair['source'], ROOT/pair['output']
        if source.resolve() == output.resolve():
            raise ValueError('Keep the straight source asset for historical experiments')
        verify_geometry_manifest(source)
        tree = ET.parse(source)
        robot = tree.getroot()
        robot.set('name', robot.get('name')+'_inward90')
        robot.set('mount_profile', cfg['mount_profile'])
        for finger, angle in cfg['yaw_degrees'].items():
            mount = robot.find("joint[@name='digit_%s_housing_mount']/origin" % finger)
            if mount is None or any(float(x) != 0 for x in mount.get('rpy', '0 0 0').split()):
                raise ValueError('Expected straight source DIGIT mount: '+finger)
            if angle:
                mount.set('rpy', '0 0 %.16g' % math.radians(angle))
        tree.write(output, encoding='utf-8', xml_declaration=True)
        verify_geometry_manifest(output)
        assert collision_spec(output) == collision_spec(source)
        outputs.append(output)
        print(output, 'collision shapes:', len(robot.findall('.//collision')))
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/digit_mount_inward.yaml')
    build(parser.parse_args().config)
