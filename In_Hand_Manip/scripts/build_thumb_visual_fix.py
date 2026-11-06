"""Create a separate compact-hand asset correcting only the thumb-root visual.

The HORA training asset allegro_internal.urdf omits the erroneous Rx(pi)
present in allegro.urdf. Mesh-collision variants need a physical correction
and must not be processed by this visual-only builder.
"""
import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def semantics(element):
    return (element.tag, dict(element.attrib), (element.text or '').strip(),
            [semantics(c) for c in element])


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError('Use a new asset path; preserve historical assets')
    if output.parent != source.parent:
        raise ValueError('Keep relative mesh/manifest paths in the same asset directory')
    from hora.utils.asset_geometry import verify_geometry_manifest
    verify_geometry_manifest(source)
    tree = ET.parse(source)
    robot = tree.getroot()
    before = copy.deepcopy(robot)
    link = robot.find("link[@name='link_12.0']")
    collisions = link.findall('collision')
    if len(collisions) != 1 or collisions[0].find('geometry/box') is None:
        raise ValueError('Not a visual-only case: thumb mesh collisions need separate physical revalidation')
    visual = link.find('visual')
    pose = visual.find('origin')
    if pose is None or pose.get('rpy') != '3.1415926518 0 0':
        raise ValueError('Unexpected source thumb visual transform')
    pose.set('rpy', '0 0 0')
    # Full semantic comparison: nothing except this visual origin may change.
    expected = before.find("link[@name='link_12.0']/visual/origin")
    expected.set('rpy', '0 0 0')
    assert semantics(before) == semantics(robot)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    verify_geometry_manifest(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'assets/allegro/allegro_digit_compact_inward90.urdf')
    parser.add_argument('--output', type=Path, default=ROOT/'assets/allegro/allegro_digit_compact_inward90_thumb_visual_fixed.urdf')
    args = parser.parse_args()
    print(build(args.source, args.output))
