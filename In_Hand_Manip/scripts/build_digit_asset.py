"""Build a parameterized rigid DIGIT approximation; not a calibrated hardware model."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def box_link(name, size, mass, color):
    link = ET.Element('link', name=name)
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', value=str(mass))
    x, y, z = size
    ET.SubElement(inertial, 'inertia', ixx=str(mass*(y*y+z*z)/12),
                  iyy=str(mass*(x*x+z*z)/12), izz=str(mass*(x*x+y*y)/12),
                  ixy='0', ixz='0', iyz='0')
    for tag in ['visual', 'collision']:
        el = ET.SubElement(link, tag)
        ET.SubElement(ET.SubElement(el, 'geometry'), 'box', size=' '.join(map(str, size)))
        if tag == 'visual':
            ET.SubElement(ET.SubElement(el, 'material', name=name), 'color', rgba=color)
    return link


def build(config_path, output_path=None):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    tree = ET.parse(ROOT / 'assets/allegro/allegro.urdf')
    robot = tree.getroot()
    robot.set('name', 'allegro_right_digit_proxy')
    # Resolve upstream's mesh paths relative to the generated URDF directory.
    for mesh in robot.iter('mesh'):
        mesh.set('filename', mesh.get('filename').replace('allegro/meshes/', 'meshes/'))
    # Restore the physical palm; upstream used a 1 cm proxy for this benchmark.
    palm = robot.find("link[@name='base_link']/collision/geometry/box")
    palm.set('size', '0.0408 0.1130 0.095')
    names = ['index', 'middle', 'ring', 'thumb']
    for finger, index in zip(names, [3, 7, 11, 15]):
        tip = 'link_%s.0_tip' % index
        robot.remove(robot.find("link[@name='%s']" % tip))
        robot.remove(robot.find("joint[@name='joint_%s.0_tip']" % index))
        distal = robot.find("link[@name='link_%s.0']/collision" % index)
        geom = distal.find('geometry')
        geom.clear()
        ET.SubElement(geom, 'box', size='0.018 0.018 %s' % (0.0423 if index == 15 else 0.0267))
        if cfg.get('show_distal_connector', False):
            # The old collision proxy extends 11 mm beyond the distal mesh.
            # Make that existing rigid connector visible, without adding a
            # collision shape, joint, mass or hidden support surface.
            distal_end = 0.0423 if index == 15 else 0.0267
            mesh_end = 0.0313 if index == 15 else 0.0157
            visual = ET.SubElement(robot.find("link[@name='link_%s.0']" % index), 'visual')
            ET.SubElement(visual, 'origin', xyz='0 0 %s' % ((mesh_end+distal_end)/2), rpy='0 0 0')
            ET.SubElement(ET.SubElement(visual, 'geometry'), 'box', size='0.018 0.018 %s' % (distal_end-mesh_end))
            ET.SubElement(ET.SubElement(visual, 'material', name='distal_connector'), 'color', rgba='0.35 0.35 0.35 1')
        housing = 'digit_%s_housing' % finger
        gel = 'digit_%s_gel' % finger
        robot.append(box_link(housing, cfg['housing_size'], cfg['housing_mass'], '0.15 0.15 0.15 1'))
        robot.append(box_link(gel, cfg['gel_size'], cfg['gel_mass'], '0.2 0.75 0.9 1'))
        mount = cfg['mounts'][finger]
        joint = ET.SubElement(robot, 'joint', name=housing+'_mount', type='fixed')
        ET.SubElement(joint, 'parent', link='link_%s.0' % index)
        ET.SubElement(joint, 'child', link=housing)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, mount['xyz'])), rpy=' '.join(map(str, mount['rpy'])))
        joint = ET.SubElement(robot, 'joint', name=gel+'_mount', type='fixed')
        ET.SubElement(joint, 'parent', link=housing)
        ET.SubElement(joint, 'child', link=gel)
        # Gel's outward normal is housing +X. Keep a small gap to avoid internal contacts.
        offset = (cfg['housing_size'][0]+cfg['gel_size'][0])/2 + 0.0001
        ET.SubElement(joint, 'origin', xyz='%s 0 0' % offset, rpy='0 0 0')
    output = Path(output_path) if output_path else ROOT / cfg.get('output_asset', 'assets/allegro/allegro_digit.urdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output), encoding='utf-8', xml_declaration=True)
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'configs/digit_mount.yaml'))
    parser.add_argument('--output', type=Path, help='Optional output path; otherwise use output_asset from config')
    args = parser.parse_args()
    build(args.config, args.output)
