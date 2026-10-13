"""Build a sourced DIGIT assembly from published meshes and mounting transforms.

No CAD packages needed to regenerate the URDF. Collision meshes are prepared
once by scripts/assets/prepare_digit_meshes.py and versioned with their hashes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import yaml

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'assets/allegro'
FINGERS={'index':3,'middle':7,'ring':11,'thumb':15}


def mesh_element(link,tag,path,xyz=(0,0,0),color=None):
    element=ET.SubElement(link,tag)
    ET.SubElement(element,'origin',xyz=' '.join('%.12g'%x for x in xyz),rpy='0 0 0')
    ET.SubElement(ET.SubElement(element,'geometry'),'mesh',filename=path)
    if tag=='visual':
        ET.SubElement(ET.SubElement(element,'material',name='digit_surface'),'color',rgba=color)


def set_inertia(link,mesh,mass,frame):
    # Published CAD has no measured material-density/electronics mass map.
    # State the approximation explicitly; do not reuse arbitrary box inertia.
    solid=mesh.convex_hull
    center=solid.center_mass-np.asarray(frame)
    inertia=solid.moment_inertia*(mass/solid.volume)
    element=ET.SubElement(link,'inertial')
    ET.SubElement(element,'origin',xyz=' '.join('%.12g'%x for x in center),rpy='0 0 0')
    ET.SubElement(element,'mass',value='%.12g'%mass)
    ET.SubElement(element,'inertia',**{key:'%.12g'%inertia[i,j] for key,i,j in
                  [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]})


def build(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text())
    if cfg['mount_profile']!='tacto_straight':raise ValueError('Unsupported mount profile')
    manifest=json.loads((ROOT/cfg['mesh_manifest']).read_text())
    package=(ROOT/cfg['mesh_manifest']).parent
    for name,digest in manifest['files_sha256'].items():
        if hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Mesh changed without regeneration: '+name)
    source=ET.parse(ROOT/cfg['source_assembly']).getroot()
    tree=ET.parse(ASSETS/'allegro.urdf');robot=tree.getroot();robot.set('name','allegro_right_digit_official_geometry')
    robot.set('geometry_manifest',str((ROOT/cfg['mesh_manifest']).relative_to(ASSETS)))
    robot.set('geometry_manifest_sha256',hashlib.sha256((ROOT/cfg['mesh_manifest']).read_bytes()).hexdigest())
    # Original HORA joints/inertias/actuator limits stay in place; only geometry
    # and the four attached sensor inertias are replaced in this build.
    for mesh in robot.iter('mesh'):
        mesh.set('filename',mesh.get('filename').replace('allegro/meshes/','meshes/'))
    for finger,index in FINGERS.items():
        robot.remove(robot.find("link[@name='link_%d.0_tip']"%index))
        robot.remove(robot.find("joint[@name='joint_%d.0_tip']"%index))
    for link in robot.findall('link'):
        for collision in list(link.findall('collision')):link.remove(collision)
        visual=link.find('visual')
        mesh=visual.find('geometry/mesh')
        key=Path(mesh.get('filename')).stem
        pose=visual.find('origin')
        for filename in manifest['collision'][key]:
            collision=ET.SubElement(link,'collision')
            if pose is not None:collision.append(copy.deepcopy(pose))
            ET.SubElement(ET.SubElement(collision,'geometry'),'mesh',filename=filename)
    housing=trimesh.load(package/'visual/digit_housing.obj',force='mesh')
    gel=trimesh.load(package/'visual/digit_gel.obj',force='mesh')
    apex=gel.vertices[np.argmax(gel.vertices[:,0])].copy()
    volumes=[housing.convex_hull.volume,gel.convex_hull.volume]
    masses=np.asarray(volumes)/sum(volumes)*cfg['sensor_nominal_mass']
    for finger,index in FINGERS.items():
        body='digit_'+finger+'_housing';sense='digit_'+finger+'_gel'
        for name,kind,mesh,mass,frame,color in [
            (body,'digit_housing',housing,masses[0],np.zeros(3),'0.18 0.18 0.20 1'),
            (sense,'digit_gel',gel,masses[1],apex,'0.2 0.75 0.9 1')]:
            link=ET.SubElement(robot,'link',name=name)
            set_inertia(link,mesh,mass,frame)
            mesh_element(link,'visual','official_digit/visual/'+kind+'.obj',-frame,color)
            for filename in manifest['collision'][kind]:mesh_element(link,'collision',filename,-frame)
        original=source.find("joint[@name='joint_%d.0_tip']"%index)
        # The four distal joint frames have the same local conventions in
        # vendor left/right hands. Root/thumb-base handedness remains RIGHT.
        joint=ET.SubElement(robot,'joint',name=body+'_mount',type='fixed')
        ET.SubElement(joint,'parent',link='link_%d.0'%index);ET.SubElement(joint,'child',link=body)
        joint.append(copy.deepcopy(original.find('origin')))
        joint=ET.SubElement(robot,'joint',name=sense+'_mount',type='fixed')
        ET.SubElement(joint,'parent',link=body);ET.SubElement(joint,'child',link=sense)
        ET.SubElement(joint,'origin',xyz=' '.join('%.12g'%x for x in apex),rpy='0 0 0')
    output=ROOT/cfg['output_asset']
    tree.write(output,encoding='utf-8',xml_declaration=True)
    output.write_bytes(b'\n'.join(line.rstrip() for line in output.read_bytes().splitlines())+b'\n')
    print(output)
    print('Sensor masses [housing, gel] kg:',masses.tolist(),'gel frame apex m:',apex.tolist())
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/digit_mount.yaml')
    build(parser.parse_args().config)
