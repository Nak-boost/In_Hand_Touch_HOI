#!/usr/bin/env python3
"""Inspect hand assembly without weights, cached grasps, training or Isaac Gym.

Exports a JSON audit and actual URDF visual/collision views for three poses.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hora.utils.mount_geometry import POSTURES, FINGERS, forward_kinematics, origin, boxes, inspect


def render(asset, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import trimesh
    from hora.utils.mount_geometry import rotation

    fig = plt.figure(figsize=(15, 10))
    world = np.eye(4)
    world[:3, :3] = rotation([0, 1, 0], -np.pi/2) @ rotation([1, 0, 0], np.pi/2)
    scene_meshes = {}
    for column, (name, q) in enumerate(POSTURES.items()):
        robot, transforms = forward_kinematics(asset, q)
        for row in range(2):
            ax = fig.add_subplot(2, 3, row*3+column+1, projection='3d')
            triangles, colors = [], []
            if row == 0:
                for link in robot.findall('link'):
                    for visual in link.findall('visual'):
                        box, mesh = visual.find('geometry/box'), visual.find('geometry/mesh')
                        if box is not None:
                            geometry = trimesh.creation.box(extents=np.fromstring(box.get('size'), sep=' '))
                        elif mesh is not None:
                            file = str(asset.parent/mesh.get('filename'))
                            if file not in scene_meshes:
                                scene_meshes[file] = trimesh.load(file, force='mesh')
                            geometry = scene_meshes[file].copy()
                            geometry.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
                        else:
                            raise ValueError('Unsupported visual geometry')
                        geometry.apply_transform(world @ transforms[link.get('name')] @ origin(visual.find('origin')))
                        color = visual.find('material/color')
                        rgba = np.fromstring(color.get('rgba'), sep=' ') if color is not None else [.28,.28,.3,1]
                        light=np.array([1.,-1.,2.]);light/=np.linalg.norm(light)
                        intensity=.55+.45*np.maximum(0,geometry.face_normals @ light)
                        triangles.extend(geometry.triangles)
                        colors.extend(np.column_stack([intensity[:,None]*rgba[:3],np.full(len(intensity),rgba[3])]))
            else:
                for link in robot.findall('link'):
                    for collision in link.findall('collision'):
                        box = collision.find('geometry/box')
                        mesh = collision.find('geometry/mesh')
                        if box is not None:
                            geometry = trimesh.creation.box(extents=np.fromstring(box.get('size'), sep=' '))
                        elif mesh is not None:
                            file = str(asset.parent/mesh.get('filename'))
                            if file not in scene_meshes:
                                scene_meshes[file] = trimesh.load(file, force='mesh')
                            geometry = scene_meshes[file].copy()
                            geometry.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
                        else:
                            raise ValueError('Unsupported collision geometry')
                        pose = transforms[link.get('name')] @ origin(collision.find('origin'))
                        geometry.apply_transform(world @ pose)
                        color = '#18bace' if link.get('name').endswith('_gel') else '#555966'
                        from matplotlib.colors import to_rgba
                        triangles.extend(geometry.triangles)
                        colors.extend([to_rgba(color,.22)]*len(geometry.faces))
            ax.add_collection3d(Poly3DCollection(triangles,facecolor=colors,
                              edgecolor='none' if row==0 else '#404040',linewidth=.3))
            for finger, index in FINGERS.items():
                for body, color in [('link_%d.0'%index, '#e65328'), ('digit_'+finger+'_gel', '#00a3c7')]:
                    pose = world @ transforms[body]
                    # Orange: distal palmar +X. Cyan: gel outward +X.
                    ax.quiver(*pose[:3, 3], *pose[:3, 0], length=.024, color=color, linewidth=1.5)
            ax.set(xlim=(-.20,.04), ylim=(-.12,.12), zlim=(-.035,.205),
                   xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)')
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=25, azim=45)
            ax.set_title(name.replace('_',' ') + (' / visual' if row==0 else ' / collision geometry'))
    fig.suptitle(asset.name+'\nOrange: distal palmar normal; cyan: gel normal. Diagnostic poses, no grasp/policy.', fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=Path('assets/allegro/allegro_digit_official.urdf'), help='Absolute path or relative to In_Hand_Manip')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent/'research/digit_mount_review')
    parser.add_argument('--no-plot', action='store_true')
    args=parser.parse_args()
    asset=(ROOT/args.asset.expanduser()).resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    import xml.etree.ElementTree as ET
    if ET.parse(asset).getroot().get('geometry_manifest'):
        from hora.utils.asset_geometry import verify_geometry_manifest
        from hora.utils.mesh_inspection import inspect_mesh_asset
        verify_geometry_manifest(asset)
        report=inspect_mesh_asset(asset)
    else:
        report=inspect(asset)
    report['asset_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest()
    report.setdefault('method','URDF FK, box separating-axis tests; not a dynamics or hardware validation')
    stem=asset.stem
    path=args.output_dir/(stem+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    if not args.no_plot:
        render(asset,args.output_dir/(stem+'.png'))
    for pose, result in report['postures'].items():
        print(pose, 'enabled overlaps:',len(result['enabled_collision_overlaps']))
        for overlap in result['enabled_collision_overlaps']:
            print(' ',overlap)
    print('Saved:',path)


if __name__=='__main__':
    main()
