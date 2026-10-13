"""Quantify collision approximation and gel/CAD agreement in millimetres."""
import json
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/allegro'
PACKAGE=ASSETS/'official_digit'


def summary(x):
    return {'mean_mm':float(np.mean(x)*1000),'p95_mm':float(np.percentile(x,95)*1000),'max_mm':float(np.max(x)*1000)}


def main():
    np.random.seed(42)
    manifest=json.loads((PACKAGE/'mesh_manifest.json').read_text())
    report={'manifest_sha256':hashlib.sha256((PACKAGE/'mesh_manifest.json').read_bytes()).hexdigest(),
            'method':'Seed 42, vertices plus 3000 area-uniform samples; distances to triangle surfaces. Sampled, not an analytic Hausdorff bound.', 'parts':{}}
    for name,paths in manifest['collision'].items():
        if name.startswith('digit_'):
            source=PACKAGE/'visual'/(name+'.obj')
        else:source=ASSETS/manifest['hand_meshes'][name]['source']
        original=trimesh.load(source,force='mesh')
        if name=='digit_housing':
            # Evaluate the declared closed exterior envelope, not internal PCB geometry.
            components=original.split(only_watertight=False)
            original=trimesh.util.concatenate([m for m in components if len(m.faces)>2500]).convex_hull
        collision=trimesh.util.concatenate([trimesh.load(ASSETS/p,force='mesh') for p in paths])
        points=np.vstack([original.vertices,original.sample(3000)])
        d=trimesh.proximity.closest_point(collision,points)[1]
        # Hull/hull internal seams are not part of the external surface. Reverse
        # comparison uses hull vertices, and is reported separately, without
        # claiming every union-internal face is externally exposed.
        reverse=trimesh.proximity.closest_point(original,collision.vertices)[1]
        report['parts'][name]={'hulls':len(paths),'surface_to_collision':summary(d),'hull_vertices_to_source':summary(reverse)}
    full=trimesh.load(PACKAGE/'source/tacto_digit.stl',force='mesh')
    nf=PACKAGE/'source/neuralfeels_digit.obj'
    if nf.exists():
        reference=trimesh.load(nf,force='mesh')
        report['neuralfeels_vs_tacto_vertices']=summary(cKDTree(full.vertices).query(reference.vertices)[0])
    cad=ROOT.parent/'research/digit_mount_review/sources/cad_meshes/digit_gel_0.obj'
    if cad.exists():
        reference=trimesh.load(cad,force='mesh')
        gel=trimesh.load(PACKAGE/'visual/digit_gel.obj',force='mesh')
        reference.vertices=reference.vertices[:,[2,0,1]]
        shift=[gel.bounds[0,0]-reference.bounds[0,0],0,.022]
        reference.vertices+=shift
        report['gel_cad_crosscheck']={'axis_permutation':[2,0,1],'translation_m':shift,
              'registration_note':'Derived coordinate-frame registration, not a published mounting transform',
              'cad_vertices_to_tacto_surface':summary(trimesh.proximity.closest_point(gel,reference.vertices)[1])}
    output=ROOT.parent/'research/digit_mount_review/mesh_fidelity.json'
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
