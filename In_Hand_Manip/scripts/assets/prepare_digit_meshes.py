"""Extract the official DIGIT surface and prepare explicit convex collision pieces.

Offline generation only: project Python + trimesh/scipy/pybullet. Runtime
Isaac Gym uses the checked-in meshes and does not need these build packages.
No dimensions are rescaled to paper nominal values; all source STL units are m.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/allegro'
OUT=ASSETS/'official_digit'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decompose(mesh,name,resolution):
    import pybullet as pb
    work=OUT/'build_cache';work.mkdir(exist_ok=True)
    source=work/(name+'_mm.obj');dest=work/(name+'_vhacd.obj')
    scaled=mesh.copy();scaled.apply_scale(1000);scaled.export(source)
    pb.vhacd(str(source),str(dest),str(work/(name+'.log')),
             resolution=resolution,concavity=.001,maxNumVerticesPerCH=64,
             minVolumePerCH=.00001,convexhullDownsampling=4,planeDownsampling=4)
    # Read each VHACD OBJ object independently; shared coordinates must not
    # merge disconnected convex pieces back into one concave mesh.
    vertices=[];faces=[];parts=[]
    def finish():
        if faces:
            m=trimesh.Trimesh(vertices=np.asarray(vertices)*.001,faces=np.asarray(faces),process=True)
            m.remove_unreferenced_vertices();parts.append(m.convex_hull)
    for line in dest.read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
        elif fields[0]=='f':faces.append([int(x.split('/')[0])-1 for x in fields[1:4]])
        elif fields[0]=='o':finish();faces=[]
    finish()
    if not parts:raise ValueError('No collision hulls: '+name)
    for old in (OUT/'collision').glob(name+'_*.obj'):
        old.unlink()
    paths=[]
    for i,part in enumerate(parts):
        if len(part.vertices)>64:raise ValueError('PhysX convex vertex limit exceeded')
        path=OUT/'collision'/('%s_%03d.obj'%(name,i));part.export(path);paths.append(str(path.relative_to(ASSETS)))
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution',type=int,default=400000)
    args=parser.parse_args()
    for folder in ['visual','collision']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    full=trimesh.load(OUT/'source/tacto_digit.stl',force='mesh')
    # Do not fill tiny holes while partitioning the published visual surface.
    components=full.split(only_watertight=False,repair=False)
    # Published assembly has one connected forward elastomer solid. This
    # selector is tied to the pinned source hash and cross-checked with gel STEP.
    candidate=[c for c in components if c.bounds[0,0]>.016 and c.extents[1]>.018 and c.extents[2]>.024]
    if len(candidate)!=1 or len(candidate[0].faces)!=2650:
        raise ValueError('Official mesh topology changed; re-identify gel against CAD')
    gel=candidate[0]
    housing=trimesh.util.concatenate([c for c in components if c is not gel])
    housing.export(OUT/'visual/digit_housing.obj');gel.export(OUT/'visual/digit_gel.obj')
    # Ignore electronics enclosed by the case for collision cooking. Keep the
    # three outer casing components. The visual mesh retains every component.
    casing=[c for c in components if c is not gel and len(c.faces)>2500]
    if len(casing)!=3:raise ValueError('Expected three exterior casing pieces')
    # One outer hull fills the case interior/ports but never includes the gel.
    # A closed housing envelope avoids hundreds of invisible internal contacts.
    casing_hull=trimesh.util.concatenate(casing).convex_hull
    geometry={'digit_gel':gel,'digit_housing':casing_hull}
    hand_meshes={}
    for name in ['base_link','link_0.0','link_1.0','link_2.0','link_3.0','link_12.0_right','link_13.0','link_14.0','link_15.0']:
        path=ASSETS/'meshes/allegro'/(name+'.obj')
        # Palm uses the manufacturer STL directly; the inherited OBJ agrees
        # with it to below one micrometre, but the original is more precise.
        if name=='base_link':path=OUT/'source/vendor_base.stl'
        hand_meshes[name]={'source':str(path.relative_to(ASSETS)),'sha256':digest(path)}
        geometry[name]=trimesh.load(path,force='mesh')
    manifest={'source_stl_sha256':digest(OUT/'source/tacto_digit.stl'),
              'units':'metres','gel_bounds_m':gel.bounds.tolist(),
              'full_digit_bounds_m':full.bounds.tolist(),
              'gel_faces':len(gel.faces),'housing_faces':len(housing.faces),
              'method':{'algorithm':'PyBullet V-HACD','resolution':args.resolution,
                        'concavity':.001,'max_vertices_per_hull':64,
                        'housing_note':'Convex outer casing envelope; internal electronics and ports are not separately collidable'},
              'hand_meshes':hand_meshes,'collision':{}}
    for name,mesh in geometry.items():
        print('PREPARE',name,flush=True)
        manifest['collision'][name]=decompose(mesh,name,args.resolution)
    manifest['files_sha256']={str(p.relative_to(OUT)):digest(p) for folder in ['visual','collision'] for p in sorted((OUT/folder).glob('*.obj'))}
    manifest['hand_visual_sha256']={str(p.relative_to(ASSETS)):digest(p) for p in sorted((ASSETS/'meshes/allegro').glob('*.obj')) if p.stem in geometry}
    (OUT/'mesh_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Wrote',OUT/'mesh_manifest.json',flush=True)


if __name__=='__main__':main()
