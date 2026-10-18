"""Build isolated collision alternatives; retain sourced DIGIT assembly and dynamics.

vendor: original vendor hand boxes; palm23: vendor fingers and existing palm mesh.
Split candidates use convex hulls of CAD clipped into spatial regions, only on palm.
No runtime configuration or existing asset is overwritten.
"""
import copy, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'assets/allegro'
OUT=ASSETS/'collision_baselines'

def clip_triangles(triangles, axis, value, sign):
    result=[]
    for tri in triangles:
        poly=[]
        for a,b in zip(tri,np.roll(tri,-1,axis=0)):
            da=sign*(a[axis]-value);db=sign*(b[axis]-value)
            if da>=-1e-12: poly.append(a)
            if (da>0 and db<0) or (da<0 and db>0):poly.append(a+(b-a)*da/(da-db))
        for i in range(1,len(poly)-1):result.append([poly[0],poly[i],poly[i+1]])
    return np.asarray(result)

def semantics(x):
    return x.tag,x.attrib,(x.text or '').strip(),[semantics(c) for c in x]

def build():
    OUT.mkdir(exist_ok=True)
    current=ET.parse(ASSETS/'allegro_digit_official.urdf').getroot()
    vendor=ET.parse(ASSETS/'official_digit/source/vendor_right.urdf').getroot()
    palm=trimesh.load(ASSETS/'official_digit/source/vendor_base.stl',force='mesh')
    original_manifest=json.loads((ASSETS/'official_digit/mesh_manifest.json').read_text())
    results={}
    regions={
      'palm2_y': [[(1,0,1)],[(1,0,-1)]],
      'palm2_z': [[(2,-.055,1)],[(2,-.055,-1)]],
      'palm3': [[(1,.025,-1)],[(1,.025,1),(2,-.055,1)],[(1,.025,1),(2,-.055,-1)]],
      'palm4': [[(1,0,a),(2,-.055,b)] for a in [-1,1] for b in [-1,1]],
    }
    regions['compact'] = regions['palm3']
    for mode in ['vendor','palm23']+list(regions):
        robot=copy.deepcopy(current);robot.set('name','allegro_digit_'+mode)
        files={}; hulls=[]
        if mode in regions:
            for i,planes in enumerate(regions[mode]):
                triangles=palm.triangles.copy()
                for axis,value,sign in planes:triangles=clip_triangles(triangles,axis,value,sign)
                h=trimesh.convex.convex_hull(triangles.reshape(-1,3))
                # Keep full hull geometry. PhysX asset loading is separately checked;
                # candidates above current 64-vertex policy are not production-approved.
                path=OUT/f'{mode}_{i}.obj';h.export(path)
                files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                hulls.append({'file':str(path.relative_to(ASSETS)),'vertices':len(h.vertices)})
        for link in robot.findall('link'):
            name=link.get('name')
            if name.startswith('digit_'):continue
            if name=='base_link' and mode=='palm23':continue
            for c in list(link.findall('collision')):link.remove(c)
            if mode=='compact' and name in ['link_3.0','link_7.0','link_11.0','link_15.0']:
                # The vendor distal boxes produce several millimetres of contact
                # outside the nominal CAD envelope in the recorded pose sweep.
                # One convex hull fixes the footprint without adding shapes.
                visual=link.find('visual');mesh=visual.find('geometry/mesh')
                h=trimesh.load(ASSETS/mesh.get('filename'),force='mesh').convex_hull
                assert len(h.vertices)<=64
                file=OUT/('compact_'+name+'.obj');h.export(file)
                files[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
                c=ET.SubElement(link,'collision')
                if visual.find('origin') is not None:c.append(copy.deepcopy(visual.find('origin')))
                ET.SubElement(ET.SubElement(c,'geometry'),'mesh',filename=str(file.relative_to(ASSETS)))
            elif name=='base_link' and mode in regions:
                for h in hulls:
                    c=ET.SubElement(link,'collision');ET.SubElement(c,'origin',xyz='0 0 0',rpy='0 0 0')
                    ET.SubElement(ET.SubElement(c,'geometry'),'mesh',filename=h['file'])
            else:
                source=vendor.find("link[@name='%s']"%('palm_link' if name=='base_link' else name))
                for c in source.findall('collision'):link.append(copy.deepcopy(c))
        # Existing verifier resolves mesh entries relative to this manifest.
        protected={'../official_digit/'+k:v for k,v in original_manifest['files_sha256'].items()}
        protected.update(files)
        manifest={'files_sha256':protected,'hand_visual_sha256':original_manifest['hand_visual_sha256'],
                  'method':('vendor boxes; palm spatial hulls and four distal single hulls' if mode=='compact'
                            else 'vendor hand collision boxes; only palm replaced for non-vendor variants'),
                  'palm_regions':regions.get(mode),'palm_hulls':hulls,
                  'vendor_source_sha256':hashlib.sha256((ASSETS/'official_digit/source/vendor_right.urdf').read_bytes()).hexdigest()}
        mp=OUT/f'{mode}_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
        robot.set('geometry_manifest',str(mp.relative_to(ASSETS)));robot.set('geometry_manifest_sha256',hashlib.sha256(mp.read_bytes()).hexdigest())
        path=ASSETS/f'allegro_digit_{mode}.urdf';ET.ElementTree(robot).write(path,encoding='utf-8',xml_declaration=True)
        # Check everything except hand collision entries and metadata is preserved.
        reference=copy.deepcopy(current);actual=copy.deepcopy(robot)
        reference.attrib.clear();actual.attrib.clear()
        for tree in [reference,actual]:
            for link in tree.findall('link'):
                if not link.get('name').startswith('digit_'):
                    for c in list(link.findall('collision')):link.remove(c)
        assert semantics(reference)==semantics(actual)
        from hora.utils.hand_collision import collision_spec
        from hora.utils.asset_geometry import verify_geometry_manifest
        assert collision_spec(path)==collision_spec(ASSETS/'allegro_digit_official.urdf')
        verify_geometry_manifest(path)
        results[mode]={'asset':str(path),'shapes':len(robot.findall('.//collision')),'palm_hulls':hulls}
    print(json.dumps(results,indent=2))
    return results
if __name__=='__main__':build()
