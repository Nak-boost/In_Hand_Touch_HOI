from pathlib import Path
import xml.etree.ElementTree as E
import copy,json,hashlib
import trimesh
root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip/assets/allegro');out=Path(__file__).parent
r=E.parse(root/'allegro_digit_official.urdf').getroot();p=root/'collision_baselines'
for l in r.findall('link'):
 if l.get('name').startswith('digit_'):continue
 vis=l.find('visual');path=root/vis.find('geometry/mesh').get('filename');h=trimesh.load(path,force='mesh').convex_hull
 file=p/('envelope_'+l.get('name')+'.obj');h.export(file)
 for c in list(l.findall('collision')):l.remove(c)
 c=E.SubElement(l,'collision')
 if vis.find('origin') is not None:c.append(copy.deepcopy(vis.find('origin')))
 E.SubElement(E.SubElement(c,'geometry'),'mesh',filename=str(file.relative_to(root)))
r.attrib.pop('geometry_manifest',None);r.attrib.pop('geometry_manifest_sha256',None)
E.ElementTree(r).write(root/'allegro_digit_cad_envelope.urdf')
