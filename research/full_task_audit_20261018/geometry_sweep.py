"""Independent static collision audit. CAD convex envelopes give one-sided evidence."""
import pathlib,json,copy,itertools,xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
import pybullet as pb
from hora.utils.hand_collision import collision_spec
from hora.utils.mount_geometry import POSTURES
out=pathlib.Path(__file__).parent;r=json.loads((out/'inventory.json').read_text());asset=pathlib.Path(r[0]['asset']);robot=ET.parse(asset).getroot();spec=collision_spec(asset)
# Absolute mesh paths for the isolated test fixtures; original files unchanged.
for m in robot.findall('.//mesh'):m.set('filename',str(asset.parent/m.get('filename')))
physical=out/'physical_fixture.urdf';ET.ElementTree(robot).write(physical)
ref=copy.deepcopy(robot);hdir=out/'reference_hulls';hdir.mkdir(exist_ok=True)
for link in ref.findall('link'):
 name=link.get('name')
 if name.startswith('digit_'):continue
 vis=link.find('visual');mesh=vis.find('geometry/mesh');h=trimesh.load(mesh.get('filename'),force='mesh').convex_hull
 hp=hdir/(name+'.obj');h.export(hp)
 for c in list(link.findall('collision')):link.remove(c)
 c=ET.SubElement(link,'collision');o=vis.find('origin')
 if o is not None:c.append(copy.deepcopy(o))
 if name=='link_12.0':c.find('origin').set('rpy','0 0 0')
 ET.SubElement(ET.SubElement(c,'geometry'),'mesh',filename=str(hp),scale=mesh.get('scale','1 1 1'))
reference=out/'cad_envelope_fixture.urdf';ET.ElementTree(ref).write(reference)
client=pb.connect(pb.DIRECT);h=pb.loadURDF(str(physical),useFixedBase=True);cad=pb.loadURDF(str(reference),useFixedBase=True)
links={'base_link':-1};joints={};limits={}
for i in range(pb.getNumJoints(h)):
 info=pb.getJointInfo(h,i);links[info[12].decode()]=i
 if info[2]==pb.JOINT_REVOLUTE:
  num=int(info[1].decode().split('_')[1].split('.')[0]);joints[num]=i;limits[num]=info[8:10]
for body in (h,cad):
 for i in links.values():pb.changeDynamics(body,i,collisionMargin=0)
poses=[]
for a,b in [('open','half_closed'),('half_closed','opposition')]:
 for t in np.linspace(0,1,21):poses.append(('interpolation',((1-t)*np.array(POSTURES[a])+t*np.array(POSTURES[b])).tolist()))
for a in np.linspace(*limits[12],13):
 for b in np.linspace(*limits[13],13):
  q=list(POSTURES['open']);q[12]=a;q[13]=b;poses.append(('thumb_grid',q))
rng=np.random.RandomState(18)
for i in range(128):poses.append(('random',[float(rng.uniform(*limits[j])) for j in range(16)]))
for record in r:
 bank=torch.load(record['cache'],map_location='cpu');order=[int(n.split('_')[1].split('.')[0]) for n in bank['joint_names']]
 for obj,states in bank['states'].items():
  for idx in np.linspace(0,len(states)-1,16,dtype=int):
   q=np.zeros(16);q[order]=states[idx,:16].numpy();poses.append(('cache_'+record['config']['task']['name']+'_'+obj,q.tolist()))
pairs=[(a,b) for a,b in itertools.combinations(sorted(links),2) if [a,b] not in spec['excluded_pairs'] and (a,b) not in spec['excluded_pairs']]
records=[];summary={}
for kind,q in poses:
 for body in (h,cad):
  for num,j in joints.items():pb.resetJointState(body,j,q[num])
 mismatches=[];overlaps=[]
 for a,b in pairs:
  contacts=pb.getClosestPoints(h,h,.0001,linkIndexA=links[a],linkIndexB=links[b]);d=min([x[8] for x in contacts]+[0.])
  if d>=-.0001:continue
  overlaps.append((a,b,-d*1000))
  refc=pb.getClosestPoints(cad,cad,.001,linkIndexA=links[a],linkIndexB=links[b]);rd=min([x[8] for x in refc]+[.001])
  if rd>0:mismatches.append(dict(pair=[a,b],model_penetration_mm=-d*1000,reference_separation_lower_bound_mm=rd*1000,tabletop_enabled=not(a.endswith('_gel') and b.endswith('_gel'))))
 entry=summary.setdefault(kind,dict(poses=0,poses_with_overlap=0,poses_with_reference_disagreement=0));entry['poses']+=1;entry['poses_with_overlap']+=bool(overlaps);entry['poses_with_reference_disagreement']+=bool(mismatches)
 if mismatches:records.append(dict(kind=kind,q=q,disagreements=mismatches))
pb.disconnect(client)
result=dict(poses=len(poses),asset=str(asset),method='PyBullet zero margin, >0.1mm model penetration while corrected CAD convex envelopes are separated; both tasks share local geometry, tabletop additionally excludes gel-gel. Random configurations need not be reachable. This is not hardware ground truth; CAD envelopes overlap cannot prove real collision.',summary=summary,disagreements=records)
(out/'geometry_sweep.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(poses=len(poses),summary=summary,disagreements=records),indent=2))
