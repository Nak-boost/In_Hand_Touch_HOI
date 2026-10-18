import json,itertools
from pathlib import Path
import numpy as np
import pybullet as pb
from hora.utils.mount_geometry import POSTURES
from hora.utils.hand_collision import collision_spec
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');OUT=Path(__file__).parent
modes=["official","palm3","cad_envelope"]
client=pb.connect(pb.DIRECT);data={};rng=np.random.RandomState(14)
for mode in modes:
 asset=ROOT/'assets/allegro'/f'allegro_digit_{mode}.urdf';hand=pb.loadURDF(str(asset),useFixedBase=True)
 links={'base_link':-1};joints={};limits={}
 for i in range(pb.getNumJoints(hand)):
  inf=pb.getJointInfo(hand,i);links[inf[12].decode()]=i
  if inf[2]==pb.JOINT_REVOLUTE:
   number=int(inf[1].decode().split('_')[1].split('.')[0]);joints[number]=i;limits[number]=inf[8:10]
 for i in links.values():pb.changeDynamics(hand,i,collisionMargin=0)
 if mode==modes[0]:
  poses=[]
  for a,b in [('open','half_closed'),('half_closed','opposition')]:
   for t in np.linspace(0,1,21):poses.append(('interpolation',((1-t)*np.array(POSTURES[a])+t*np.array(POSTURES[b])).tolist()))
  for a in np.linspace(*limits[12],13):
   for b in np.linspace(*limits[13],13):
    q=list(POSTURES['open']);q[12]=a;q[13]=b;poses.append(('thumb_sweep',q))
  for i in range(128):poses.append(('random', [float(rng.uniform(*limits[j])) for j in range(16)]))
 excluded=set(map(tuple,collision_spec(asset)['excluded_pairs']));pairs=[(a,b) for a,b in itertools.combinations(sorted(links),2) if (a,b) not in excluded]
 results=[]
 for label,q in poses:
  for num,j in joints.items():pb.resetJointState(hand,j,q[num])
  overlap={}
  for a,b in pairs:
   c=pb.getClosestPoints(hand,hand,1e-4,linkIndexA=links[a],linkIndexB=links[b]);d=min([x[8] for x in c]+[0])
   if d<-.0001:overlap[a+'|'+b]=-d*1000
  results.append({'kind':label,'q':q,'overlaps_mm':overlap})
 data[mode]=results;pb.removeBody(hand)
pb.disconnect(client)
(OUT/'sweep_extra_raw.json').write_text(json.dumps(data))
summary={}
for mode,rs in data.items():
 sm={}
 for kind in ['interpolation','thumb_sweep','random']:
  ids=[i for i,r in enumerate(rs) if r['kind']==kind]; extra={};allpairs={}
  for i in ids:
   for pair,depth in rs[i]['overlaps_mm'].items():
    allpairs[pair]=allpairs.get(pair,0)+1
    if pair not in data['official'][i]['overlaps_mm']:
     e=extra.setdefault(pair,{'count':0,'max_depth_mm':0,'example_index':i});e['count']+=1;e['max_depth_mm']=max(depth,e['max_depth_mm'])
  sm[kind]={'n':len(ids),'pairs':allpairs,'pairs_absent_in_85_shape_reference':extra}
 summary[mode]=sm
(OUT/'sweep_extra_summary.json').write_text(json.dumps(summary,indent=2))
for mode,sm in summary.items():
 print(mode)
 for kind,s in sm.items():print(kind,'pairs',len(s['pairs']),'reference_disagreements',s['pairs_absent_in_85_shape_reference'])
