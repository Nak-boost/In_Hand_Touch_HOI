"""Geometry-only grasp proposals; every proposal must subsequently pass PhysX checks."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from hora.utils.mount_geometry import origin,rotation
import xml.etree.ElementTree as ET
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
robot=ET.parse(ROOT/'assets/allegro/allegro_digit_official.urdf').getroot()
joints=robot.findall('joint');lower=np.zeros(16);upper=np.zeros(16)
ordered=[];ready={'base_link'}
while len(ordered)<len(joints):
 for j in joints:
  if j in ordered or j.find('parent').get('link') not in ready:continue
  ordered.append(j);ready.add(j.find('child').get('link'))
chain=[]
for j in ordered:
 num=int(j.get('name').split('_')[1].split('.')[0]) if j.get('type')=='revolute' else None
 if num is not None:
  lower[num]=float(j.find('limit').get('lower'));upper[num]=float(j.find('limit').get('upper'))
 chain.append((j.find('parent').get('link'),j.find('child').get('link'),origin(j.find('origin')),num,np.fromstring(j.find('axis').get('xyz'),sep=' ') if num is not None else None))
world=np.eye(4);world[:3,:3]=rotation([0,1,0],-np.pi/2)@rotation([1,0,0],np.pi/2);world[:3,3]=[0,0,.5]
def fk(q):
 tf={'base_link':world}
 for parent,child,offset,num,axis in chain:
  t=offset
  if num is not None:
   r=np.eye(4);r[:3,:3]=rotation(axis,q[num]);t=t@r
  tf[child]=tf[parent]@t
 return [tf['digit_'+f+'_gel'] for f in ['index','middle','ring','thumb']]
seed=np.array([0,.7,.4,.25]*3+[1.1,.5,.5,.3])
out=[]
for diameter in [.050,.055,.060,.065,.070]:
 def residual(x):
  q=x[:16];center=x[16:]
  contact=np.concatenate([(t[:3,3]+t[:3,0]*diameter/2-center)/.03 for t in fk(q)])
  return np.r_[contact,.045*(q-seed),.12*(center-[-.005,.005,.64])/.03]
 sol=least_squares(residual,np.r_[seed,[-.005,.005,.64]],bounds=(np.r_[lower+.06,[-.045,-.04,.59]],np.r_[upper-.06,[.035,.06,.68]]),max_nfev=180)
 row=dict(diameter=diameter,q=sol.x[:16].tolist(),center=sol.x[16:].tolist(),max_contact_error_m=float(np.max(np.abs(residual(sol.x)[:12]))*.03),cost=float(sol.cost))
 out.append(row);print(json.dumps(row),flush=True)
Path(__file__).with_name('ik_seeds.json').write_text(json.dumps(out,indent=2))
