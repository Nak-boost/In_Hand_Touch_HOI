"""Collision-aware static proposals using published convex meshes, not a physics substitute."""
import json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import pybullet as pb
from hora.utils.hand_collision import collision_spec
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');OUT=Path(__file__).resolve().parent
asset=ROOT/'assets/allegro/allegro_digit_official.urdf'
pb.connect(pb.DIRECT)
quat=pb.multiplyTransforms([0,0,0],pb.getQuaternionFromEuler([0,-np.pi/2,0]),[0,0,0],pb.getQuaternionFromEuler([np.pi/2,0,0]))[1]
hand=pb.loadURDF(str(asset),basePosition=[0,0,.5],baseOrientation=quat,useFixedBase=True)
links={'base_link':-1};joints={};lo=np.zeros(16);hi=np.zeros(16)
for i in range(pb.getNumJoints(hand)):
 info=pb.getJointInfo(hand,i);links[info[12].decode()]=i
 if info[2]==pb.JOINT_REVOLUTE:
  n=int(info[1].decode().split('_')[1].split('.')[0]);joints[n]=i;lo[n]=info[8];hi[n]=info[9]
for i in links.values():pb.changeDynamics(hand,i,collisionMargin=0)
exclude=set(map(tuple,collision_spec(asset)['excluded_pairs']))
pairs=[tuple(sorted((links[a],links[b]))) for a,b in itertools.combinations(sorted(links),2) if (a,b) not in exclude]
pair_idx={p:i for i,p in enumerate(pairs)}
gels=[links['digit_'+f+'_gel'] for f in ['index','middle','ring','thumb']]
alllinks=list(links.values());non=[i for i in alllinks if i not in gels]
experiments=[('sphere55','sphere',[.055]*3),('sphere65','sphere',[.065]*3),('box45','box',[.045]*3),('box55','box',[.055]*3),('capsule70','capsule',[.07,.05,.05]),('capsule80','capsule',[.08,.06,.06])]
results=[]
for name,shape,size in experiments:
 if shape=='sphere':geom=pb.createCollisionShape(pb.GEOM_SPHERE,radius=size[0]/2);rot=[0,0,0,1]
 elif shape=='box':geom=pb.createCollisionShape(pb.GEOM_BOX,halfExtents=np.array(size)/2);rot=[0,0,0,1]
 else:geom=pb.createCollisionShape(pb.GEOM_CAPSULE,radius=size[1]/2,height=size[0]-size[1]);rot=pb.getQuaternionFromEuler([0,np.pi/2,0])
 obj=pb.createMultiBody(baseMass=0,baseCollisionShapeIndex=geom,basePosition=[0,0,.64],baseOrientation=rot);pb.changeDynamics(obj,-1,collisionMargin=0)
 candidates=[]
 for selected in [[0,1,3],[1,2,3],[0,1,2,3]]:
  seed=np.array([0,1,.25,.1]*3+[1.2,.6,.4,.1]);center=np.array([-.015 if selected[0]==0 else .012,.01,.635])
  def residual(x):
   for k,j in joints.items():pb.resetJointState(hand,j,x[k])
   pb.resetBasePositionAndOrientation(obj,x[16:],rot)
   gaps={i:.15 for i in alllinks}
   for c in pb.getClosestPoints(hand,obj,.15):gaps[c[3]]=min(gaps[c[3]],c[8])
   selfgap=np.zeros(len(pairs))
   for c in pb.getClosestPoints(hand,hand,.002):
    key=tuple(sorted((c[3],c[4])))
    if key in pair_idx:selfgap[pair_idx[key]]=max(selfgap[pair_idx[key]],.002-c[8])
   return np.r_[[gaps[gels[i]]/.01 for i in selected],
                [min(0,gaps[i]-.001)/.004 for i in non],selfgap/.003,.025*(x[:16]-seed),.06*(x[16:]-center)/.03]
  sol=least_squares(residual,np.r_[seed,center],bounds=(np.r_[lo+.07,[-.045,-.025,.595]],np.r_[hi-.07,[.045,.07,.68]]),diff_step=1e-4,max_nfev=120,ftol=1e-5)
  r=residual(sol.x);row=dict(name=name,shape=shape,size=size,selected=selected,q=sol.x[:16].tolist(),center=sol.x[16:].tolist(),cost=float(sol.cost),contact_errors_mm=(r[:len(selected)]*10).tolist(),self_penetration_margin_max_mm=float(max(r[len(selected)+len(non):len(selected)+len(non)+len(pairs)])*3))
  candidates.append(row);print(json.dumps(row),flush=True)
 best=min(candidates,key=lambda c:c['cost']);results.append(best)
 (OUT/'collision_seeds.json').write_text(json.dumps(results,indent=2))
 pb.removeBody(obj)
pb.disconnect()
