import json
from pathlib import Path
import numpy as np
import pybullet as pb
import yaml
ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');OUT=Path(__file__).resolve().parent
pb.connect(pb.DIRECT);quat=pb.multiplyTransforms([0,0,0],pb.getQuaternionFromEuler([0,-np.pi/2,0]),[0,0,0],pb.getQuaternionFromEuler([np.pi/2,0,0]))[1]
hand=pb.loadURDF(str(ROOT/'assets/allegro/allegro_digit_official.urdf'),[0,0,.5],quat,useFixedBase=True)
links={};joints={}
for i in range(pb.getNumJoints(hand)):
 info=pb.getJointInfo(hand,i);links[info[12].decode()]=i;pb.changeDynamics(hand,i,collisionMargin=0)
 if info[2]==pb.JOINT_REVOLUTE:joints[int(info[1].decode().split('_')[1].split('.')[0])]=i
for seed in json.loads((OUT/'collision_seeds_supported.json').read_text()):
 size=seed['size'];shape=seed['shape'];q=np.array(seed['q'])
 if shape=='sphere':geom=pb.createCollisionShape(pb.GEOM_SPHERE,radius=size[0]/2);rot=[0,0,0,1]
 elif shape=='box':geom=pb.createCollisionShape(pb.GEOM_BOX,halfExtents=np.array(size)/2);rot=[0,0,0,1]
 else:geom=pb.createCollisionShape(pb.GEOM_CAPSULE,radius=size[1]/2,height=size[0]-size[1]);rot=pb.getQuaternionFromEuler([0,np.pi/2,0])
 obj=pb.createMultiBody(baseMass=0,baseCollisionShapeIndex=geom,basePosition=seed['center'],baseOrientation=rot);pb.changeDynamics(obj,-1,collisionMargin=0)
 for k,j in joints.items():pb.resetJointState(hand,j,q[k])
 delta=np.zeros(16)
 for finger in seed['selected']:
  gel=links['digit_'+['index','middle','ring','thumb'][finger]+'_gel'];grad=[]
  for k in range(4*finger,4*finger+4):
   distances=[]
   for eps in [-.001,.001]:
    pb.resetJointState(hand,joints[k],q[k]+eps);cs=pb.getClosestPoints(hand,obj,.1,linkIndexA=gel);distances.append(min(c[8] for c in cs))
   pb.resetJointState(hand,joints[k],q[k]);grad.append((distances[1]-distances[0])/.002)
  grad=np.array(grad);delta[4*finger:4*finger+4]=np.clip(-.0015*grad/(grad@grad+1e-8),-.08,.08)
 cfgpath=ROOT/'configs'/('manip_official_'+seed['name']+'.yaml');cfg=yaml.safe_load(cfgpath.read_text());cfg['task']['env']['grasp']['preload_delta']=delta.tolist();cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False));print(seed['name'],delta.tolist(),flush=True)
 pb.removeBody(obj)
pb.disconnect()
