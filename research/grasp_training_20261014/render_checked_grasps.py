"""Static geometry preview of checked cache rows; not an additional dynamics validation."""
from pathlib import Path
import json
import numpy as np
import torch
import pybullet as pb
from PIL import Image, ImageDraw, ImageFont
from hydra import compose, initialize_config_dir

ROOT=Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
OUT=Path(__file__).resolve().parent
pb.connect(pb.DIRECT)
base_q=pb.multiplyTransforms([0,0,0],pb.getQuaternionFromEuler([0,-np.pi/2,0]),[0,0,0],pb.getQuaternionFromEuler([np.pi/2,0,0]))[1]
hand=pb.loadURDF(str(ROOT/'assets/allegro/allegro_digit_official.urdf'),basePosition=[0,0,.5],baseOrientation=base_q,useFixedBase=True)
joints={}
for i in range(pb.getNumJoints(hand)):
 info=pb.getJointInfo(hand,i)
 if info[2]==pb.JOINT_REVOLUTE:joints[info[1].decode()]=i
canvas=Image.new('RGB',(1920,1120),'white');draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
records=[]
for idx,name in enumerate(['sphere55','sphere65','box45','box55','capsule70','capsule80']):
 with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):cfg=compose(config_name='manip_official_'+name)
 bank=torch.load(ROOT/cfg.task.env.grasp.cache,map_location='cpu');state=bank['states'][name][0].numpy()
 for k,n in enumerate(bank['joint_names']):pb.resetJointState(hand,joints[n],float(state[k]))
 obj=bank['objects'][0];shape=obj['shape'];size=obj['size'];quat=state[19:23].tolist()
 if shape=='sphere':visual=pb.createVisualShape(pb.GEOM_SPHERE,radius=size[0]/2,rgbaColor=[.92,.45,.1,1])
 elif shape=='box':visual=pb.createVisualShape(pb.GEOM_BOX,halfExtents=np.array(size)/2,rgbaColor=[.92,.45,.1,1])
 else:
  visual=pb.createVisualShape(pb.GEOM_CAPSULE,radius=size[1]/2,length=size[0]-size[1],rgbaColor=[.92,.45,.1,1])
  quat=pb.multiplyTransforms([0,0,0],quat,[0,0,0],pb.getQuaternionFromEuler([0,np.pi/2,0]))[1]
 body=pb.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,basePosition=state[16:19].tolist(),baseOrientation=quat)
 target=np.array(state[16:19]);eye=target+[-.19,-.25,.21]
 view=pb.computeViewMatrix(eye.tolist(),target.tolist(),[0,0,1]);proj=pb.computeProjectionMatrixFOV(45,640/520,.01,2)
 frame=pb.getCameraImage(640,520,view,proj,renderer=pb.ER_TINY_RENDERER,lightDirection=[-1,-1,3],shadow=1)[2]
 x=(idx%3)*640;y=(idx//3)*560
 canvas.paste(Image.fromarray(np.array(frame,dtype=np.uint8)[:,:,:3]),(x,y+30));draw.text((x+12,y+10),name+' | checked row 0 | static geometry preview',fill='black',font=font)
 records.append(dict(object=name,cache=str(ROOT/cfg.task.env.grasp.cache),row=0))
 pb.removeBody(body)
canvas.save(OUT/'checked_grasps_preview.png');(OUT/'checked_grasps_preview.json').write_text(json.dumps(records,indent=2)+'\n');pb.disconnect()
