"""Static convex-mesh collision inspection; an offline diagnostic, not training."""
import itertools
from pathlib import Path
import numpy as np
from hora.utils.mount_geometry import POSTURES, FINGERS, forward_kinematics
from hora.utils.hand_collision import collision_spec


def inspect_mesh_asset(asset):
    import pybullet as pb
    asset=Path(asset)
    excluded=set(map(tuple,collision_spec(asset)['excluded_pairs']))
    report={'asset':asset.name,'method':'URDF FK and PyBullet zero-margin convex closest-point queries', 'postures':{}}
    client=pb.connect(pb.DIRECT)
    try:
        hand=pb.loadURDF(str(asset),useFixedBase=True,physicsClientId=client)
        links={'base_link':-1};joint_ids={}
        for i in range(pb.getNumJoints(hand,physicsClientId=client)):
            info=pb.getJointInfo(hand,i,physicsClientId=client)
            links[info[12].decode()]=i
            if info[2]==pb.JOINT_REVOLUTE:
                joint_ids[int(info[1].decode().split('_')[1].split('.')[0])]=i
        for i in links.values():pb.changeDynamics(hand,i,collisionMargin=0,physicsClientId=client)
        for name,q in POSTURES.items():
            for number,i in joint_ids.items():pb.resetJointState(hand,i,q[number],physicsClientId=client)
            robot,transforms=forward_kinematics(asset,q)
            alignment={}
            for finger,index in FINGERS.items():
                parent=transforms['link_%d.0'%index];gel=transforms['digit_'+finger+'_gel']
                housing=transforms['digit_'+finger+'_housing']
                relative=np.linalg.inv(parent) @ housing
                alignment[finger]={'housing_origin_in_distal_m':relative[:3,3].tolist(),
                                  'normal_deviation_deg':float(np.degrees(np.arccos(np.clip(parent[:3,0] @ gel[:3,0],-1,1))))}
            overlaps=[]
            for a,b in itertools.combinations(sorted(links),2):
                if (a,b) in excluded:continue
                contacts=pb.getClosestPoints(hand,hand,0,linkIndexA=links[a],linkIndexB=links[b],physicsClientId=client)
                deepest=min([c[8] for c in contacts]+[0])
                if deepest < -1e-6:overlaps.append({'bodies':[a,b],'penetration_mm':float(-deepest*1000)})
            report['postures'][name]={'joint_angles':q,'alignment':alignment,'enabled_collision_overlaps':overlaps}
    finally:pb.disconnect(client)
    return report
