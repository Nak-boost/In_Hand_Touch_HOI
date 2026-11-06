from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from omegaconf import OmegaConf
from hora.utils.scan_kinematics import ScanKinematics
from hora.utils.mount_geometry import rotation
root=Path('/root/In_Hand_Reconstruction/In_Hand_Manip');s=OmegaConf.load(root/'configs/tabletop_scan.yaml');b=OmegaConf.load(root/'configs'/s.base_config);t=OmegaConf.merge(b.task,s.task)
T=np.eye(4);T[:3,:3]=rotation([0,1,0],np.pi/2)@rotation([1,0,0],np.pi/2);T[:3,3]=t.env.tabletop.hand_position
kin=ScanKinematics(root/t.env.asset.handAsset,T);q=np.array(t.env.grasp.canonical_q);p,n,j=kin.evaluate(q);c=np.array(s.scan.center);r=p-c;r[:,2]=0;r/=np.linalg.norm(r,axis=1,keepdims=True)
for diameter in [.06,.07,.08]:
 def res(x):
  p,n,_=kin.evaluate(x);rad=p-c;rad[:,2]=0;norm=np.linalg.norm(rad,axis=1);rad/=norm[:,None]
  return np.r_[norm-(diameter/2+.009),.08*(n+rad).ravel(),.3*(p[:,2]-.54),.02*(p[:,:2]-(c+r*(diameter/2+.009))[:,:2]).ravel(),np.maximum(.523-p[:,2],0)*3,np.maximum(p[:,2]-.554,0)*3]
 qo=least_squares(res,np.clip(q,kin.lower+1e-6,kin.upper-1e-6),bounds=(kin.lower+1e-6,kin.upper-1e-6),max_nfev=250).x
 po,no,_=kin.evaluate(qo);rad=po-c;rad[:,2]=0;rad/=np.linalg.norm(rad,axis=1,keepdims=True)
 print('diam',diameter,'q',qo.round(3),'p',po.round(4),'align',np.rad2deg(np.arccos(np.clip(np.sum(-rad*no,axis=1),-1,1))).round(2),'vertical',np.rad2deg(np.arcsin(abs(no[:,2]))).round(2),flush=True)
 for angle in [0,10,20,30]:
  R=rotation([0,0,1],np.deg2rad(angle));des=c+(po-c)@R.T;des[:,:2]=c[:2]+(des-c)[:,:2]*(diameter/2)/(diameter/2+.009);dn=-rad@R.T
  def rr(x):
   pp,nn,_=kin.evaluate(x);err=(pp-des)*np.array([1,1,.15]);return np.r_[err.ravel(),.08*(nn-dn).ravel(),np.maximum(.523-pp[:,2],0)*3,np.maximum(pp[:,2]-.554,0)*3]
  qq=least_squares(rr,qo,bounds=(kin.lower+1e-6,kin.upper-1e-6),max_nfev=150).x;pp,nn,_=kin.evaluate(qq)
  print('angle',angle,'xyerr',np.linalg.norm((pp-des)[:,:2],axis=1).round(4),'z',pp[:,2].round(4),'normerr',np.rad2deg(np.arccos(np.clip(np.sum(nn*dn,axis=1),-1,1))).round(2),flush=True)
