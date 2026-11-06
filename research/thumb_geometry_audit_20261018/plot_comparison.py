from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from hora.utils.mount_geometry import origin
A=Path('/root/In_Hand_Reconstruction/In_Hand_Manip/assets/allegro')
fig=plt.figure(figsize=(13,8))
for i,name in enumerate(['allegro_digit_compact_inward90.urdf','allegro_digit_compact_inward90_thumb_visual_fixed.urdf']):
 r=E.parse(A/name).getroot(); ax=fig.add_subplot(1,2,i+1,projection='3d')
 for link,col in [('link_12.0','#515968'),('link_13.0','#df903a')]:
  v=r.find("link[@name='%s']/visual"%link); m=trimesh.load(A/v.find('geometry/mesh').get('filename'),force='mesh'); T=origin(v.find('origin'))
  if link=='link_13.0': T=origin(r.find("joint[@name='joint_13.0']/origin"))@T
  m.apply_transform(T)
  ax.add_collection3d(Poly3DCollection(m.triangles*1000,facecolor=col,edgecolor='none',alpha=.96))
 c=r.find("link[@name='link_12.0']/collision"); b=trimesh.creation.box(extents=np.fromstring(c.find('geometry/box').get('size'),sep=' ')); b.apply_transform(origin(c.find('origin')))
 ax.add_collection3d(Line3DCollection(b.vertices[b.edges_unique]*1000,colors='#167bca',linewidths=.8,alpha=.6))
 ax.set(xlim=(-50,10),ylim=(-35,35),zlim=(-50,70),xlabel='Root local X (mm)',ylabel='Y (mm)',zlabel='Z (mm)'); ax.set_box_aspect((60,70,120)); ax.view_init(elev=16,azim=-57)
 ax.tick_params(labelsize=9)
 ax.set_title(('Current: visual Rx(180 deg)\n31.7 mm empty z interval' if i==0 else 'Corrected: visual Rx(0 deg)\nJoint and collision unchanged'),fontsize=12,pad=10)
fig.suptitle('Thumb-root assembly comparison (joint_13 = 0)',fontsize=16,y=.97)
fig.text(.5,.925,'Gray: root mesh     Orange: next link     Blue wireframe: collision box',ha='center',fontsize=11)
fig.subplots_adjust(top=.82,bottom=.11,left=.03,right=.94,wspace=.08)
fig.savefig(Path(__file__).parent/'thumb_before_after.png',dpi=170); plt.close(fig)
