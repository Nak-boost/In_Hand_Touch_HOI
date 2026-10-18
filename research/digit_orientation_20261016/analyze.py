"""Read-only FK diagnostics of existing cache states; no model changes."""
import sys, json, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

ROOT = Path('/root/In_Hand_Reconstruction/In_Hand_Manip')
sys.path.insert(0, str(ROOT))
from hora.utils.mount_geometry import rotation, origin, FINGERS
OUT = Path(__file__).resolve().parent
asset = ROOT/'assets/allegro/allegro_digit_compact.urdf'
robot = ET.parse(asset).getroot()

def fk(q):
    # Use measured cache q verbatim, including any solver limit tolerance.
    transforms = {'base_link': np.eye(4)}
    todo = list(robot.findall('joint'))
    while todo:
        ready = [j for j in todo if j.find('parent').get('link') in transforms]
        assert ready
        for j in ready:
            t = origin(j.find('origin'))
            if j.get('type') == 'revolute':
                idx = int(j.get('name').split('_')[1].split('.')[0])
                turn = np.eye(4)
                turn[:3,:3] = rotation(np.fromstring(j.find('axis').get('xyz'), sep=' '), q[idx])
                t = t @ turn
            transforms[j.find('child').get('link')] = transforms[j.find('parent').get('link')] @ t
            todo.remove(j)
    return transforms

result = {'asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
          'angle_definition': 'Angle between gel apex +X and apex-to-object-center vector. Not a contact measurement; for boxes not the local surface-normal error.',
          'cases': {}}
examples = []
for name, filename, tabletop in [
    ('sphere55', 'digit_sphere55_compact27_checked_resume20261016.pt', False),
    ('tabletop_box45', 'digit_tabletop_box45_actual_checked_v3.pt', True)]:
    bank = torch.load(ROOT/'cache'/filename, map_location='cpu')
    assert bank['asset_sha256'] == result['asset_sha256']
    indices = [int(n.split('_')[1].split('.')[0]) for n in bank['joint_names']]
    states = next(iter(bank['states'].values())).numpy()
    world = np.eye(4)
    world[:3,:3] = rotation([0,1,0], np.pi/2 if tabletop else -np.pi/2) @ rotation([1,0,0], np.pi/2)
    world[:3,3] = bank['scene']['hand_position'] if tabletop else [0,0,.5]
    angles, distances, bases = [], [], []
    for s in states:
        q = np.zeros(16); q[indices] = s[:16]
        tf = {k: world @ v for k,v in fk(q).items()}
        a, d = [], []
        for finger in FINGERS:
            t = tf['digit_'+finger+'_gel']; delta = s[16:19] - t[:3,3]
            d.append(float(np.linalg.norm(delta)))
            a.append(float(np.degrees(np.arccos(np.clip(t[:3,0] @ delta / d[-1], -1, 1)))))
        angles.append(a); distances.append(d); bases.append(q[[0,4,8]])
    angles, distances = np.asarray(angles), np.asarray(distances)
    case = {'cache': filename, 'states': len(states), 'fingers': {},
            'base_joint_median_deg': np.degrees(np.median(bases, axis=0)).tolist()}
    for i,finger in enumerate(FINGERS):
        case['fingers'][finger] = {'angle_deg_p10_p50_p90': np.percentile(angles[:,i], [10,50,90]).tolist(),
                                 'apex_to_center_mm_median': float(np.median(distances[:,i])*1000)}
    if not tabletop:
        old = json.loads(Path(bank['quality_validation']['report']).read_text())
        accepted = old['accepted_indices']
        case['prior_contact_probe_note'] = 'Existing validation: real gel-object contact after 1 s hold, selected accepted cache states, not a newly run simulation or trained-policy statistic.'
        for entry in old['retractions']:
            case['fingers'][entry['finger']]['prior_probe_contact_count'] = int(np.asarray(entry['initial_contact'])[accepted].sum())
    result['cases'][name] = case
    s = states[0]; q = np.zeros(16); q[indices] = s[:16]
    examples.append((name, {k: world @ v for k,v in fk(q).items()}, s[16:19]))
(OUT/'geometry.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))

# True 3D FK skeleton and apex normals, not rendered sensor meshes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,5.5))
colors = ['#0072B2','#E69F00','#009E73','#CC79A7']
for col,(name,tf,center) in enumerate(examples):
    ax = fig.add_subplot(1,2,col+1,projection='3d')
    for (finger,distal),color in zip(FINGERS.items(),colors):
        points = np.array([tf['link_%d.0'%j][:3,3] for j in range(distal-3,distal+1)] + [tf['digit_'+finger+'_gel'][:3,3]])
        points = (points-center)*1000
        ax.plot(*points.T, '-o', color=color, markersize=3, label=finger)
        t = tf['digit_'+finger+'_gel']; p = (t[:3,3]-center)*1000
        ax.quiver(*p, *t[:3,0], length=22, color=color, linewidth=2)
        ax.plot([p[0],0],[p[1],0],[p[2],0],':',color=color,alpha=.5)
    ax.scatter([0],[0],[0],color='black',s=40,label='object center')
    if name=='sphere55':
        u,v=np.mgrid[0:2*np.pi:25j,0:np.pi:15j]
        ax.plot_wireframe(27.5*np.cos(u)*np.sin(v),27.5*np.sin(u)*np.sin(v),27.5*np.cos(v),color='gray',alpha=.2)
    ax.set(xlabel='World X offset (mm)',ylabel='World Y offset (mm)',zlabel='World Z offset (mm)',title=name+' / cache state 0')
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=32,azim=-58)
    if col==0: ax.legend(fontsize=8)
fig.suptitle('Existing mount: solid arrows = gel apex outward normals\nFK geometry only; dotted lines point to object center, not measured contacts')
fig.tight_layout()
fig.savefig(OUT/'normals.png',dpi=150)
