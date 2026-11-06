import isaacgym
import json,pathlib,hashlib,itertools,xml.etree.ElementTree as ET
import torch
from hora.utils.misc import AverageScalarMeter
from hora.utils.hand_collision import collision_spec,collision_signature
from hora.utils.asset_geometry import verify_geometry_manifest
from hora.tasks.tabletop_math import tabletop_collision_spec
from hora.utils.manip_control import joint_targets
out=pathlib.Path(__file__).parent;records=json.loads((out/'inventory.json').read_text());results=[]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for r in records:
 asset=pathlib.Path(r['asset']);verify_geometry_manifest(asset);root=ET.parse(asset).getroot();cfg=r['config'];task=cfg['task'];spec=collision_spec(asset)
 if task['name']=='AllegroDigitTabletop':spec=tabletop_collision_spec(spec,task['env']['tabletop'])
 bank=torch.load(r['cache'],map_location='cpu');assert bank['asset_sha256']==sha(asset) and bank['collision_sha256']==collision_signature(spec)
 assert bank['objects']==task['env']['objects'];assert bank['generation_action_reference']==task['env']['manipulation']['action_reference']
 exclusions={tuple(sorted(p)) for p in spec['excluded_pairs']}
 for a,b in itertools.combinations(sorted(spec['body_masks']),2):assert bool(spec['body_masks'][a]&spec['body_masks'][b])==((a,b) in exclusions)
 joint={j.get('name'):j for j in root.findall('joint') if j.get('type')!='fixed'}
 lower=torch.tensor([float(joint[n].find('limit').get('lower')) for n in bank['joint_names']]);upper=torch.tensor([float(joint[n].find('limit').get('upper')) for n in bank['joint_names']])
 provenance=bank.get('per_object_provenance',{list(bank['states'])[0]:{'quality_validation':bank.get('quality_validation')}})
 checks=[]
 for obj,states in bank['states'].items():
  targets=bank['targets'][obj];assert torch.isfinite(states).all() and torch.isfinite(targets).all();assert (states[:,:16]>=lower-1e-6).all() and (states[:,:16]<=upper+1e-6).all();assert (targets>=lower-1e-6).all() and (targets<=upper+1e-6).all();assert torch.allclose(states[:,19:23].norm(dim=-1),torch.ones(len(states)),atol=1e-5)
  prov=provenance[obj];q=prov['quality_validation'];assert sha(q['report'])==q['report_sha256'];report=json.loads(pathlib.Path(q['report']).read_text());assert report['action_reference']=='actual_position'
  if 'path' in prov:
   assert sha(prov['path'])==prov['sha256'];individual=torch.load(prov['path'],map_location='cpu');assert torch.equal(states,individual['states'][obj]) and torch.equal(targets,individual['targets'][obj])
  checks.append({'object':obj,'n':len(states),'mean_target_preload_rad':float((targets-states[:,:16]).abs().mean()),'quality_hash_valid':True})
 ckpath=pathlib.Path(r['run'])/'stage1_nn/best.pth';ck=torch.load(ckpath,map_location='cpu') if ckpath.exists() else {}
 results.append(dict(pid=r['pid'],mesh_manifest=True,mask_truth_table=True,cache_checks=checks,best_checkpoint={k:ck.get(k) for k in ('agent_steps','epoch_num','best_rewards')},original_link_positive_mass=all(float(l.find('inertial/mass').get('value'))>0 for l in root.findall('link'))))
m=AverageScalarMeter(100);best=-10000;bug=(len(m)==0 and m.get_mean()>best);assert bug
q=torch.tensor([[.5]]);old_target=torch.tensor([[.7]]);withdrawn=old_target-.08;zero=torch.zeros_like(q);lo=torch.tensor([0.]);hi=torch.tensor([1.])
a,_=joint_targets(q,old_target,zero,lo,hi,'actual_position');b,_=joint_targets(q,withdrawn,zero,lo,hi,'actual_position');assert torch.equal(a,b)
result={'runs':results,'empty_meter_best_bug_reproduced':bug,'old_retraction_ignored_under_actual_position':bool(torch.equal(a,b))}
(out/'static_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
