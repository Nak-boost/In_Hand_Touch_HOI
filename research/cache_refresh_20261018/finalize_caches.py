"""Intersect complete independent trials; never overwrite historical cache files."""
import copy,hashlib,json,pathlib
import torch
root=pathlib.Path('/root/In_Hand_Reconstruction/In_Hand_Manip');out=pathlib.Path(__file__).parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
plan=json.loads((out/'plan.json').read_text());summary={}
# Preflight all reports before writing any new bank.
for job in plan:
 status=json.loads((out/(job['name']+'.status.json')).read_text());assert status['exit_code']==0
 json.loads((out/(job['name']+'.json')).read_text())
for tag in ('table_sphere','table_box','air_six'):
 jobs=[j for j in plan if j['tag']==tag];reports=[json.loads((out/(j['name']+'.json')).read_text()) for j in jobs]
 for j in jobs:
  status=json.loads((out/(j['name']+'.status.json')).read_text());assert status['exit_code']==0
 source=root/jobs[0]['cache'];old=torch.load(source,map_location='cpu');new=copy.deepcopy(old)
 for d in reports:
  assert d['input_cache_sha256']==sha(source)
  assert d['asset_sha256']==old['asset_sha256'] and d['collision_sha256']==old['collision_sha256']
  assert d['validation_contract']==reports[0]['validation_contract']
  assert d['hold_seconds']>=30-1e-5 and d['margin_force_N']>=.15
 selected={};counts={}
 for obj in old['objects']:
  name=obj['name'];ids=sorted(set.intersection(*(set(d['accepted_indices'][name]) for d in reports)))
  assert len(ids)>=32,(name,len(ids),'too few robust states')
  selected[name]=ids;counts[name]=dict(before=len(old['states'][name]),accepted=len(ids),per_seed=[len(d['accepted_indices'][name]) for d in reports])
  new['states'][name]=old['states'][name][ids].clone();new['targets'][name]=old['targets'][name][ids].clone()
 quality=dict(validation_kind='multi_seed_contact_margin_v1',passed=True,action_reference=reports[0]['validation_contract']['action_reference'],control_dt=reports[0]['control_dt'],validation_contract=reports[0]['validation_contract'],asset_sha256=old['asset_sha256'],collision_sha256=old['collision_sha256'],input_cache=str(source),input_cache_sha256=sha(source),seeds=[d['seed'] for d in reports],hold_seconds=reports[0]['hold_seconds'],margin_force_N=.15,accepted_indices=selected,counts=counts,trials=[dict(path=str(out/(j['name']+'.json')),sha256=sha(out/(j['name']+'.json'))) for j in jobs],criteria='Each state passes all 3 seeds: 30s bounded tracking, no task failure, >=2 object-gel loads >0.15N on >=90% frames, nongel support <=10%; tabletop >=90% table support and max net gel force <=2N; in-hand <=2 joints within .03rad of limits. Single-finger withdrawals are diagnostics, not a gait feasibility guarantee.')
 qfile=out/(tag+'_quality.json');assert not qfile.exists();qfile.write_text(json.dumps(quality,indent=2)+'\n')
 new['validation_contract']=reports[0]['validation_contract']
 new['previous_quality_validation']=old.get('quality_validation',old.get('per_object_provenance'))
 new.pop('per_object_provenance',None)
 new['quality_validation']=dict(report=str(qfile),report_sha256=sha(qfile),action_reference=quality['action_reference'],controller='bounded_target_tracking_v1',validation_kind=quality['validation_kind'])
 filename={'table_sphere':'digit_tabletop_sphere55_inward90_robust_20261018.pt','table_box':'digit_tabletop_box45_inward90_robust_20261018.pt','air_six':'digit_six_objects_inward90_robust_20261018.pt'}[tag]
 dest=root/'cache'/filename;assert not dest.exists();torch.save(new,dest)
 summary[tag]=dict(cache=str(dest),sha256=sha(dest),quality=str(qfile),counts=counts)
 if tag=='air_six':
  for obj in new['objects']:
   one=copy.deepcopy(new);name=obj['name'];one['objects']=[obj];one['states']={name:new['states'][name]};one['targets']={name:new['targets'][name]}
   dest=root/'cache'/f'digit_{name}_inward90_robust_20261018.pt';assert not dest.exists();torch.save(one,dest)
  summary[tag]['single_sphere_cache']=str(root/'cache/digit_sphere55_inward90_robust_20261018.pt')
(out/'cache_manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
