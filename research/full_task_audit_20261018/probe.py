"""Read frozen task code; isolated 12/16-environment integrity probes, no training."""
import argparse,sys,json,pathlib,time
p=argparse.ArgumentParser();p.add_argument('index',type=int);a=p.parse_args()
out=pathlib.Path(__file__).parent;r=json.loads((out/'inventory.json').read_text())[a.index]
sys.path.insert(0,r['source'])
import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
from hora.tasks.allegro_digit_two_axis import AllegroDigitTwoAxis
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.manip_control import tracking_action
np.random.seed(901);torch.manual_seed(901);torch.set_num_threads(1)
c=r['config']['task'];c['env']['numEnvs']=12 if a.index==0 else 16;c['rl_device']='cpu';c['stage']='teacher'
cls=AllegroDigitTwoAxis if a.index==0 else AllegroDigitTabletop
e=cls(c,'cuda:0',-1,True)
e.stage='grasps'  # keep first episodes; do not track an old target after auto-reset
result={'source':r['source'],'task':c['name'],'num_envs':e.num_envs,'runtime_hand_shapes':e.gym.get_asset_rigid_shape_count(e.hand_asset),'checks':{},'modes':{}}
try:
 names=e.gym.get_asset_rigid_body_names(e.hand_asset)
 envnames=names+['object']+(['table'] if a.index>0 else [])
 assert [names[i] for i in e.gel_ids]==['digit_'+f+'_gel' for f in ('index','middle','ring','thumb')]
 for i,x in enumerate(e.envs):
  assert e.gym.get_actor_name(x,1)=='object'
  assert e.object_contact_reader.object_bodies[i]==len(names)
 result['checks']['body_order']=True
 forbidden={tuple(sorted(x)) for x in e.collision_spec['excluded_pairs']}
 violations=[];max_force_error=0.;self_gel_frames=0
 for mode in ('track_cached_targets','random_small'):
  obs=e.reset();desired=e.targets.clone();reward_sum=0.;failures=torch.zeros(e.num_envs,dtype=torch.bool);twoc=0;timecost=0.;failure_examples=[]
  for s in range(80):
   if mode=='track_cached_targets': act=tracking_action(e.q,e.targets,desired,e.actions,e.cfg)
   else: act=(torch.rand_like(e.actions)*2-1)*.25
   obs,reward,done,info=e.step(act)
   assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
   assert torch.allclose(sum(e.reward_terms.values()),reward)
   assert torch.equal(e.reward_terms['contact'],e.cfg['contact_weight']*(e.task_gel_contacts.sum(-1)>=2))
   assert torch.equal(done.bool(),e.terminal['done'])

   newly=e.terminal['failed'] & ~failures
   for ix in newly.nonzero().flatten().tolist():
    failure_examples.append(dict(env=ix,step=s+1,object=e.catalog[int(e.object_ids[ix])]['name'],mass_kg=float(e.domain_params[ix,3]),contact_lost=bool(e.contact_lost[ix]) if a.index>0 else None,outside_workspace=bool(e.outside_workspace[ix]) if a.index>0 else None,object_position=e.object_state[ix,:3].tolist(),gel_load_N=e.object_gel_normal_force[ix].tolist(),target_increment_lost=float(e.target_increment_lost[ix])))
   reward_sum+=float(reward.mean());failures|=e.terminal['failed'];twoc+=int((e.task_gel_contacts.sum(-1)>=2).sum());timecost+=e.object_contact_reader.last_wall_ms
   # Recompute loads with a separate scalar loop using Gym body names, no reader helper.
   scalar=np.zeros((e.num_envs,4),dtype=np.float32)
   for i,x in enumerate(e.envs):
    for ct in e.gym.get_env_rigid_contacts(x):
     b0,b1=int(ct['body0']),int(ct['body1']);f=float(ct['lambda'])
     if f<=0:continue
     n0=envnames[b0] if b0>=0 else 'ground'
     n1=envnames[b1] if b1>=0 else 'ground'
     pair=tuple(sorted((n0,n1)))
     if pair in forbidden or (a.index>0 and ('table' in pair and any(n.startswith('digit_') and n.endswith('_gel') for n in pair))):violations.append(pair)
     ob=e.object_contact_reader.object_bodies[i]
     other=b1 if b0==ob else b0 if b1==ob else -1
     if other in e.gel_ids:scalar[i,e.gel_ids.index(other)]+=f
   max_force_error=max(max_force_error,float(np.max(np.abs(scalar-e.object_gel_normal_force.numpy()))))
  result['modes'][mode]=dict(steps=80,finite=True,mean_reward=reward_sum/80,initial_states_ever_failed=int(failures.sum()),two_gel_fraction=twoc/(80*e.num_envs),mean_pair_wall_ms=timecost/80,first_failure_examples=failure_examples)
 result['checks'].update(forbidden_positive_contacts=len(violations),max_independent_contact_load_error_N=max_force_error,reward_sum_and_contact_reward=True)
 assert not violations and max_force_error<1e-4
 # Failure logic boundary without advancing physics; isolated test scene only.
 if a.index>0:
  e.object_state[:,:2]=e.workspace_center;e.missing_steps[:]=e.missing_limit-1
  no=torch.zeros(e.num_envs,4,dtype=torch.bool);finite=torch.ones(e.num_envs,dtype=torch.bool)
  assert e._failure_conditions(no,torch.zeros(e.num_envs),finite).all()
  no[:,0]=True
  assert not e._failure_conditions(no,torch.zeros(e.num_envs),finite).any()
  assert (e.missing_steps==0).all()
  result['checks']['loss_timeout_and_recontact_boundary']=True
 result['passed']=True
 (out/f'probe_{a.index}.json').write_text(json.dumps(result,indent=2));print('PROBE_RESULT',json.dumps(result),flush=True)
finally:
 print('CLOSING_ISOLATED_SCENE',flush=True);e.close();print('CLOSED',flush=True)
