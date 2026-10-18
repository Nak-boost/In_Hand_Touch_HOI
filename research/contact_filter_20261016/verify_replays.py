from pathlib import Path
import json
import numpy as np
p=Path(__file__).resolve().parent
a=np.load(p/'cpu_off_1.npz');b=np.load(p/'cpu_on_1.npz')
checks={}
assert np.array_equal(a['initial'],b['initial'])
checks['identical_initial_state']=True
checks['cross_run_q_max_difference']=float(np.max(np.abs(a['q']-b['q'])))
checks['cross_run_trajectory_note']='Independent GPU PhysX runs diverge despite same initial states and tape; do not compare reward frame-by-frame across runs.'
for name,data,enabled in [('off',a,False),('on',b,True)]:
 expected=data['pair_load']>.1 if enabled else data['raw']
 assert np.array_equal(data['task'],expected)
 assert np.allclose(data['reward_contact'],.05*(expected.sum(-1)>=2))
 missing=np.zeros(expected.shape[1],dtype=np.int64)
 for step,c in enumerate(expected):
  missing=np.where(c.any(-1),0,missing+1)
  assert np.array_equal(data['missing'][step],missing)
  outside=np.square(data['object'][step,:,:2]).sum(-1)>.1**2
  failure=(missing>=40)|outside
  assert np.array_equal(data['failed'][step],failure)
  assert np.allclose(data['reward_failure'][step],-250*failure)
 checks[name+'_reward_and_failure_match']=True
 checks[name+'_envs_ever_failed']=int(data['failed'].any(0).sum())
checks['removed_reward_environment_frames']=int(((b['raw'].sum(-1)>=2)&(b['reward_contact']==0)).sum())
checks['raw_signal_without_object_environment_frames']=int((b['raw'].any(-1)&~b['task'].any(-1)).sum())
raw_missing=np.zeros(b['raw'].shape[1],dtype=np.int64)
raw_failure=[]
for step,c in enumerate(b['raw']):
 raw_missing=np.where(c.any(-1),0,raw_missing+1)
 raw_failure.append((raw_missing>=40)|(np.square(b['object'][step,:,:2]).sum(-1)>.1**2))
checks['failure_on_while_raw_counterfactual_alive_frames']=int((b['failed']&~np.array(raw_failure)).sum())
checks['actor_self_contact_observation_frames']=int(((b['obs'][1:,:,64:68]>0).any(-1)&b['raw'][:-1].any(-1)&~b['task'][:-1].any(-1)).sum())
assert np.array_equal(b['obs'],b['history_last'])
assert np.all(b['obs'][1:,:,64:68] <= b['raw'][:-1])
assert checks['actor_self_contact_observation_frames']>0
checks['total_environment_frames']=int(b['raw'].shape[0]*b['raw'].shape[1])
assert checks['removed_reward_environment_frames']>0
assert checks['failure_on_while_raw_counterfactual_alive_frames']>0
(p/'verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
