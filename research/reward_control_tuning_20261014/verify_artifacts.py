"""Verify saved final probes and the short PPO integration run."""
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch, numpy as np, json
root=Path(__file__).parent;p=root/'ppo_smoke/reward_control_20261014_terminal250'
ea=EventAccumulator(str(p/'stage1_tb'));ea.Reload();tags=ea.Tags()['scalars']
assert all(np.isfinite([v.value for v in ea.Scalars(t)]).all() for t in tags)
terms=[t for t in tags if t.startswith('reward/') and t!='reward/total']
assert len(terms)==13
vals=np.stack([[v.value for v in ea.Scalars(t)] for t in terms]);total=np.array([v.value for v in ea.Scalars('reward/total')]);np.testing.assert_allclose(vals.sum(0),total,rtol=1e-5,atol=1e-5)
ck=torch.load(p/'stage1_nn/last.pth',map_location='cpu');assert all(torch.isfinite(v).all() for v in ck['model'].values())
result=dict(all_logged_scalars_finite=True,reward_terms=terms,reward_sum_max_error=float(np.abs(vals.sum(0)-total).max()),logged_updates=len(total),checkpoint_agent_steps=ck['agent_steps'],checkpoint_epoch=ck['epoch_num'],model_parameters_finite=True)
for scene in ['inhand','tabletop','official','official_terminal250']:
    stem=scene if scene.endswith('terminal250') else scene+'_final'
    file=root/(stem+'_samples.npz')
    if not file.exists():continue
    a=np.load(file);result[scene]={}
    meta=json.loads((root/(stem+'_metadata.json')).read_text())
    drop=meta['config']['task']['env']['manipulation']['drop_penalty']
    for protocol in ['hold','small_noise','outward_push','alternating']:
        v={k.split('__')[1]:a[k] for k in a.files if k.startswith(protocol+'__')}
        assert all(np.isfinite(x).all() for x in v.values())
        assert np.all(v['failure_penalty']==-drop*v['failed'])
        assert np.all(v['off_axis_penalty']>=-.50001) and np.all(v['overspeed_penalty']>=-.25001)
        assert np.all(v['limit_push_penalty']<=0) and np.all(v['limit_push_penalty']>=-.25001)
        if protocol=='hold':assert not v['limit_push_penalty'].any()
        result[scene][protocol]=dict(drop_penalty=drop,failed=int(v['failed'].sum()),timeout_only=int((v['done'] & ~v['failed']).sum()),limit_cost_mean=float(-v['limit_push_penalty'].mean()))
(root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
