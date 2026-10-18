"""Paired reward comparison on fixed physics traces; not a policy performance test."""
from pathlib import Path
import json
import numpy as np
p=Path(__file__).parent
exclude={'failed','done','length','blocked','off_axis_squared','overspeed_squared','failure_penalty','pose_penalty','overspeed_penalty','off_axis_penalty','limit_push_penalty'}
def summary(x):
    x=np.asarray(x)
    return dict(count=len(x),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),maximum=float(x.max())) if len(x) else None
result={}
for scene in ['inhand','tabletop','official']:
    stem=scene+('_final' if scene=='official' else '')
    if not (p/(stem+'_samples.npz')).exists():continue
    meta=json.loads((p/(stem+'_metadata.json')).read_text())
    pose_weight=meta['config']['task']['env']['manipulation']['pose_weight']
    a=np.load(p/(stem+'_samples.npz'));records={};required=[];suffix_long=[];suffix_short=[];comparisons=[]
    for protocol in ['hold','small_noise','outward_push','alternating']:
        v={k.split('__')[1]:a[k] for k in a.files if k.startswith(protocol+'__')}
        common=sum(v[k] for k in v if k not in exclude)
        off=.15*v['off_axis_squared'] if scene!='tabletop' else np.zeros_like(common)
        over=.1*v['overspeed_squared'] if scene!='tabletop' else np.zeros_like(common)
        pose=-v['pose_penalty']*(.3/pose_weight);limit=.25*v['blocked']
        old=common-off-over-pose
        new=common-.5*off/(.5+off)-.25*over/(.25+over)-pose/6-limit
        prot=[]
        for e in range(v['done'].shape[1]):
            start=0
            for t in np.flatnonzero(v['done'][:,e]):
                failed=bool(v['failed'][t,e]);L=t-start+1;discount=.99**np.arange(L)
                Go=float(discount@old[start:t+1,e]);Gn=float(discount@new[start:t+1,e]);den=1-.99**(L-1) if failed else 1
                if den>1e-8:required.append(max(0,-Gn/den))
                r=dict(length=L,failed=failed,old_return=Go-(.99**(L-1)*5 if failed else 0),new_return=Gn-(.99**(L-1)*250 if failed else 0))
                prot.append(r);comparisons.append(r)
                g=0.
                for s in range(t,start-1,-1):
                    g=float(new[s,e])+.99*g;remaining=t-s+1;d=1-.99**(remaining-1) if failed else 1
                    if d>1e-8:(suffix_long if remaining>=20 else suffix_short).append(max(0,-g/d))
                start=t+1
        records[protocol]={'mean_pose_cost_old':float(pose.mean()),'mean_pose_cost_new':float((pose/6).mean()),
            'mean_off_cost_old':float(off.mean()),'mean_off_cost_new':float((.5*off/(.5+off)).mean()),
            'overspeed_cost_max_old':float(over.max()),'overspeed_cost_max_new':float((.25*over/(.25+over)).max()),
            'mean_limit_push_cost':float(limit.mean()),'mean_blocked_command':float(v['blocked'].mean()),
            'complete_episodes':len(prot),
            'fraction_worse_than_immediate_failure_old':sum(r['old_return'] < -5 for r in prot)/len(prot) if prot else None,
            'fraction_worse_than_immediate_failure_new':sum(r['new_return'] < -250 for r in prot)/len(prot) if prot else None}
    result[scene]={'protocols':records,'required_D_episode_start':summary(required),
        'required_D_suffix_at_least_20_steps':summary(suffix_long),'required_D_suffix_under_20_steps':summary(suffix_short),
        'note':'Immediate failure benchmark is -D only, excludes action-dependent same-step shaping. Finite observed trajectories only; cannot prove policy incentives globally.'}
(p/'final_scale_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
