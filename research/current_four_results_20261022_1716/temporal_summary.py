"""Describe late progress and sustained binary contact-return events; not a gait classifier."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent
out={}
for path in sorted(P.glob('timeline_*.npz')):
    z=np.load(path);alive=z['alive'];g=z['gels'];turns=z['turns'];dt=float(z['dt']);rows={}
    meta=json.loads((P/path.name.replace('timeline_','eval_').replace('.npz','.json')).read_text())
    catalog=list(meta['per_object'])
    for oid,name in enumerate(catalog):
        ids=np.where(z['object_ids']==oid)[0];surv=ids[alive[-1,ids]]
        row={'n':len(ids),'survivors':len(surv),'segments':{}}
        for lo,hi in [(0,10),(10,20),(20,30)]:
            a,b=round(lo/dt),round(hi/dt)
            if len(surv):
                inc=turns[b-1,surv]-(turns[a-1,surv] if a else 0)
                gels=g[a:b,surv].sum(-1)
                row['segments'][f'{lo}-{hi}s']={'net_turns_survivors':float(inc.mean()),'fraction_survivors_net_ge_30deg':float((inc>=1/12).mean()),'two_gels_survivor_frames':float((gels>=2).mean()),'no_gel_survivor_frames':float((gels==0).mean())}
        events=productive=0;event_envs=set();productive_envs=set()
        for i in ids:
            end=int(alive[:,i].sum())
            for f in range(4):
                seq=g[:end,i,f];bounds=np.r_[0,np.where(seq[1:]!=seq[:-1])[0]+1,end]
                for k in range(2,len(bounds)-1):
                    start,stop=int(bounds[k]),int(bounds[k+1])
                    # Stable contact, stable release, stable return, all >=0.2s.
                    if not seq[start] or min(stop-start,bounds[k]-bounds[k-1],bounds[k-1]-bounds[k-2])*dt<.199:continue
                    if start*dt<10 or start+round(1/dt)>=end:continue
                    events+=1;event_envs.add(int(i))
                    # Association only: object may be driven by other fingers.
                    if turns[start+round(1/dt),i]-turns[start,i]>=10/360:
                        productive+=1;productive_envs.add(int(i))
        row['after10s_sustained_contact_returns']={'events':events,'episodes':len(event_envs),'events_with_next1s_net_ge10deg':productive,'episodes_with_next1s_net_ge10deg':len(productive_envs)}
        rows[name]=row
    out[path.stem]=rows
(P/'temporal_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
