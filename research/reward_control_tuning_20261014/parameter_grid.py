from pathlib import Path
import numpy as np,json
p=Path(__file__).parent;rows=[]
for pose in [.3,.1,.05,.03]:
 for speed_cap in [0,.25]:
  for scene in ['inhand','tabletop']:
   a=np.load(p/(scene+'_samples.npz'));required=[];parts={}
   for protocol in ['hold','small_noise','outward_push','alternating']:
    v={k.split('__')[1]:a[k] for k in a.files if k.startswith(protocol+'__')}
    base=sum(v[k] for k in v if k not in ['failed','done','length','blocked','off_axis_squared','overspeed_squared','failure_penalty','pose_penalty','overspeed_penalty'])+v['pose_penalty']*(pose/.3)
    c=-v['overspeed_penalty'];base-= c if not speed_cap else speed_cap*c/(speed_cap+c)
    rr=[]
    for env in range(32):
     start=0
     for t in np.flatnonzero(v['done'][:,env]):
      b=base[start:t+1,env];L=len(b);f=v['failed'][t,env];G=np.dot(.99**np.arange(L),b);den=1-.99**(L-1) if f else 1
      if den>1e-8:rr.append(max(0,-G/den))
      start=t+1
    parts[protocol]={'required_drop_p95':float(np.quantile(rr,.95)),'required_drop_max':float(max(rr)),'mean_pose_cost':float((-v['pose_penalty']*pose/.3).mean())};required+=rr
   row=dict(scene=scene,pose_weight=pose,overspeed_cost_cap=speed_cap,required_drop_p95=float(np.quantile(required,.95)),required_drop_p99=float(np.quantile(required,.99)),required_drop_max=float(max(required)),parts=parts);rows.append(row)
   print({k:v for k,v in row.items() if k!='parts'})
(p/'parameter_grid.json').write_text(json.dumps(rows,indent=2))
