import concurrent.futures,json,os,pathlib,subprocess,time
root=pathlib.Path('/root/In_Hand_Reconstruction/In_Hand_Manip');out=pathlib.Path(__file__).parent
jobs=[]
for tag,config,cache in [
 ('table_sphere','manip_tabletop_inward90_hora16384_sphere55','cache/digit_tabletop_sphere55_inward90_checked_20261016.pt'),
 ('table_box','manip_tabletop_inward90_hora16384_box45','cache/digit_tabletop_box45_inward90_checked_20261016.pt'),
 ('air_six','manip_hora16384_six_objects_gel_rotation','cache/digit_six_objects_inward90_checked_20261017.pt')]:
 for seed in (901,1701,2901):
  jobs.append(dict(tag=tag,config=config,cache=cache,seed=seed,name=f'{tag}_{seed}'))
(out/'plan.json').write_text(json.dumps(jobs,indent=2))
def run(j):
 name=j['name'];command=[str(root/'scripts/python_manip.sh'),str(root/'scripts/revalidate_grasp_cache.py'),'--config',j['config'],'--cache',j['cache'],'--seed',str(j['seed']),'--output',str(out/(name+'.json'))]
 if j['seed']==901:command.append('--retractions')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1');started=time.time()
 with (out/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  print('START',name,proc.pid,flush=True);code=proc.wait()
 result=dict(job=j,exit_code=code,seconds=time.time()-started)
 (out/(name+'.status.json')).write_text(json.dumps(result,indent=2));print('END',name,code,flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
(out/'execution.json').write_text(json.dumps(results,indent=2))
if any(r['exit_code']!=0 for r in results):raise SystemExit('Some validation jobs failed; inspect logs')
