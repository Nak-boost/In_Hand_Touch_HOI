from pathlib import Path
import json, math, hashlib, datetime, subprocess
P=Path(__file__).resolve().parent
B=P.parent/'current_four_runs_20261019'
names={'air_six':'悬空六物体','air_sphere55':'悬空55mm球','table_sphere55':'桌面55mm球','table_box45':'桌面45mm方块','old_baseline':'旧单球有效策略＋当前缓存'}
def read(p):return json.loads(p.read_text())
def finite(x):
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return not isinstance(x,float) or math.isfinite(x)
def agg(d):
    rows=list(d['per_object'].values());n=sum(r['n'] for r in rows)
    a={'n':n,'steps':d['checkpoint_steps']}
    for k in ['retained_30s','one_turn_retained','mean_turns','mean_return']:
        a[k]=sum(r[k]*r['n'] for r in rows)/n
    ns=sum(r['retained_30s']*r['n'] for r in rows)
    a['late_speed']=sum((r['last5s_speed_survivors'] or 0)*r['n']*r['retained_30s'] for r in rows)/max(ns,1)
    frames=sum(r['mean_seconds']*r['n'] for r in rows)
    for k in ['two_gels','no_gel','gel_count','target_lost','displacement']:
        a[k]=sum(r['frame_means'][k]*r['mean_seconds']*r['n'] for r in rows)/frames
    return a
s=read(P/'training_summary.json')
records=read(P/'evaluation_processes.json')+read(P/'additional_processes.json')
assert len(records)==10
evaluations={};summary={}
for r in records:
    assert r['result_saved'] and r['returncode']==0,r
    group,ck=r['group'],r['checkpoint'];key=group+'_'+ck
    d=read(P/f'eval_{key}_215.json');assert finite(d)
    assert d['checkpoint_sha256']==hashlib.sha256((P/f'{key}.pth').read_bytes()).hexdigest()
    assert sum(x['n'] for x in d['per_object'].values())==(384 if group=='air_six' else 64)
    evaluations[key]=d;summary[key]=agg(d)
assert all(not s[g]['nonfinite_tags'] for g in names if g!='old_baseline')
(P/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
status={'time':datetime.datetime.now().astimezone().isoformat(),'runs':{}}
for g in names:
    if g=='old_baseline':continue
    status['runs'][g]=read(B/(g+'_monitor/status.json'))
status['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
(P/'status_snapshot_end.json').write_text(json.dumps(status,indent=2))

lines=['# 最新四组实验分析：2026-10-20（Asia/Shanghai）','',
'对象是10月19日重新从头训练的四组，不是10月17日旧实验的续训。三组仍在运行，桌面球因监测显存保护中止；均未完成15亿步预算。本次只做独立分析、冻结检查点和小规模回放，没有修改或停止训练。','',
'| 任务 | GPU | 日志取样步数M | last步数M | best步数M | 状态 |','|---|---:|---:|---:|---:|---|']
for i,g in enumerate(['air_six','air_sphere55','table_sphere55','table_box45']):
    v=s[g];c=v['checkpoints'];state='已停止：memory_pressure' if g=='table_sphere55' else '运行中'
    lines.append(f"| {names[g]} | {i} | {v['latest_steps']/1e6:.2f} | {c['last']['agent_steps']/1e6:.2f} | {c['best']['agent_steps']/1e6:.2f} | {state} |")
lines += ['',
'本轮周期保存间隔为500轮，即65,536,000环境步。因此last可能落后实时进度数千万步；桌面球的best=176.95M反而比last=131.07M新。本文明确区分训练日志与实际回放检查点。',
'',
'每个单物体检查点用64个缓存起点，六物体每种64个、共384个，seed215；确定性均值动作，沿用冻结源码、配置、物理随机化、原始600步/30秒首回合。终止后帧不计入指标。所有新评估均正常完成并清理。',
'',
'| 任务/检查点 | 步数M | 30秒无任务失败 | 无失败且净转≥1圈 | 平均净圈数 | 末5秒净轴向速度 rad/s | ≥2胶面帧 | 零胶面帧 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for g in ['air_six','air_sphere55','table_sphere55','table_box45']:
    for ck in ['last','best']+(['selected'] if g=='table_box45' else []):
        r=summary[g+'_'+ck];label={'last':'last','best':'best（无有效回合）' if g=='table_box45' else 'best','selected':'周期快照中回报较高者'}[ck]
        lines.append(f"| {names[g]} {label} | {r['steps']/1e6:.2f} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.3f} | {r['late_speed']:.4f} | {r['two_gels']:.1%} | {r['no_gel']:.1%} |")
lines += ['',
'桌面任务的“无失败”只表示未越出操作区；物体独自静置桌面同样可以达到100%，不代表被手抓住或持续操作。桌面失触终止已关闭，失触只记录。圈数是目标轴的有符号角增量累计，不是严格纯轴旋转认证；末段净速度接近零不排除来回摆动。',
'',
'六物体最新last逐物体结果：','',
'| 物体 | 无失败比例 | 平均净圈数 | ≥2胶面帧 | 零胶面帧 | 目标增量裁剪损失 |','|---|---:|---:|---:|---:|---:|']
for o,r in evaluations['air_six_last']['per_object'].items():
    f=r['frame_means'];lines.append(f"| {o} | {r['retained_30s']:.1%} | {r['mean_turns']:.3f} | {f['two_gels']:.1%} | {f['no_gel']:.1%} | {f['target_lost']:.1%} |")
lines += ['', '训练各组最近1000万步：','',
'| 任务 | 回合回报 | 时长s | 轴向速度rad/s | 有效胶面数 | 目标增量裁剪损失 | 姿态代价/步 |','|---|---:|---:|---:|---:|---:|---:|']
for g in ['air_six','air_sphere55','table_sphere55','table_box45']:
    w=s[g]['windows']['last10M'];lines.append(f"| {names[g]} | {w['episode_rewards/step']:.2f} | {w['episode_lengths/step']*.05:.2f} | {w['rotation/speed']:.4f} | {w['contact/task_gel_count']:.4f} | {w['control/target_increment_lost']:.1%} | {w['reward/pose_penalty']:.4f} |")
lines += ['',
'桌面方块退化最明显：90–100M窗口回报约−82.9、胶面数0.102；390–400M约−144.9、胶面数0.051；当前约−390.8、胶面数0.00021。当前失触阈值已越过的帧占比约93.4%，平均连续失触时长15.08秒，目标增量损失47.9%。30秒回合完整结束不能当作操作成功。姿态代价−0.444/步是主要负项，旋转项约−0.0047/步；这说明退化状态伴随远离初始姿态，不能据此直接认定删掉姿态代价就能解决。',
'',
'最新方块回放还显示：零胶面帧99.5%，原始动作分量被[-1,1]裁剪的比例90.0%，目标在关节限位附近的比例64.2%，请求增量被关节限位裁掉的损失47.6%。物体角度在5秒后基本不变，非胶面手部—物体接触采样帧仅约0.05%，并非仅从胶面转移到外壳继续操作。上述动作裁剪、目标贴限位和目标增量损失是三个不同统计量。',
'',
'本次确认一个检查点选择问题：table_box45 best.pth只有131,072步/epoch1，best_rewards=0。AverageScalarMeter在没有完成回合时mean初始化为0；PPO没有检查有效回合数量便参与best比较。之后负回报的已训练策略无法超过这个虚假的0分。日志开头回报和回合长度均为0支持此判断。它会误导选模型，但不会直接改变PPO梯度，也不能解释全部行为退化。已额外评估现有周期快照中记录回报最高的131.07M（−51.65）权重；未改训练代码。后续建议仅在完成回合数>0时参与best评选。',
'',
'桌面球停止原因：北京时间10月19日23:14:43，监测记录memory_pressure；整卡23803/24564MiB，差值761MiB低于2048MiB阈值，但本训练进程为6904MiB。日志是SIGINT/KeyboardInterrupt，退出−2，不能称为训练自己CUDA OOM或destroy_sim段错误。最后179.70M，最新保存best176.95M，相差2.75M步；max_restarts=0所以没有自动恢复。',
'',
'与上一轮比较时的限制：本轮六物体策略网络从512/256/128改为256/256/128，特权编码器从256/128/8改为128/64/8，同时更换了复验抓取缓存并从头训练。手模型资产哈希、碰撞签名和actual_position控制基准与上一轮一致。不是单变量对照，不能把本轮变差直接归因于网络缩小或缓存变动。',
'',
'补充对照：旧单球488.24M有效权重在当前robust缓存上评估，仍使用该权重原网络与原控制配置，64个初始状态：','',
'| 权重＋起点 | 无失败比例 | 无失败且≥1圈 | 平均净圈数 | 末5秒净速度rad/s | ≥2胶面帧 |','|---|---:|---:|---:|---:|---:|']
r=summary['old_baseline_best'];lines.append(f"| 旧单球best＋当前缓存 | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.3f} | {r['late_speed']:.4f} | {r['two_gels']:.1%} |")
lines += ['',
'这项补充对照用于检查已有效策略在当前起点上是否仍可用，不能单独分离网络容量、随机种子、探索过程与训练轨迹的影响。上一轮71.9%成功率来自旧缓存两个seed共128次，本次64次不能当作严格同一评估集的提升或下降。',
'',
'实测旧策略在当前缓存仍有43/64=67.2%保持且满一圈、平均2.05圈、末5秒0.390rad/s。因此当前缓存与手模型下确实存在可用的单球持续旋转策略，不能把本轮失败笼统归因于物理上转不动。本轮单球没有复现这项能力，应优先定位学习过程与初始化/配置差异，而不是再次大幅修改碰撞几何。',
'',
'后续优先级建议：先修正best有效回合判定，保留周期快照并依据独立旋转评估选模型；把旧有效单球策略作为可用性对照，再逐步加入新尺寸/形状；桌面方块优先诊断失触后回撤与再接触是否发生，以及限位外推，避免只追加训练步数。奖励、探索和示范/课程策略应各自对照，不能用单一曲线认定唯一根因。本次不执行这些改动或重启。',
'',
'评估是特权教师仿真结果，不是纯触觉学生、实机或未知物体泛化。胶面接触按目标物体碰撞对和0.1N阈值判定；非胶面手部接触统计已排除桌面。未重复相同检查点/同seed测试，不提供未经估计的置信区间。',
'',
'证据：training_summary.json、*_scalars.npz、eval_*.json、evaluation_summary.json、evaluation_processes.json、additional_processes.json、status_snapshot_start/end.json、固定pth快照、evaluate.py及本报告生成脚本。'
]
(P/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2));print('Verified and saved',P/'report.md')
