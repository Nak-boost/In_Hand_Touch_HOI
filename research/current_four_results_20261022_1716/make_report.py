from pathlib import Path
import json,math,hashlib,datetime
from zoneinfo import ZoneInfo
P=Path(__file__).resolve().parent;OLD=P.parent/'current_four_results_20261020_1725';B=P.parent/'current_four_runs_20261019'
def read(p):return json.loads(p.read_text())
def finite(x):
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return not isinstance(x,float) or math.isfinite(x)
def agg(d):
    rows=list(d['per_object'].values());n=sum(r['n'] for r in rows);a={'n':n,'steps':d['checkpoint_steps']}
    for k in ['retained_30s','one_turn_retained','mean_turns','mean_return']:
        a[k]=sum(r[k]*r['n'] for r in rows)/n
    ns=sum(r['retained_30s']*r['n'] for r in rows)
    a['late_speed']=sum((r['last5s_speed_survivors'] or 0)*r['n']*r['retained_30s'] for r in rows)/max(ns,1)
    frames=sum(r['mean_seconds']*r['n'] for r in rows)
    for k in ['two_gels','no_gel','gel_count','target_lost','raw_action_clipped','target_at_limit']:
        a[k]=sum(r['frame_means'][k]*r['mean_seconds']*r['n'] for r in rows)/frames
    return a
s=read(P/'training_summary.json');prev=read(OLD/'training_summary.json');prior_eval=read(OLD/'evaluation_summary.json');temporal=read(P/'temporal_summary.json')
names={'air_six':'悬空六物体','air_sphere55':'悬空单球','table_sphere55':'桌面球','table_box45':'桌面方块'}
records=read(P/'evaluation_processes.json')+read(P/'additional_processes.json');assert len(records)==9
summary={};evals={}
for r in records:
    assert r['result_saved'] and r.get('returncode',0)==0
    g,ck=r['group'],r['checkpoint'];key=g+'_'+ck;d=read(P/f'eval_{key}_215.json');assert finite(d)
    assert d['checkpoint_sha256']==hashlib.sha256((P/f'{key}.pth').read_bytes()).hexdigest()
    assert sum(v['n'] for v in d['per_object'].values())==(384 if g=='air_six' else 64)
    evals[key]=d;summary[key]=agg(d)
assert all(not s[g]['nonfinite_tags'] for g in names)
assert len(temporal)==5
(P/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
status={g:read(B/(g+'_monitor/status.json')) for g in names};(P/'status_snapshot.json').write_text(json.dumps(status,indent=2))
lines=['# 最新训练结果：2026-10-22（Asia/Shanghai）','',
'结论：三组完成约15亿步预算，但最终权重仍没有可靠的持续旋转。悬空单球相对10月20日有所恢复，改善主要是初段转动和保持；六物体基本停滞；桌面方块持续严重退化。桌面球仍是此前被显存保护中断的旧结果。',
'',
'| 任务 | 最终日志步数M | last步数M | best步数M | 状态与结束时间（北京时间） |','|---|---:|---:|---:|---|']
for g in names:
    a=status[g]['attempts'][-1];dt=datetime.datetime.fromtimestamp(a['finished'],ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S');v=s[g]
    st='预算完成，清理崩溃' if a.get('heartbeat',{}).get('training_complete') else '显存保护中断'
    lines.append(f"| {names[g]} | {v['latest_steps']/1e6:.2f} | {v['checkpoints']['last']['agent_steps']/1e6:.2f} | {v['checkpoints']['best']['agent_steps']/1e6:.2f} | {st}，{dt} |")
lines += ['',
'三个完成任务的日志均有max steps achieved；心跳training_complete=true，之后phase=destroy_sim、退出−11。最终last已保存并核验可读，步数为1,500,119,040，略超预算源自整批采样。此次不是中途失败或最后只有14.41亿步周期快照。当前无这四组训练进程在运行。本次没有重启、修改、停止训练。',
'',
'独立回放：每物体64个缓存起点，六物体共384次，seed215、最长30秒、确定性教师均值动作。未变的best及桌面球按SHA256确认后复用10月20日评估；三个新last、单球新best和旧有效策略对照重新评估并记录完整接触时序。所有新回放均正常退出。',
'',
'| 权重 | 无失败30秒 | 无失败且净转≥1圈 | 平均净圈数 | 末5秒净轴向速度rad/s | ≥2胶面帧 | 零胶面帧 |','|---|---:|---:|---:|---:|---:|---:|']
for key,label in [('air_six_last','六物体最终15亿'),('air_six_best','六物体best 1.69亿'),('air_sphere55_last','单球最终15亿'),('air_sphere55_best','单球best 9.14亿'),('table_sphere55_best','桌面球best 1.77亿（未更新）'),('table_box45_last','桌面方块最终15亿'),('old_baseline_best','上一轮有效单球＋当前缓存')]:
    r=summary[key];lines.append(f"| {label} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.3f} | {r['late_speed']:.4f} | {r['two_gels']:.1%} | {r['no_gel']:.1%} |")
lines += ['',
'桌面无失败只代表不越界，物体自己静置也可达到100%。净速度接近零不排除局部往复。教师带物体特权状态，以上不代表学生、实机或未知物体泛化表现。单个seed的64次不能被解释为精确的总体成功率。',
'',
'与10月20日同任务last比较：','',
'| 任务 | 平均圈数：上次→最终 | 无失败比例：上次→最终 | ≥2胶面帧：上次→最终 |','|---|---:|---:|---:|']
for g in ['air_six','air_sphere55','table_box45']:
    a=prior_eval[g+'_last'];b=summary[g+'_last'];lines.append(f"| {names[g]} | {a['mean_turns']:.3f}→{b['mean_turns']:.3f} | {a['retained_30s']:.1%}→{b['retained_30s']:.1%} | {a['two_gels']:.1%}→{b['two_gels']:.1%} |")
lines += ['',
'关于“没有学会释放、回撤、重新捏合”的时序证据：下面仅统计各自存活到30秒的回合，区间圈数为增量，因此与全体回合平均总圈数不应直接相加比较。',
'',
'| 策略/物体 | 0–10秒增量圈数 | 10–20秒增量圈数 | 20–30秒增量圈数 | 20–30秒零胶面比例 |','|---|---:|---:|---:|---:|']
for key,title in [('timeline_air_sphere55_last_215','最终单球'),('timeline_air_sphere55_best_215','本轮单球best'),('timeline_old_baseline_best_215','旧有效单球'),('timeline_table_box45_last_215','最终桌面方块')]:
    r=next(iter(temporal[key].values()));v=r['segments'];lines.append(f"| {title} | {v['0-10s']['net_turns_survivors']:.4f} | {v['10-20s']['net_turns_survivors']:.4f} | {v['20-30s']['net_turns_survivors']:.4f} | {v['20-30s']['no_gel_survivor_frames']:.1%} |")
lines += ['',
'再接触诊断定义：同一胶面先连续接触≥0.2秒，再连续无接触≥0.2秒，再连续接触≥0.2秒；只计10秒后的事件，且存在完整后续1秒有效回放。观察该1秒物体净正转是否≥10°。这是二值接触时序的诊断阈值，不是人工标注的完整换指动作；不保证主动回撤，也不能把物体转动因果归于该手指。',
'',
'| 策略 | 10秒后稳定再接触事件 | 随后1秒净正转≥10°的事件 |','|---|---:|---:|']
for key,title in [('timeline_air_six_last_215','最终六物体（6×64）'),('timeline_air_sphere55_last_215','最终单球（64）'),('timeline_air_sphere55_best_215','本轮单球best（64）'),('timeline_table_box45_last_215','最终桌面方块（64）'),('timeline_old_baseline_best_215','旧有效单球（64）')]:
    rows=temporal[key].values();a=sum(r['after10s_sustained_contact_returns']['events'] for r in rows);b=sum(r['after10s_sustained_contact_returns']['events_with_next1s_net_ge10deg'] for r in rows);lines.append(f'| {title} | {a} | {b} |')
lines += ['',
'结论应表述为“没有形成有效、可重复的接触—推动循环”，不能笼统说从未释放或从未重新接触。最终单球确实存在接触断开后恢复的事件，但后续没有达到上述正转阈值；六物体绝大多数再接触同样没有伴随明显正转。旧有效策略在相同任务与当前缓存上，三个时间区间均能产生持续进展。抓取物理可行性不是全部问题，下一步应诊断释放后的重新定位是否有效，而不只是增加接触次数奖励。',
'',
'训练日志最近1000万步对比（包含采样动作，与均值动作回放不是同一测量）：','',
'| 任务 | 回合回报：上次→最终 | 有效胶面数：上次→最终 | 目标增量损失：上次→最终 |','|---|---:|---:|---:|']
for g in ['air_six','air_sphere55','table_box45']:
    a=prev[g]['windows']['last10M'];b=s[g]['windows']['last10M'];lines.append(f"| {names[g]} | {a['episode_rewards/step']:.2f}→{b['episode_rewards/step']:.2f} | {a['contact/task_gel_count']:.4f}→{b['contact/task_gel_count']:.4f} | {a['control/target_increment_lost']:.1%}→{b['control/target_increment_lost']:.1%} |")
lines += ['',
'桌面方块的失触与限位退化仍在加重；不能以完整30秒回合当作成功。六物体best仍停在1.69亿步，此后训练未提升滚动回报最佳。单球best更新到9.14亿步、训练滚动最高回报62.04，但独立回放仍未满圈；训练reward-best与任务成功并不等价。',
'',
'桌面方块best选择问题未改变：它仍是第1轮0.131M检查点，无完成回合时统计器默认0分被当作best，之后负回报策略无法替换。此问题影响选模型，不直接改变PPO梯度。本次不修改代码。桌面球也没有自动续训，停止原因仍为10月19日整卡显存压力触发SIGINT，非新发生的清理段错误。',
'',
'建议：本轮15亿步预算已给出充分的停滞证据，不把同配置加长训练作为第一选择。先以旧有效单球权重为参照，验证从它开始逐步扩展物体的训练；针对循环能力，用完整“释放—重新定位—再接触—后续正转”片段的示范或过程状态课程做单独对照。接触奖励不能只追求接触次数；评价应包含后半段净旋转及再接触后的进展。best保存的有效回合判定应单独修正。以上均为建议，本次只分析。',
'',
'证据：training_summary.json、evaluation_summary.json、temporal_summary.json、timeline_*.npz（逐步q、胶面接触、物体净角度、胶面到物体中心距离和有效帧掩码）、eval_*.json、evaluation_processes.json、additional_processes.json、status_snapshot.json及冻结权重/分析脚本。所有读取标量有限，检查点哈希和模拟模型/控制签名通过核验。'
]
(P/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2));print('Verified and wrote',P/'report.md')
