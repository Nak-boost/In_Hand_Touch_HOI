from pathlib import Path
import json,datetime
from zoneinfo import ZoneInfo
P=Path(__file__).resolve().parent;OLD=Path('/root/In_Hand_Reconstruction/research/gel_rotation_results_20261018')
training=json.loads((P/'training_summary.json').read_text());previous=json.loads((OLD/'training_summary.json').read_text());prior_pool=json.loads((OLD/'single_pooled.json').read_text())
pooled={}
for ck in ['best','last']:
 rows=[json.loads((P/f'eval_single_{ck}_{seed}.json').read_text())['per_object']['sphere55'] for seed in [215,216]]
 total=sum(r['n'] for r in rows);frames=sum(r['n']*r['mean_seconds'] for r in rows);surv=sum(r['n']*r['retained_30s'] for r in rows)
 out={k:sum(r[k]*r['n'] for r in rows)/total for k in ['retained_30s','one_turn_retained','mean_turns','mean_seconds','mean_return']}
 out.update(n=total,checkpoint_steps=training['single']['checkpoints'][ck]['agent_steps'],frame_means={k:sum(r['frame_means'][k]*r['n']*r['mean_seconds'] for r in rows)/frames for k in rows[0]['frame_means']},last5s_speed_survivors=sum(r['last5s_speed_survivors']*r['n']*r['retained_30s'] for r in rows)/surv,last5s_abs_speed_lt_005_survivors=sum(r['last5s_abs_speed_lt_005_survivors']*r['n']*r['retained_30s'] for r in rows)/surv)
 pooled[ck]=out
(P/'single_pooled.json').write_text(json.dumps(pooled,indent=2))
multi=json.loads((P/'eval_multi_last_215.json').read_text())['per_object'];mbest=json.loads((P/'eval_multi_best_215.json').read_text())['per_object'];mprev=json.loads((OLD/'eval_multi_last_215.json').read_text())['per_object']
lines=['# 手内旋转训练更新分析（2026-10-18，America/Chicago）','', '取样起始时间见 analysis_context.json（约07:28 CDT）。本报告仅分析GPU 0/1两组悬空手内旋转，与上次分析比较；不包含另两组桌面训练。','',
'## 运行状态','', '| 组别 | 上次训练步数 | 当前取样步数 | 最新保存检查点 | 状态 |','|---|---:|---:|---:|---|']
for g,label,state in [('single','GPU 1 单球55mm','已停止'),('multi','GPU 0 六规格混合','仍在运行')]:
 s=training[g];lines.append(f"| {label} | {previous[g]['latest_steps']/1e6:.2f} M | {s['latest_steps']/1e6:.2f} M | {s['checkpoints']['last']['agent_steps']/1e6:.2f} M | {state} |")
lines+=['','单球体于2026-10-18 01:59 CDT停止：旧监测记录reason=memory_pressure，整卡24564MiB中已用22788MiB，监测差值1776MiB低于2048MiB阈值。训练进程7748MiB。先收到SIGINT并记录KeyboardInterrupt，随后destroy_sim清理阶段SIGSEGV，退出-11。不是达到15亿步预算；也不能把清理崩溃解释为训练中最初的停止原因。最后613416960步，最新完整保存609484800步，相差3932160步。','',
'此前取消监测是对GPU 0混合组实施；GPU 1仍沿用原监测。本次仅分析，没有重启、停止或改动任何训练，没有新增监测。','',
'## 单球体：部分恢复，但未回到历史最佳','', '| 回放检查点 | 保持30秒 | 保持且净转≥1圈 | 平均净圈数 | ≥2有效胶面帧比例 | 存活回合末5秒速度 |','|---|---:|---:|---:|---:|---:|']
for label,r in [('历史最佳 488.24M（相同文件，复用此前验证）',pooled['best']),('上次最新 570.16M',prior_pool['last']),('此次最新 609.48M',pooled['last'])]:
 lines.append(f"| {label} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.3f} | {r['frame_means']['two_gels']:.1%} | {r['last5s_speed_survivors']:.3f} rad/s |")
lines+=['','每个检查点两个随机种子215/216，各64个初始抓取，合计128次。best文件SHA256与上次相同，未重复运行同一评估；最新检查点已重新回放。相同seed与检查点间使用相同抓取选择/物理随机化初始化，但GPU物理不保证位级确定性。','',
'最近1000万步训练窗口的回报18.82→101.24、速度0.217→0.247rad/s、有效胶面数1.65→1.90、目标增量损失20.3%→16.0%。这些变化支持“有部分恢复”；历史最佳仍为488.24M，不能说最新已经超过历史最佳，也不能把上次退化当作之后一直单调恶化。','',
'## 六物体：回报改善与持续旋转分开看','', '| 物体 | 上次平均圈数 | 当前平均圈数 | 当前保持30秒 | 当前保持且≥1圈 | ≥2胶面帧比例 | 末5秒速度 rad/s |','|---|---:|---:|---:|---:|---:|---:|']
for name,r in multi.items():lines.append(f"| {name} | {mprev[name]['mean_turns']:.3f} | {r['mean_turns']:.3f} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['frame_means']['two_gels']:.1%} | {r['last5s_speed_survivors']:.4f} |")
lines+=['','当前last=596.38M；新best=595.20M，训练器最高滚动回报由4.64升至10.11。best按最近100个完成回合的回报选取，并非独立旋转成功率。新best独立回放如下：','', '| 物体 | best平均圈数 | best保持且≥1圈 | best末5秒速度 |','|---|---:|---:|---:|']
for name,r in mbest.items():lines.append(f"| {name} | {r['mean_turns']:.3f} | {r['one_turn_retained']:.1%} | {r['last5s_speed_survivors']:.4f} |")
lines+=['','最近1000万步窗口：平均回报-32.84→-10.87；平均回合时长28.75→29.21秒；轴向速度0.0862→0.0836rad/s；有效胶面数1.119→1.111。旋转奖励0.1301→0.1263，失败代价-0.0350→-0.0211，姿态代价-0.0961→-0.0723。回报变好主要伴随代价减少，未显示持续旋转提升。','',
'## 建议','',
'1. 单球体仍以488.24M历史最佳作为稳定对照起点，并保留此次609.48M恢复后的权重用于比较。下一步更值得做固定评估集上的短分支对照，而不是单凭训练回报挑选模型。','2. 六物体先以“末段持续旋转+逐物体成功率”为判据。若回放仍显示前段调姿后停住，增加相同训练预算的收益有限；优先测试从单球有效策略逐渐加入一种尺寸/形状的课程，再扩展六种。','3. 本次证据不足以确定唯一根因。尚不能认定改学习率、姿态惩罚、接触奖励或形状输入中的某一项必然解决；应逐项对照。','',
'## 评估范围及证据','',
'确定性均值动作的特权教师，沿原+Z目标的角增量累计，最长30秒，只统计第一个回合和终止前帧。不是纯触觉学生或实机结果，也不是严格纯轴旋转/未知物体泛化证明。≥2胶面帧比例指各胶面与目标物体法向载荷超过0.1N，并非接触面积或重建覆盖率。','',
'最新单球每seed64次，两seed共128次；混合best/last每物体64次、每检查点384次。沿用上次冻结源代码、物理配置、缓存和验收口径，检查模型/碰撞/控制签名。冻结pth快照避免正在训练覆盖文件。所有读取的TensorBoard标量有限。独立回放过程及是否正常清理见evaluation_processes.json。','',
'原始结果：training_summary.json、single_pooled.json、eval_*.json、*_scalars.npz、四个固定pth快照、evaluation_processes.json。上次报告：../gel_rotation_results_20261018/report.md。']
(P/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(pooled,indent=2));print('REPORT',P/'report.md')
