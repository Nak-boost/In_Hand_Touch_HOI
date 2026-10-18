from pathlib import Path
import json,datetime,numpy as np
P=Path(__file__).resolve().parent
training=json.loads((P/'training_summary.json').read_text());pooled={}
for ck in ['best','last']:
 rows=[json.loads((P/f'eval_single_{ck}_{seed}.json').read_text())['per_object']['sphere55'] for seed in [215,216]]
 total=sum(r['n'] for r in rows);frames=sum(r['n']*r['mean_seconds'] for r in rows);surv=sum(r['n']*r['retained_30s'] for r in rows)
 out={k:sum(r[k]*r['n'] for r in rows)/total for k in ['retained_30s','one_turn_retained','mean_turns','mean_seconds','mean_return']}
 out.update(n=total,checkpoint_steps=training['single']['checkpoints'][ck]['agent_steps'],frame_means={k:sum(r['frame_means'][k]*r['n']*r['mean_seconds'] for r in rows)/frames for k in rows[0]['frame_means']},last5s_speed_survivors=sum(r['last5s_speed_survivors']*r['n']*r['retained_30s'] for r in rows)/surv,last5s_abs_speed_lt_005_survivors=sum(r['last5s_abs_speed_lt_005_survivors']*r['n']*r['retained_30s'] for r in rows)/surv)
 pooled[ck]=out
(P/'single_pooled.json').write_text(json.dumps(pooled,indent=2));print(json.dumps(pooled,indent=2))
multi=json.loads((P/'eval_multi_last_215.json').read_text())['per_object'];mbest=json.loads((P/'eval_multi_best_215.json').read_text())['per_object']
lines=['# 两组胶面关联旋转训练分析（2026-10-18）','', '分析快照约为北京时间 13:23。训练继续运行；本次仅复制检查点、读取日志并执行独立回放，未修改奖励、控制、碰撞模型、训练预算或监测设置。','',
'## 结论','', '单球体组曾学到兼顾旋转与胶面接触的策略，但最新权重明显退化；六物体混合组主要学到一次性调姿后保持，尚未学会持续旋转。增加训练时长不保证改善，当前不宜只根据进程存活或回合长度判断成功。','',
'## 训练进度与窗口统计','', '| 组别 | 快照训练步数 | 最新保存 | 历史最佳保存 | 预算完成度 |','|---|---:|---:|---:|---:|']
for group,label in [('single','55mm单球'),('multi','六规格混合')]:
 s=training[group];lines.append(f"| {label} | {s['latest_steps']/1e6:.2f} M | {s['checkpoints']['last']['agent_steps']/1e6:.2f} M | {s['checkpoints']['best']['agent_steps']/1e6:.2f} M | {s['latest_steps']/1.5e9:.1%} |")
lines+=['','每组均为16384环境，PPO教师，预算15亿步。best按训练器最近100个完成回合的平均回报选出，不是独立评估结果。所有已读取TensorBoard标量有限。','',
'| 组别／窗口 | 回合回报 | 平均回合秒数 | 轴向速度 rad/s | 有效胶面数 | 目标增量损失 |','|---|---:|---:|---:|---:|']
for group,label in [('single','单球'),('multi','混合')]:
 for window in ['390-400M','last10M']:
  w=training[group]['windows'][window];lines.append(f"| {label} {window} | {w['episode_rewards/step']:.2f} | {w['episode_lengths/step']*.05:.2f} | {w['rotation/speed']:.3f} | {w['contact/task_gel_count']:.2f} | {w['control/target_increment_lost']:.1%} |")
lines+=['','奖励/控制诊断按整个rollout平均；rotation/contact类指标是每个rollout最后一个控制步的环境均值。回合回报/长度是滚动完成回合统计；episode/failed不是整回合失败率。因此成功率采用下面的独立回放。','',
'单球体480–500M窗口回报约210–212，速度约0.40–0.38 rad/s；510M以后明显下降，最后1000万步回报18.82、速度0.217。实际学习率已经被KL自适应降到约6.7e-5，不是仍以初始0.005训练。存在退化证据，但不能仅凭曲线认定是学习率、熵下降或奖励权重中的某一项导致。','',
'## 独立回放协议','', '使用每次训练目录中保存的source与原配置，检查手模型/碰撞/控制签名；复制best/last后固定检查点，避免训练中覆盖造成比较漂移。确定性均值动作的特权教师，CPU小批量推理、GPU PhysX、现有胶面—物体接触筛选；每次只统计第一个回合，最多600控制步/30秒，终止后的帧剔除。','',
'单球体每个检查点测试两个seed（215/216），每seed从原缓存无放回抽64个抓取，共128次；同seed的best/last共享抓取选择和随机化初始化。混合组每个检查点用seed215，每种物体64次，共384次。不同训练组的随机物理参数不构成严格逐样本配对。仍是训练物体与训练抓取分布，不是未知物体泛化。','',
'圈数是每步旋转增量投影到目标+Z轴的有符号累计/2π，不等于严格纯Z轴完整姿态回转。教师使用真实物体状态等特权输入，不代表纯触觉学生或实机已经达到同样表现。非胶面接触每10步抽样，其出现率不是承重力比例。','',
'## 单球体：最佳与最新','', '| 检查点 | 保持30秒 | 保持且净转≥1圈 | 平均净圈数 | ≥2胶面帧占比 | 末5秒速度 rad/s |','|---|---:|---:|---:|---:|---:|']
for ck,label in [('best','best 488.24M'),('last','last 570.16M')]:
 r=pooled[ck];lines.append(f"| {label} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.2f} | {r['frame_means']['two_gels']:.1%} | {r['last5s_speed_survivors']:.3f} |")
lines+=['','末5秒速度仅统计存活满30秒者。最佳检查点仍有约38%的存活回合末5秒净轴向速度绝对值低于0.05rad/s，说明仍有明显个体停滞；最新该比例更高，见single_pooled.json。该指标可能包含正反抵消，不代表全部关节静止。','',
'与此前无胶面门控参考回放相比，最佳检查点≥2胶面帧占比由约6.6%提高到约79%，物体平均偏移由约8.5cm降到约2.8cm。这是有价值的结果，但预算和评估抽样不同，不能把全部差异严格归因于门控奖励。最佳的非胶面接触出现率仍约88–89%，以拇指外壳等为主，可与胶面接触同时发生；胶面接触改善不等于完全消除外壳参与，也不等于已证明表面重建覆盖率。','',
'## 六物体：最新权重逐物体结果','', '| 物体 | 保持30秒 | 保持且净转≥1圈 | 平均净圈数 | ≥2胶面帧占比 | 末5秒速度 rad/s |','|---|---:|---:|---:|---:|---:|']
for name,r in multi.items():lines.append(f"| {name} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.3f} | {r['frame_means']['two_gels']:.1%} | {r['last5s_speed_survivors']:.4f} |")
lines+=['','六类均主要在前5–10秒转动，后半程累计角度几乎不增长；存活回合末5秒有约95–100%的净轴向速度绝对值低于0.05rad/s。早期best（84.54M）也只有sphere55的1/64个回合保持且达到1圈，其余五类为0，并非曾经学会后最新检查点才全部忘记。','',
'混合组最近目标增量损失约5.3%，不能把整体停转主要归因为此前80%以上限位饱和。球和胶囊回放损失约1.7–3.2%，方块约10.8–12.9%；各类停转都存在。它更符合从初始状态转到稳定姿态后不再形成循环换指步态。','',
'方块的多胶面接触较弱，平均位置偏移约5.8/6.8cm；全组非胶面接触出现率约93–95%。风险、接触约束、重抓取和持续旋转之间的优化尚未建立。单球体同为390–400M时速度约0.444rad/s，混合约0.094；但混合每物体仅约1/6样本，且形状难度不同，不是等量每物体数据的比较。','',
'## 建议与未验证假设','',
'1. 保留单球体488.24M快照作为后续对照起点，不能用last替代best。用固定回放成功率、多胶面比例、末段转速联合选择模型，而非仅最近100个训练回合的最高回报。','2. 单球体优先做从best恢复的短对照，测试更保守的更新强度（例如降低学习率上限或KL目标，逐项改变）。当前实际LR已很低，不能预先保证继续降低会解决；需同时观察长期旋转与是否只剩稳抓。','3. 多物体优先尝试从单球体有效策略开始，先加入一种相近尺寸/形状，再扩展到六规格，逐物体验证。直接延长当前混合策略预算的收益存疑，但本次未停止训练。','4. 只有课程/更新对照不足时，再分别测试姿态约束、失败代价和持续旋转奖励等调整。不能凭当前结果认定唯一根因，也不建议同时修改一组奖励参数。','5. 代码核对发现教师特权状态提供尺寸、质量、摩擦、质心等，但无显式形状类别；sphere55与box55的三维尺寸数值相同。状态与接触反馈仍可提供区分线索，此事实不证明混合失败由不可辨识导致，后续可单独检验形状先验是否有帮助。','',
'## 性能及验证限制','',
'近100轮Last FPS约单球7128、混合5699环境步/秒。采样侧仍占绝大多数耗时；最近窗口接触对读取约占环境步耗时的单球25%、混合22%，并非99%。共享GPU/CPU负载波动明显，不能把当前吞吐当成独占机器基准。','',
'六次回放（单球4、混合2）均完整导出结果且正常退出，见evaluation_processes.json与recheck_processes.json。这只说明本次短评估没有重现清理崩溃，不证明长期仿真没有崩溃风险。没有对现有训练添加监测或自动中断。','',
'原始证据：training_summary.json、single_pooled.json、*_scalars.npz、eval_*.json、固定的四个pth快照及对应日志。']
(P/'report.md').write_text('\n'.join(lines)+'\n')
