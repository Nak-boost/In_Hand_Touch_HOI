import json, math
from pathlib import Path
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'tensorboard_summary.json').read_text());status=json.loads((OUT/'run_status.json').read_text());ck=json.loads((OUT/'checkpoints.json').read_text())
rows=[]
for m in [10,30,60,100,150,200,230]:
 v=lambda k:s[k]['stages'][str(m)]
 rows.append(f"|{m-10}–{m} M|{v('episode_rewards/step'):.2f}|{v('episode_lengths/step')*.05:.2f}|{v('rotation/speed'):.3f}|")
results=[json.loads(p.read_text()) for p in sorted(OUT.glob('eval_sphere55_*.json'))]
evalrows=[]
for r in results:
 evalrows.append(f"|{r['checkpoint']} / {r['mode']} / {r['seed']}|{r['checkpoint_step']/1e6:.2f} M|{r['retained_30s']:.1%}|{r['one_turn_retained']:.1%}|{r['mean_turns']:.2f}|{r['frame_means']['speed']:.3f}|")
b=next(r for r in results if r['checkpoint']=='best' and r['mode']=='mean' and r['seed']==215)
f=b['frame_means'];c=b['contact_sources_sampled_every_10_steps'];den=c['sampled_frames'];bodies=b['object_contact_frame_counts_by_body_every_10_steps']
report=f"""# 16384 环境球体 +Z 训练分析（2026-10-17）

分析对象：teacher_sphere55_z_hora16384_pairfilter_20261016。只读分析和独立回放；未改变训练配置、奖励或重启实验。

## 运行状态

2026-10-16 23:27 开始续训，2026-10-17 09:07:39 异常结束，进程返回 -11（SIGSEGV）。累计日志到 233,308,160 环境步；续训运行 {status['elapsed_hours']:.2f} 小时。不是训练完成，亦不是策略“停住”导致进程结束。全部已写出的 TensorBoard 标量和保存模型权重均有限。日志无 Python traceback、CUDA OOM 或 PhysX 容量错误；本进程峰值 7558 MiB，最后整卡 12000/24564 MiB，未触发内存监控阈值。无法仅凭这些信息定位段错误原因；内核日志读取权限不足，无可用 core 文件。

最近 100 轮 Last FPS 调和均值 {status['last100_harmonic_fps']:.0f} 环境步/秒。恢复后的累计 FPS 分子包含旧步数，不能用于启动阶段的吞吐比较。

best.pth 为 217,317,376 步；last.pth 为 196,608,000 步。best 比 last 新，不能将 last 当作中断前最终状态。中断前约 1599 万步没有更晚已保存检查点。HORA 原始每 500 轮保存一次，在此规模相隔约 6554 万步（约 2.7 小时），恢复代价明显。

## 训练趋势

每行取对应 1000 万环境步窗口内已记录标量的均值，不是该单点或独立评估。episode reward/length 是训练器滚动完成回合统计；rotation/speed 是每个 rollout 最后一个控制步的环境均值，非所有 rollout 帧均值。

|窗口|回合回报|回合时长 / 秒|+Z 速度 / rad/s|
|---|---:|---:|---:|
"""+'\n'.join(rows)+f"""

最后 100 轮：回合回报 {s['episode_rewards/step']['last100']:.2f}，平均回合 {s['episode_lengths/step']['last100']*.05:.2f} 秒，上限 30 秒。不存在训练整体重新退化到零转速的证据。目标速度 0.5 rad/s。

## 独立回放

使用原训练目录 source 中的任务、网络和控制代码，校验手模型与碰撞签名；逐一安装 246 个已验证抓取及配套预紧目标。使用保存的域随机化范围，每个 seed 重新抽样。每环境只计第一个回合，最多 600 步/30 秒，终止后不再计入统计；禁用自动重置以检查终止前接触。GPU PhysX、原 CPU 状态管线与物体接触筛选保留，小批量推理在 CPU。mean 为确定性均值动作；sampled 保留高斯采样。所有均为特权教师，不是纯触觉学生。

|检查点 / 动作 / seed|训练步数|保持30秒|保持30秒且净转≥1圈|平均净圈数|有效帧平均轴向速度|
|---|---:|---:|---:|---:|---:|
"""+'\n'.join(evalrows)+f"""

“圈数”为每步旋转增量投影到 +Z 后的有符号累计 / 2π，不能等同于严格纯 Z 轴完成姿态一周。未测试未知形状/尺寸、未见抓取、两轴或实机泛化。

## 关键偏差：转动不等于胶面主导的操作

best / mean / seed215：有效帧平均胶面—物体触点数 {f['object_gel_count']:.3f}；没有 >0.1N 胶面—物体载荷的帧 {f['no_object_gel']:.1%}；至少两个有效胶面的帧 {f['two_object_gels']:.1%}。低于阈值仍可能存在轻触，不能理解为所有这些帧都绝对没有几何接触。

每 10 个控制步额外查询一次接触对（正载荷），物体接触非胶面部位占抽样有效帧 {c['non_gel_object_frames']/den:.1%}；手掌 base_link 为 {bodies.get('base_link',0)/den:.1%}，拇指外壳 {bodies.get('digit_thumb_housing',0)/den:.1%}，食指外壳 {bodies.get('digit_index_housing',0)/den:.1%}。这些统计是接触出现率，不是承重力份额，且各部位可以同时接触。足以确认明显使用了掌部/指节/外壳接触，但不能量化各部位承担多少力。

物体相对初始位置平均移动 {f['displacement']*100:.2f} cm。头 5 秒后通常已远离原来的指尖抓取位置。结合接触对证据，策略更接近包含掌部和指节支撑的手内滚动，而非持续保持多指胶面接触的指尖旋转。对纯触觉重建，接触覆盖与可测局部曲面信息仍不足。

筛选仍然有效：contact 奖励只来自物体—胶面。问题在目标定义：它只给至少两个胶面一个较小可选奖励；没有要求旋转过程必须持续有胶面接触，也未统一惩罚所有非胶面物体接触。当前 housing 惩罚依据外壳净力，不能把它当成所有非胶面承托的专用代价。

best 回放平均每步旋转项 {f['reward_rotation']:+.4f}，胶面接触项 {f['reward_contact']:+.4f}；姿态 {f['reward_pose_penalty']:+.4f}、位置 {f['reward_position_penalty']:+.4f}、偏轴 {f['reward_off_axis_penalty']:+.4f}、无效限位指令 {f['reward_limit_push_penalty']:+.4f}。即使少有胶面接触，也能获得正总回报。接触项约为旋转项的 {f['reward_contact']/f['reward_rotation']:.1%}，这只是实际平均贡献之比，不是奖励梯度因果归因。

## 仍存在的控制退化

best 回放原始动作均值被 [-1,1] 裁剪的分量占 {f['raw_action_clipped']:.1%}；目标增量损失指标 {f['target_lost']:.1%}，目标处在限位附近的关节占 {f['target_at_limit']:.1%}。目标—实测关节平均绝对误差 {f['target_tracking_error']:.4f} rad。不能把目标增量损失说成同等比例的动作数量：它按原实现定义的增量比例计算。未再出现此前 80% 以上目标增量损失的程度，但限位和动作裁剪未解决。

回放最后 5 秒速度下降，见各评估 trace/last5s_speed；不能只凭整个30秒累计约2圈，就断言能无限持续旋转。记录中偏轴平方速度也非零，需要进一步检查运动轨迹是否符合纯轴向旋转目标。

## 建议（本次未实施）

1. 保留当前结果作为“可转动、允许非胶面支撑”的有效基线。不能把进步单独归因于环境数：相较旧实验，训练预算、PPO 超参数、模型安装及控制等也有差异。
2. 先定位 SIGSEGV 并增加异常诊断；恢复时明确 best 新于 last。保存周期建议从500轮缩短至50–100轮，以减少长时间训练丢失，属于检查点管理而非奖励更改。
3. 对触觉重建目标，优先单独对照胶面接触覆盖和非胶面物体接触约束；使用碰撞对而非所有外壳净力。不要未经确认新增悬空失触终止条件（用户此前明确不新增）。也不直接关闭真实非胶面物理碰撞来“强迫”接触。
4. 用60–120秒的专门持续性测试核对后期减速；同时报告实际接触覆盖、轴向/偏轴运动和限位占比。随后再考虑增加对象与尺寸；当前不构成泛化证明。

## 评估进程限制

第一组 best/mean 回放已完成600步并完整写出JSON，随后清理阶段发生段错误（shell 139）。其余进程状态见 evaluation_processes.json；若也在 env.close 发生错误，会在报告补充。清理崩溃不能自动解释原训练中途的 SIGSEGV；已导出的完整轨迹指标与未正常清理必须分别报告。

原始结果：tensorboard_summary.json、scalars.npz、checkpoints.json、run_status.json、eval_*.json 和对应日志。
"""
(OUT/'report.md').write_text(report)
print(OUT/'report.md')
