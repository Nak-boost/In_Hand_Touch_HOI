# 距离引导与实际关节角增量控制（2026-10-15）

按用户要求，新的 `manip.yaml` 默认配置采用下列设置；桌面配置继承它们。

| 参数 | 新默认值 | 含义 |
|---|---|---|
| `pose_weight` | 0.05（保留） | `-0.05 * sum((q-initial_q)^2)` |
| `contact_weight` | 0.05（保留） | 至少两路胶面净接触力超过阈值时 +0.05 |
| `fingertip_distance_weight` | 0.1（新增） | 四个胶面参考点接近物体中心的平均得分 |
| `action_reference` | `actual_position`（修改） | 以测得的当前关节角而非上一目标为基准 |

## 距离奖励

令 `d_i = ||p_gel_i - p_object||`，单位为米：

```
reward_distance = 0.1 * mean(clip(0.1 / (0.02 + 4 * d_i), 0, 1))
```

公式与权重参考 Rotating without Seeing 附录 D：<https://arxiv.org/html/2303.10880v4>。针对 DIGIT，参考点选用独立 gel link 原点；官方 CAD 构建器将它置于胶面顶点（`scripts/build_official_digit_asset.py` 中的 `apex`），不是手指关节原点，也不是胶体质心。

这一项使用胶面顶点到物体中心的距离，不是到物体表面的最短距离，不判断实际接触，也不区分靠近方向。得分在距离 <= 2 cm 时饱和；加权后每环境每步在 [0, 0.1]，不是每指各 +0.1。0.1 是有出处的实验起点，未证明是当前系统最优值；不同尺寸需要重新检查其尺度。

物体位置来自仿真，仅参与训练奖励和日志，没有加入学生的触觉/本体观测。新日志：`reward/fingertip_distance`、`contact/gel_object_center_distance_m`。

当前两指接触奖励的判定保留原实现：胶面刚体的净接触力阈值，而不是胶面—目标物体的接触对识别。因此它仍可能计入胶面与其他手部部位的接触；新增距离项没有修复或掩盖这一限制。

## 控制定义

```
smoothed_action = 0.5 * clipped_action + 0.5 * previous_smoothed_action
increment = 0.026 * smoothed_action
target = clamp(q_actual + increment, joint_lower, joint_upper)
```

该定义参考 Visual Dexterity 的 Action space 与公开控制代码：
<https://github.com/Improbable-AI/dexenv/blob/ad9634e9d26cf5555d18728244a58f5b412c1eb2/dexenv/envs/dclaw_base.py#L882>。

每个策略步从实际关节位置重新建立目标；该步内低层 PD 仍跟踪这个固定目标。动作尺度、平滑、PD 增益和力矩限幅保持原设置。关节观测噪声仍只加到策略观测，目标基准使用仿真实际 q；部署使用调用者传入的实测 q。

关节被阻挡时，目标不会因相同增量持续累加。平滑动作真正归零后，目标等于当前角度，不再保留缓存/上一目标的静态预紧力；保持接触需要策略持续输出适当动作。物理关节限位仍然可能被触及。

限位增量损失与 `limit_push_penalty` 同步改为相对本次基准角计算，避免把“旧目标重置回实际位置”误判成被裁掉的动作。它们衡量命令被硬限位裁剪，不衡量电机是否实际完成动作。

## 保存、回放与旧结果

教师/学生检查点均保存 `action_reference`；训练恢复、蒸馏、评估和回放会核对控制模式。部署适配器从检查点读取模式，使用与仿真相同的目标计算函数。

历史配置/检查点缺少 `action_reference` 时解释为 `previous_target`；缺少距离权重时解释为 0。因而旧配置仍可用于历史回放，但不能把旧权重直接续训进新的默认控制方式。历史回放仍遵守原有的源码差异检查。

回放入口还修正为按保存的任务名称选择桌面或手内环境，避免桌面策略错误创建为无桌面的环境。

此修改未改变碰撞模型、接触过滤、失败条件、桌面偏轴/超速权重，也没有重设其他奖励项。正式新实验应从头训练；验证记录见 `../../research/distance_actual_control_20261015/`。
