# B 实验：正向旋转奖励关联胶面接触

配置：`manip_hora16384_sphere55_gel_rotation`，继承原 `manip_hora16384_sphere55`。解析后的配置只增加 `task.env.manipulation.rotation_contact_multipliers: [0.1, 0.5, 1.0]`。

令 `r_dir` 为原有方向旋转奖励（含 rotation_weight），`n` 为当前控制步经过碰撞对筛选、法向载荷超过原阈值0.1N的胶面数量：

`r_rotation = g(n) * max(r_dir, 0) + min(r_dir, 0)`

`g(0)=0.1`、`g(1)=0.5`、`g(n>=2)=1.0`。反向旋转惩罚不减弱；静止不会凭此得到旋转奖励。禁止在关闭物体接触筛选时启用，避免把自接触或桌面接触作为奖励依据。

本次只改旋转项。位置权重保持2，独立接触奖励权重保持0.05；30度进度目标奖励、其他奖励、接触阈值、actor观测、动作控制、模型和抓取缓存保持基线设置。不新增悬空失触失败条件。旧配置缺少该字段时保留原奖励，包括已有桌面实验。

诊断项 `rotation/contact_multiplier`、`rotation/unconditioned_reward`、`rotation/contact_reward_reduction` 是每轮最后一个控制步的环境均值；`reward/rotation` 则按现有记录方式统计整轮平均。不要把两者直接混用为相同统计窗口。

下次启动B实验时，将监控入口的 `--config` 设为 `manip_hora16384_sphere55_gel_rotation`，并使用新的输出目录及run前缀。恢复旧教师权重属于奖励改变后的继续优化，不是同目标断点续训；公平对照应注明初始化检查点与额外训练预算。已有独立基线配置可继续用于A/B比较。

本次实现不启动正式训练。验证记录位于 `../research/gel_rotation_reward_20261017/`（相对项目上级目录）；短回放只核对奖励，不证明学习后的胶面使用率会提高。
