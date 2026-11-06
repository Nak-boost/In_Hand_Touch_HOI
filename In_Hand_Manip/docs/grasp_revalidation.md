# 当前控制方式下的抓取复验

2026-10-18：悬空 HORA 训练参数配置恢复项目网络 `256/256/128` 和 `128/64/8`；桌面各配置启用
`rotation_contact_multipliers: [0.1, 0.5, 1.0]`，仅作用于连续正向旋转项。
训练触觉阈值仍为 0.1 N，PD、物理步长、动作缩放和平滑不变。

`scripts/revalidate_grasp_cache.py` 按当前任务完整配置复验已有候选状态。缓存中的 targets 是预载目标，
必须经 `tracking_action` 转为与策略相同的有界动作，再进入平滑、actual_position 目标转换与 PD。
零动作不是持续保持旧预载目标；直接改 env.targets 后输入零动作也不能实现回撤。

每个状态在 seeds 901、1701、2901 下分别保持 30 秒，取三次通过集合的交集：

- 没有任务失败；至少两路胶面—物体法向载荷超过 **0.15 N** 的帧占比至少 90%。
- 非胶面手部支撑不超过 10%。桌面还要求至少 90% 帧有桌面支撑、单胶面净合力不超过 2 N。
- 悬空沿用“不超过两个关节距限位小于 .03 rad”的验收条件。
- 使用实际 `control_dt` 把秒换算为步数。当前约 20 Hz 策略控制、120 Hz 物理/PD 更新。
- seed 901 另做四指各 0.06 rad 的回撤/恢复诊断。单指回撤不等于协调换指，未作为抓取库硬门槛。

物体质量、摩擦、COM 在每次独立场景初始化时重新采样，控制增益在 reset 时重新采样。
三个 seed 不保证覆盖全部随机参数或实机情况；报告保留每状态结果、接触阈值裕度和回撤实际位移。

复验脚本示例（生成报告，不覆盖缓存）：

```bash
CUDA_VISIBLE_DEVICES=1 scripts/python_manip.sh scripts/revalidate_grasp_cache.py \
  --config manip_tabletop_inward90_hora16384_sphere55 \
  --cache cache/digit_tabletop_sphere55_inward90_checked_20261016.pt \
  --seed 901 --seconds 30 --margin-force .15 --retractions \
  --output /tmp/table_sphere_validation_new.json
```

本次九组任务、交集导出脚本、逐状态报告和缓存清单位于
`../../research/cache_refresh_20261018/`。新缓存带 `validation_contract`，加载时检查物理 dt/substeps、
PD/控制降频、动作模式/缩放/平滑、触觉阈值和随机化范围；修改这些字段后应重新复验。
历史缓存仍支持原先加载规则，旧快照保持原文件。

现有冻结训练不会热加载这些变更。旧大网络权重不能直接用于恢复后的小网络；新实验需使用新输出目录。
