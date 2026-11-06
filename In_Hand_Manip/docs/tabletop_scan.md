# 桌面触觉扫描状态机

该基线用于桌面支撑下的小角度旋转和重复触觉接触，不需要 RL 权重或抓取缓存。入口为 `scripts/run_tabletop_scan.sh`，配置为 `configs/tabletop_scan.yaml`。悬空 RL 和旧桌面实验不受此入口影响。此次一并保存的历史训练改动另见 [训练兼容性说明](training_compatibility_20261106.md)。

## 控制与明确的假设

阶段：`settle → approach → hold → rotate → release → reset → approach`。

- 固定手腕，物体预先放在已知桌面区域；默认圆柱直径 60 mm、高 70 mm、质量 60 g。
- 初始开手姿态由 URDF 运动学与尺寸提示求解；不是未知物体抓取器。之后控制只读本体、胶面净接触载荷和固定放置中心，不读取物体真值。
- 各指径向缓慢接近；用带滞回、时间滤波的接触阈值判断接触，并通过小幅位置调整维持轻载。四指接触持续满足条件后进入推进。
- 目标是围绕固定中心协调推进 10°，通过带胶面法向约束的阻尼逆运动学计算关节目标，再走现有的动作限幅、平滑与 PD 控制接口。保持预载不能简单使用零动作。
- 推进允许短暂失触，最低接触数默认为 2，不保证整个阶段四指同时承载。10° 是指尖运动指令，不是保证物体实际旋转 10°。
- 释放先径向撤离、确认无接触，再复位；复位若重新接触则返回释放。超载、接近/释放/复位超时会触发回撤或故障停止。
- 桌面/物体摩擦均为 0.1、胶面摩擦为 1.0，沿用低摩擦桌面实验的显式参数；不是任意桌面上的性能承诺。
- 默认使用已有的拇指视觉修正资产，物理几何与 compact inward90 相同。新 `AllegroDigitScan` 场景启用胶面—桌面和不同胶面之间的碰撞；仅保留相邻关节/固定装配的必要排除。旧桌面碰撞策略没有修改。
- 载荷目前来自理想仿真传感代理，未模拟 DIGIT 图像、真实力估计误差和弹性胶体。真实硬件仍需接触/载荷标定。

## 安装与运行

在 `In_Hand_Manip` 目录和 `inhand_hoi` 环境下，已安装本项目依赖及 Isaac Gym 后：

```bash
python -s -m pip install -r requirements-scan.txt

# 本地窗口，不需要 checkpoint，不需要传输旧抓取缓存。
bash scripts/run_tabletop_scan.sh --seconds 60

# 无窗口连续评估；这里的 device 是当前可见 GPU 的编号。
bash scripts/run_tabletop_scan.sh --headless --seconds 300 \
  --num-envs 4 --offset-mm 2 --seed 43 --device cuda:0
```

默认结果写入 `../runs/scan/<时间>/`。可指定 `--output`，为防覆盖必须是尚不存在的目录。`--offset-mm` 只扰动物体初始放置，控制器仍使用名义中心；不把扰动后的真值给控制器。`--rotation-degrees` 可覆盖推进角度（符号表示方向）。

保存配置可以直接复用：

```bash
bash scripts/run_tabletop_scan.sh --config ../runs/scan/某次运行/config.yaml --seconds 60
```

离线诊断图：

```bash
python -s -m pip install -r requirements-scan-vis.txt
bash scripts/python_manip.sh scripts/plot_tabletop_scan.py --run-dir ../runs/scan/某次运行
```

窗口需要可用的 Vulkan 图形设备。Ctrl-C 会结束采集并保存已完成步骤。修改物体尺寸/形状须在单独配置中同时检查初始高度、手指接触高度和可达性，当前默认只验证圆柱。首个控制器故障会停止整个并行评估批次。

开发时相机视频已成功导出，但服务器上的 Preview 4 在图形资源销毁时发生原生崩溃。因此稳定入口暂不包含相机录制选项；正常无窗口运行已完成测试，视频调试问题不改变记录的物理结果。

## 数据与指标

- `config.yaml`：解析后的场景与控制配置。
- `setup.json`：开手姿态、逆运动学误差、资产/碰撞签名、输入信息边界。
- `report.json`：每个环境的阶段事件、完整循环净转角、累计净转角、后半程净转角、最大漂移、桌面支撑、非胶面接触、载荷和接触方位覆盖。
- `trajectory.npz`：逐步关节位置/速度、PD 目标、实际限幅动作、阶段、胶面位置与净载荷；同时保存独立的物体真值和目标专属胶面载荷用于评估。
- `diagnostics.png`：离线诊断图。

`q/qdot/targets/action/phase/gel_load/object_state/axis_angle` 在动作执行前对齐；`post_object_state/post_axis_angle/table_contact/other_hand_contact` 对应执行后。`time` 为动作前时刻。关节顺序存于 `joint_names`，不能假定是数字 0–15 顺序。阶段名称存于 `phase_names`。

`contact_samples` 每行是 `[post_step, env_id, finger_id, object_local_x, object_local_y, object_local_z, normal_load]`，来自模拟器的目标物体侧接触点。当前 COM 为零；这些点不是从 DIGIT 图像恢复的点云，不能当成已完成的纯触觉重建。方位覆盖指标只是这些接触点的 36 个水平角度区间覆盖，既不等于完整表面覆盖，也不证明可配准。未来采集图像时需增加对应时间戳、传感器标定和有效接触图像验证。

`cycles` 计数的是完成复位的控制循环，包括可能提前终止推进的尝试；`completed_pushes` 是完成预定推进时间的次数。都不等于任务成功次数，应结合 `cycle_metrics.net_degrees` 和物体保持、接触指标判断。

## 后续 BC/RL 接口

阶段机可先保留为调度器；BC 学阶段内反馈，或学习完整历史到动作的映射。保存的是控制器真正发送的动作而不是关节差分。物体真值和接触身份只能作为特权监督/评估；不能混入纯触觉部署输入。当前程序不训练 BC/RL，也没有实现未知形状重建。
