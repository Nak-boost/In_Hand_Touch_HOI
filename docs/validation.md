# 验证记录：2026-10-12

当前交付是**可运行的训练实现**，不是已收敛的旋转策略。

## 自碰撞修复验证（当前）

- 原 `teacher_sphere55_z_v1` 已停止，旧实验产物保留，不用于新碰撞模型续训。
- 手部 25 个刚体，仅排除 28 对相邻连接/固定组件内部接触；其余手部件之间启用碰撞。
- 7 项 pytest 通过，包括全碰撞矩阵与旧缓存/权重签名拒绝检查。
- PhysX 对照：使用 GPU PhysX + CPU 数据接口，移开物体，在相同的随机手部动作下：
  启用规则时观测到 79 种手部接触对，应排除对为 0；关闭自碰撞的对照组接触对为 0。
  故意采样挤压姿态会产生较大瞬态力，此测试仅验证碰撞开关，不验证硬件安全或几何校准。
- 抓取生成：256 个环境、3 批，73 个通过物体—胶面真实接触对筛选的候选，保留 64 个。
  缓存 `cache/digit_sphere55_self_collision_grasps.pt`，含碰撞规则签名。
- GPU pipeline 回放：从缓存随机抽样 64 个初始状态、零动作保持 5 秒，63 个未触发失败条件。
  不是对 64 个唯一缓存状态逐一验收，也不代表会旋转。
- 新碰撞规则下教师 1536 transition、学生 40 次更新及评估 64 步完成；评估结束 49 个短回合，
  `all_axes_one_turn_rate=0.0`。这是运行链路验证。
- 证据：`docs/collision_validation/`；对照脚本 `scripts/check_hand_collisions.py`。
- 其他未标定假设与重要差异见 `docs/implementation_audit_20261012.md`。

## 当前单球体、Z 轴配置验证

默认 `configs/manip.yaml` 已改为直径 55 mm 球体、固定 +Z 轴、禁止换轴。
原三物体两轴配置由 `--config-name=manip_two_axis` 显式选择；以下旧验证记录仍对应原配置。

- 两种 Hydra 配置均已检查，物体、轴和缓存路径正确分离。
- GPU 1 上使用 512 个环境重新筛选抓取，2 批获得 85 个候选，保留 64 个于
  `cache/digit_sphere55_grasps.pt`；已检查资产哈希、物体清单及状态有限性。
- 现有 5 项 pytest 全部通过。
- `validation_sphere55_z_teacher`：24 个环境、1536 个 transition，PPO 更新和保存完成。
- `validation_sphere55_z_student`：40 次蒸馏更新和保存完成。
- `validation_sphere55_z_evaluate`：64 个控制步、52 个结束回合，验证单轴统计与报告字段。
  `commanded_axes=[[0,0,1]]`，`all_axes_one_turn_rate=0.0`，不再误用两轴指标名称。

短程检查使用 32 步 episode，只验证链路，不代表旋转成功。手部自碰撞模型仍待修正。

## 已实际执行

- Python 3.8 / PyTorch 2.0.1 CUDA 11.8 / Isaac Gym Preview 4，RTX 4090，GPU PhysX。
- 6 个环境的仿真 smoke，观测维度 `obs[6,71]`、`priv_info[6,23]`、`proprio_hist[6,30,71]`，数值有限。
- 5 项 pytest 通过：四元数符号等价和旋转方向；跨 episode 历史清理；学生特权隔离和联合梯度；
  手模型的独立胶面/外壳与有效 mesh；推理适配器与训练观测/动作滤波的一致性。
- 最终抓取筛选：768 个环境，4 批候选。每类保留 64 个状态，
  `sphere55`、`box50`、`capsule70`，共 192 个，保存在 `cache/digit_grasps.pt`。
- 最终教师检查：24 个环境，1536 个环境 transition，PPO 反传、归一化统计、检查点保存通过。
- 最终学生检查：24 个环境，40 次更新，教师固定，学生 actor/TCN 联合更新，保存通过。
- 最终学生评估：64 个控制步，完成 50 个短回合，写出 `evaluation.json`。
- 独立推理适配器：不导入 Isaac Gym，检查点加载、两轴指令、关节目标限幅通过。
- 教师恢复训练检查：计数从 1536 正确接续至 1920；修复了上游 `PPO.train()` 重置计数的问题。

训练链路检查将换轴间隔缩短为 8 步、episode 缩短为 32 步，用于覆盖切换和重置代码。
因此这些结果不能当作默认 30 秒任务、转满两圈或未知物体泛化的性能指标。
短程学生没有完成两轴各一圈；机器可读结果见 `validation.json`。

## 保留的产物

项目根目录的 `../runs/manip/` 下：

- `initial_grasp_cache_final/`：最终抓取生成配置和源码快照。
- `validation_final_teacher/`：教师检查配置、源码快照、日志、权重。
- `validation_final_student/`：学生检查配置、源码快照、日志、权重。
- `validation_final_evaluate/`：学生评估配置、源码快照、日志和 JSON。
- `_development/`：调试与早期验证记录归档。

不要把 `validation_final_*` 的短程权重部署到真实手上。正式训练请创建新的 run_name。
本机已生成默认抓取缓存，可以直接从 README 的教师训练步骤开始。

## 尚未验证

- 长时间 PPO 是否收敛到稳定两轴旋转，以及不同轴上的成功率。
- 三个训练原型以外的形状泛化、完整时间预算和长期累计漂移。
- 实物 DIGIT 尺寸、安装角度、惯量、胶皮接触与时延标定。
- 自碰撞几何与实物的精确一致性、真实 Allegro 驱动、真实触觉图像到接触信号的标定与部署。
- TACTO 接入、形状与位姿重建，以及主动探索闭环。
