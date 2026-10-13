# Allegro + DIGIT：掌心朝上的手内旋转

本仓库基于 [HORA](https://github.com/HaozhiQi/hora)，上游提交 `410d95824dd28b198f6d6910cb4c09330a6303be`。
采用 AnyRotate 启发的训练方式：**特权教师 PPO → 触觉/本体历史学生联合蒸馏**。
它不是 AnyRotate 官方代码或完整复现；这里限定固定手腕、掌心朝上。
当前默认训练 **55 mm 球体绕 +Z 轴持续旋转**，两轴、多物体配置单独保留。

## 当前范围

- Allegro 右手 16 DOF，加四个参数化 DIGIT 刚体近似；胶面与外壳是独立碰撞部件。
- 默认所有并行环境使用同一种 55 mm 球体，无尺寸随机化；质量、摩擦、重心随机化，重置时随机化 PD 增益。
- 20 Hz 策略输出关节目标增量，120 Hz 限力矩 PD 控制。
- 默认仅 +Z 轴（掌心朝上时水平旋转），不换轴；`manip_two_axis` 配置保留 +Z/+X 每 8 秒切换。
- 教师可用物体真值；学生只用 71 维触觉/本体/轴指令以及 30 帧历史。
- 学生监督采用特权潜变量 MSE 和教师高斯动作分布的期望 NLL，联合更新 TCN 和 actor。
- 默认触觉只有四个二值胶面接触，带一帧延迟和漏检；不把仿真力大小当作真实 DIGIT 可用测量。
- 不渲染 TACTO，不依赖 RGB-D；触觉图像重建和位姿估计由后续模块接入。

**状态：实现与短程验证结果见 `docs/validation.md`。不将短程检查权重视为已学会旋转的策略。**

球体、立方体、胶囊体各自训练独立 Z 轴策略的配置和日志位置见
[单物体并行实验](docs/parallel_object_experiments.md)。
将代码、权重和抓取缓存迁移到本地的说明见 [本地迁移](docs/local_transfer.md)。

## 模型的重要边界

`configs/digit_mount.yaml` 是可修改的工程起点，尺寸、质量、安装角度和接触阈值没有做实物标定。
胶面是刚体薄盒，既不是光学渲染，也不是弹性摩擦仿真。已启用跨手指、跨 DIGIT 及非相邻部件碰撞，仅排除相邻连接和同一固定组件内部。
手部重力仍沿用 HORA 的关闭设置；其他未标定假设及改动责任见 [实现审计](docs/implementation_audit_20261012.md)。
教师采用沿轴累计角度的增量目标奖励，未照搬 AnyRotate 的完整 keypoint 目标和重力课程。
当前默认仅一个球体；备用两轴配置使用球、盒子、胶囊体三个原型，均不构成广泛未知物体泛化证据。
精确转满一圈后停止、真实驱动接线、真实 DIGIT 接触检测尚不包含在仿真验证内。

## 环境

当前使用独立 Conda 环境 `/root/conda_envs/inhand_manip`，由 `active3d` 完整复制后适配。
Python 3.8.20、PyTorch 2.0.1 / CUDA 11.8、NumPy 1.23.5；不再依赖项目 `.venv`。
Isaac Gym Preview 4 的本地安装包仍由 `ISAACGYM_PATH` 指定。

```bash
conda activate /root/conda_envs/inhand_manip
# 项目依赖受 constraints-manip.txt 约束：
python -m pip install -r requirements-manip.txt
# 添加包时也使用约束，并把最终版本记录到 requirements-manip.txt：
python -m pip install -c constraints-manip.txt '包名==版本'
python -m pip check
```

训练入口不要求手动激活环境。`scripts/python_manip.sh` 可执行测试、安装和其他 Python 命令，
并配置 Isaac Gym 所需的库路径。环境复制、冲突处理和快照说明见 [docs/environment.md](docs/environment.md)。
触觉输入与论文的差异见 [docs/tactile_design.md](docs/tactile_design.md)。

所有新流程使用 `scripts/run_manip.sh`，不要用上游 `train.py/deploy.py` 加载本项目学生权重。
旧入口和原始实现保留，方便与上游比较；原 README 在 `docs/HORA_UPSTREAM.md`。
**`train_s1.sh`、`vis_s1.sh`、`eval_s1.sh`、`deploy.sh` 是原 HORA 入口，不是当前 DIGIT 任务的快捷脚本。**

## 从头运行

默认球体缓存为 `cache/digit_sphere55_self_collision_grasps.pt`；首次运行需完成第 3 步，缓存存在时可直接从第 4 步开始。
验证用短程权重不代表旋转已经学会；请为正式训练创建新的 run_name。

```bash
cd /root/In_Hand_Reconstruction/In_Hand_Manip

# 1. 生成手模型；修改安装配置后必须重新生成模型与抓取缓存。
scripts/python_manip.sh scripts/build_digit_asset.py

# 2. 仿真检查，不要求抓取缓存；它只检查运行与数值，不验证旋转成功。
scripts/run_manip.sh stage=smoke num_envs=6 evaluation_steps=30

# 3. 为当前 DIGIT 模型生成稳定初始抓取。GPU PhysX + CPU 数据接口读取接触双方身份。
# 仅保留经过重力、PD、接触和低速度筛选的状态；不同物体分别保存。
# 若 cache/digit_sphere55_self_collision_grasps.pt 已存在，使用另一个 cache 路径，程序不会覆盖它。
OMP_NUM_THREADS=1 scripts/run_manip.sh stage=grasps num_envs=256 pipeline=cpu rl_device=cpu

# 4. 单球体、+Z 轴教师 PPO。运行结束后使用对应 run 目录里的 stage1_nn/last.pth。
scripts/run_manip.sh stage=teacher num_envs=1024 run_name=teacher_sphere55_z_self_collision_v1

# 5. 从训练好的教师蒸馏学生。下面路径仅对应上一个命令的 run_name。
scripts/run_manip.sh stage=student num_envs=1024 run_name=student_sphere55_z_self_collision_v1 \
  teacher_checkpoint=/root/In_Hand_Reconstruction/runs/manip/teacher_sphere55_z_self_collision_v1/stage1_nn/last.pth

# 6. 只评估学生，不把特权教师当作可部署策略。
scripts/run_manip.sh stage=evaluate num_envs=96 evaluation_steps=600 \
  student_checkpoint=/root/In_Hand_Reconstruction/runs/manip/student_sphere55_z_self_collision_v1/stage2_nn/last.pth
```

每次执行自动生成唯一 run 目录；显式指定的 `run_name` 若已存在会报错。
默认输出位于 `../runs/manip/`，记录有效配置、Git 版本、代码快照和阶段检查点。

当前默认就是单球体、单轴。后续用 `scripts/run_manip.sh --config-name=manip_two_axis ...`
切回三物体、两轴配置，其抓取缓存为原来的 `cache/digit_two_axis_self_collision_grasps.pt`。可指定
`checkpoint=/path/to/teacher.pth` 继续 PPO，网络观测维度不变。碰撞签名不兼容的旧缓存和旧权重会被拒绝；
本轮开启自碰撞后的训练从头开始。
`max_agent_steps` 是累计预算，恢复时应设更大的预算。
评估报告用 `commanded_axes` 标识轴，`all_axes_one_turn_rate` 表示所有指定轴均累计转满一圈且回合结束时未失败的比例。
学生恢复用 `student_checkpoint=...`，同时指定原教师；`distill.steps` 也是累计预算。

小环境检查 PPO 时要同步减小 minibatch，例如：

```bash
scripts/run_manip.sh stage=teacher num_envs=24 \
  train.ppo.minibatch_size=384 train.ppo.max_agent_steps=1536
```

## 目录与接口

| 路径 | 职责 |
|---|---|
| `configs/manip.yaml` | 默认单球体、+Z 轴任务与训练参数 |
| `configs/manip_two_axis.yaml` | 备用三物体、两轴配置 |
| `configs/digit_mount.yaml` | DIGIT 近似物理模型和安装参数 |
| `hora/tasks/allegro_digit_two_axis.py` | 两轴仿真、观测、奖励、重置与抓取筛选 |
| `hora/tasks/manipulation_math.py` | 不依赖仿真的旋转、观测和历史函数 |
| `hora/algo/models/tactile_student.py` | TCN 和联合蒸馏损失 |
| `hora/algo/padapt/tactile_distill.py` | 学生执行轨迹上的教师监督与检查点 |
| `hora/algo/deploy/digit_runtime.py` | 真实传感器到关节目标的推理适配器，无硬件 I/O |
| `train_manip.py` | 新项目入口 |
| `cache/` | 与手模型哈希和物体集绑定的抓取缓存，不入 Git |
| `tests/` | 特权信息隔离、梯度、旋转和资产测试 |

`DigitPolicyRuntime.step(q, qdot, contacts, axis)` 返回 16 个关节目标。
关节数组必须按检查点 `joint_names` 排列，而不是假定编号顺序；触觉顺序为 index/middle/ring/thumb。
外层程序负责时间同步、DIGIT 接触检测、PD、通信和任务调度。初始抓取和换物体后调用 `reset()`。
完整数据约定见 `docs/interfaces.md`。

```bash
scripts/python_manip.sh -m pytest -q tests/test_manipulation.py
```
