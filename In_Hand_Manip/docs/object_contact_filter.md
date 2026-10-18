# 胶面—物体接触筛选（默认启用）

当前 `manip` 及其派生配置默认 `task.env.manipulation.object_contact_filter=true`、`pipeline=cpu`。
缺少字段的历史解析配置仍按关闭处理，重放历史实验时应使用该实验保存的配置。
开启时只改变任务接触判定，不改变物理碰撞、手模型或 teacher/student 的触觉观测。

## 运行配置

正常使用当前项目配置无需额外开启。对显式覆盖过开关或管线的命令，配套设置为：

```bash
pipeline=cpu task.env.manipulation.object_contact_filter=true
```

保留 `sim_device=cuda:0 rl_device=cuda:0`，物理计算及 PPO 仍运行在 GPU。
这里 `pipeline=cpu` 指状态张量/接触查询使用 CPU 数据管线，不是把 PhysX 改为 CPU 仿真。
Isaac Gym Preview 4 的 `get_env_rigid_contacts` 在 GPU 数据管线启动后不可调用；若开关打开却
仍使用 `pipeline=gpu`，初始化会明确报错，不自动切换管线，也不退化为几何距离近似。
旧实验若希望完全恢复原运行条件，应同时恢复 `pipeline=gpu` 和开关 `false`。

## 判定与作用范围

每个控制步，在最后一次物理步完成后查询一次每个环境的接触列表，筛出四个独立胶面刚体
与当前环境目标物体的接触。按照 Preview 4 的 `RigidContact.lambda` 定义，将该接触对各
接触点的正法向载荷相加（单位 N），超过 `task.env.digit.contact_threshold` 才计有效接触。
默认阈值保持 0.10 N。该指标是物体接触法向载荷，不是包含剪切的胶面总净力模长；不除以 dt。
相较原净力判定，这是明确的信号定义差异，不能假定两个指标在所有接触姿态下等价。

同一个 `[num_envs, 4]` 布尔结果供以下用途复用：

- 至少两路有效物体胶面接触时获得已有接触奖励（权重保持原配置）。
- 桌面任务连续四路物体胶面接触全无时累计失触计时；现有 2 秒阈值、失败代价和重置行为保持。
- 对应任务接触统计。

物体轻触胶面、但其他手指强压胶面时，自接触载荷不会把物体接触抬过阈值。
只有自接触、桌面接触或外壳—物体接触时，不计胶面—物体接触。
接触解除后的每次查询重新写入零载荷，不跨步保留旧接触。
当前实现与原奖励一样采样控制周期末的物理接触，不额外汇总整个 6 个物理步的接触历史。

所有策略的输入仍使用原胶面净接触力阈值，保留原延迟与漏检模型，包含实际发生的自接触信号。
筛选结果是仿真任务监督，不加入 `obs`、`priv_info` 或 `proprio_hist`。
当前外壳惩罚仍沿用原定义；本开关不把其他奖励项改成 object-only。
非桌面的 `AllegroDigitTwoAxis` 任务只改变接触奖励，因为其失败条件原本不是失触计时。

## 日志

- `contact/gel_count`：保留原胶面净力统计，便于对比旧实验。
- `contact/task_gel_count`：实际用于奖励/失败判定的数量。
- 开启时另记 `contact/object_gel_count`、`contact/object_gel_normal_force_N`，以及
  `contact/raw_without_object`（原始有信号但无达阈值物体接触的环境比例）。

## 验证和性能

单元测试：`scripts/python_manip.sh -m pytest -q tests/test_object_contacts.py tests/test_tabletop.py`。
`scripts/benchmark_object_contacts.py` 支持记录一次教师动作序列，随后在同一 GPU 上分别回放：
GPU 数据管线/关闭、CPU 数据管线/关闭、CPU 数据管线/开启。
这样分开衡量管线切换与接触查询的开销。测试不更新策略，计时前后同步 CUDA；失败信号仍计算，
但禁用自动重置以避免不同终止条件改变后续轨迹。该回放不是策略成功率评估。

2026-10-16 在 GPU 3、256 个环境的同进程 ABBA 交替测试中，CPU 管线关闭/开启筛选
分别为 58.09/68.30 ms 每控制步，开启增加约 17.6% 耗时，吞吐下降约 15.0%。
其中接触查询和聚合平均占 9.48 ms。奖励和失败判定复用结果，不重复查询。
另三组独立回放中，相比原 GPU 管线关闭筛选，CPU 管线开启筛选的吞吐中位数下降约 36%；
共享机器负载和独立回放轨迹存在差异，这一总开销仅作粗略参考，不是完整 PPO 训练速度。
不能将 256 环境的结果直接外推到数千环境。

26 项测试通过，同一物理状态下筛选开关前后的策略观测逐项相等。
完整方法、原始数据及限制见[测试报告](../../research/contact_filter_20261016/report.md)。

## 训练中的持续计时

开启筛选后，每个控制步记录以下 TensorBoard 指标，并在每轮 PPO rollout 内取平均：

- `performance/contact_pair_wall_ms`：遍历环境、读取接触对及聚合法向载荷的墙钟耗时。
- `performance/contact_pair_cpu_ms`：同一区间的进程 CPU 时间，包含该进程各线程消耗的 CPU 时间。
- `performance/contact_pair_wall_fraction`：查询/聚合耗时占主机观察到的整个 `env.step` 时间的比例。
- `performance/contact_env_step_wall_ms`：主机观察到的整个环境控制步耗时。
- `performance/contact_pair_wall_us_per_env`：查询/聚合耗时除以环境数量，单位微秒。

计时只读时钟，不增加接触查询、不调用额外 CUDA 同步、不改变动作、物理步或奖励。
CPU 时间与墙钟时间含义不同；前者可包含并行线程，因此不能直接当作单核占用率。
这些指标只衡量查询与聚合区间，不包含 CPU 数据管线切换的全部代价，也不是关闭筛选的反事实对照。
