# 新 DIGIT 抓取与姿态正则实验（2026-10-14）

基于 ed93077 的官方来源网格装配。新增环境奖励：
`-0.3 * sum((q - q_initial)**2)`，q_initial 为每次重置的实际抓取关节角，
仅重置对应环境时更新。保留原动作变化项和已修正的对称动作均值边界惩罚（b=1.1、系数1e-4）。
旧实验配置没有 pose_weight 时按0处理。没有修改重力、摩擦、碰撞过滤、物体失败阈值或其他奖励系数。

## 抓取生成与检查

旧canonical姿态在新装配上试采样1024个候选无通过。研究目录
`research/grasp_training_20261014/` 保留初次探测、几何种子搜索和后续物理验证记录。
先用官方网格做静态接触/自碰撞间隙优化，再由实际GPU PhysX筛选；
PyBullet仅用于离线种子提议，不替换训练物理引擎。

新缓存增加与states逐行对应的targets字段，保存筛选时的PD目标，以保留初始夹持预紧。
没有该字段的历史缓存仍以实际关节角作为目标。参考姿态惩罚始终针对实际初始角，
而非带预紧偏差的控制目标。质量检查必须使用和训练相同的恢复方式。

`search_grasps.py` 从几何种子随机扰动，并对已找到的状态局部扰动；保存每批真实接触统计、
完整配置、资产/碰撞哈希和joint_names。预紧方向由网格间隙对关节角的有限差分估计，
仍使用原限力矩PD，所有目标按关节限位裁剪；这不是实机预紧标定。

`check_grasp_cache.py` 在独立seed=20261014的物理参数采样下，逐个回放每个唯一缓存状态。
筛选条件：30秒不触发原失败条件，至少两个真实物体—胶面接触的帧占比≥90%，
物体接触非胶面部件的帧占比≤10%，初始实际关节角距限位<0.03 rad的关节不超过2个。
这不等于任何物体/摩擦条件下都能稳定保持。

另外每个手指分别做0.08 rad近端屈伸关节回撤并恢复，检查是否实际回撤≥0.02 rad、
是否掉落和是否恢复至少两胶面接触。只统计回撤前实际接触物体的手指。
单指无协同回撤不是硬筛选条件：策略可能需要先建立新接触再释放旧接触。
该诊断不能当作连续换指或旋转成功验证。

## 独立实验

规划 sphere55/sphere65（直径55/65 mm）、box45/box55（边长45/55 mm）、
capsule70/capsule80（总长70/80 mm、直径50/60 mm）。配置位于
`configs/manip_official_<物体>.yaml`，每组分别生成和检查缓存。
不把未通过检查的候选直接用于训练，也不把同一缓存重命名成不同尺寸。

每组从头训练特权教师，+Z轴、256并行环境、horizon=16、minibatch=4096，
短基线预算1000万环境步。未自动蒸馏学生。
GPU与最终进入队列的组以研究目录中的experiment_plan.json/queue_status.json为准；
训练进程不能用GPU显存余量判断算力仍然空闲，队列限制每张指定GPU最多2组。
每组保留独立配置、源码快照、日志和权重。

诊断新增TensorBoard项：reward/pose_penalty、control/target_increment_lost、
control/target_at_limit、control/torque_at_limit。
目标增量裁剪率按每环境L1比例取平均，分母极小的零动作按0处理；
不同于此前报告按全体增量L1总和计算的加权比例，二者不能直接当作同一统计量。
力矩饱和只采控制步最后一个物理子步。


## 复现入口与运行状态

从 `In_Hand_Manip/` 使用 `scripts/run_manip.sh --config-name manip_official_sphere55`，
其余组将配置后缀替换为 sphere65、box45、box55、capsule70、capsule80。
启动器使用独立 Conda 环境 `/root/conda_envs/inhand_hoi`；不需要激活旧 venv。
本次由 `scripts/run_experiment_queue.py` 按研究目录内 `experiment_plan.json` 启动，
并行数、环境数、训练预算以该计划为准。队列在等待抓取报告时不会启动对应实验。

检查报告中的 `retained` 仅表示没有触发掉落/位移失败；`accepted` 还要求胶面接触、
非胶面支撑和关节限位余量达标。训练只使用后者。不同大小分别使用独立缓存和策略，
本批不是多物体联合训练，也不能证明对未知物体的泛化。

验收记录在 `../research/grasp_training_20261014/`：
`reset_pose_validation.json`、`ppo_validation.json`、`checked_cache_validation.json`、
`training_start_validation.json`，以及每组的 `*_quality*.json`。
TensorBoard 新增项只负责诊断，未把单指回撤或接触质量筛选条件加入训练奖励。

初次512环境v1发生PhysX显存分配失败，详见研究报告。最终统一采用v2_env256从头训练，
每次启动要求至少12000 MiB空闲显存，最多同时两组，完整抓取复查已结束。
