# 单轴、单物体并行实验

2026-10-12：按用户要求，在原球体实验之外增加立方体和胶囊体实验。
三组分别训练独立教师，不是把三个物体混合训练成一个策略。

| 物体 | 尺寸 | Hydra配置 | 教师运行目录（相对 `../runs/manip/`） |
|---|---|---|---|
| sphere55 | 直径55 mm | `manip` | `teacher_sphere55_z_self_collision_v1/` |
| box50 | 边长50 mm | `manip_box50` | `teacher_box50_z_self_collision_v1/` |
| capsule70 | 总长70 mm、直径50 mm | `manip_capsule70` | `teacher_capsule70_z_self_collision_v1/` |

共同设置：+Z单轴、不换轴、自碰撞已修正、1024个并行环境、seed=42、每组1亿环境步；
相同奖励、控制器和物理参数随机化。DIGIT几何和手部重力等既有假设见实现审计。
新增两组沿用同一抓取姿态种子与筛选阈值，为各自物体独立生成64个真实物体—胶面接触的缓存。
胶囊体初始长轴沿X；未增加物体朝向随机化。

三组共享物理GPU 1，`CUDA_VISIBLE_DEVICES=1`后进程内部设备显示`cuda:0`是正常映射。
原球体运行约3.1 GB显存时GPU计算利用率已约80%，更多进程不意味着等比例吞吐提升，
会共享GPU时间。新增进程设置`OMP_NUM_THREADS=1`，减少CPU线程争抢。

## 手动复现

以下命令针对新的运行名；已存在的缓存和运行目录不会覆盖。

```bash
cd /root/In_Hand_Reconstruction/In_Hand_Manip

# 抓取生成，GPU PhysX + CPU接触接口。缓存已存在时跳过。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 scripts/run_manip.sh \
  --config-name=manip_box50 stage=grasps pipeline=cpu rl_device=cpu num_envs=256
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 scripts/run_manip.sh \
  --config-name=manip_capsule70 stage=grasps pipeline=cpu rl_device=cpu num_envs=256

# 教师训练；在不同终端执行实现并行，自动生成互不覆盖的运行目录。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 scripts/run_manip.sh \
  --config-name=manip_box50 stage=teacher num_envs=1024
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 scripts/run_manip.sh \
  --config-name=manip_capsule70 stage=teacher num_envs=1024
```

## 日志和后续阶段

每组控制台日志位于运行目录同级的 `<run_name>.console.log`，启动记录为 `<run_name>.launch.json`。
TensorBoard在各运行目录的`stage1_tb/`，权重在`stage1_nn/`。
教师完成后才能判断是否值得蒸馏；本次没有自动启动学生训练，也不把训练启动当成旋转成功。
后续学生/评估必须使用同一个`--config-name`和对应教师/学生权重。
