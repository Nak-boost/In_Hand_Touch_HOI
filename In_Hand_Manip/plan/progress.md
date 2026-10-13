# 实施进度

- 2026-10-12：克隆 HORA，固定上游来源提交。
- 已识别 Linux x86_64、4 张 RTX 4090、Isaac Gym Preview 4 本地包和 Python 3.8 环境。
- 创建项目 `.venv`，只读复用原 Python/PyTorch，未改动 TouchFM 环境。
- 安装 Hydra/Gym/TensorBoardX/Pytest，并在项目环境固定 numpy 1.23.5 兼容 Isaac Gym。
- 实现独立 DIGIT 胶面/外壳近似、两轴任务、抓取筛选、PPO 和学生联合蒸馏。
- GPU 仿真 smoke、5 项单元测试、192 个稳定抓取、短程教师/学生训练、评估和恢复训练检查通过。
- 最终配置与完整检查结果记录在 `docs/validation.md`，调试产物已归档到 runs/manip/_development。
- 硬件标定、完整训练收敛和真实机器人部署尚未完成。

- 2026-10-12 环境迁移：通过 conda-pack 本地复制 active3d，创建独立 /root/conda_envs/inhand_manip；源环境未修改，启动脚本不再使用 .venv。
- 固定 Python/PyTorch/CUDA，统一重复包元数据，适配 NumPy、typing-extensions、setuptools/wheel，并补齐 cmake；pip check 无冲突。
- 新环境验证：5 项测试、GPU 仿真、384 transitions 的 PPO、2 次学生更新通过；TouchFM 触觉编码器 CUDA 前向及 PyTorch3D CUDA KNN 通过，非完整重建验证。
- 更新环境维护文档、依赖约束与环境快照；记录 NeuralFeels/AnyRotate 的触觉输入区别，当前 binary 方案仍为简化基线。
- 2026-10-12：删除已停用的项目 .venv（约 94 MiB），验证启动入口仍使用独立 Conda 环境。
