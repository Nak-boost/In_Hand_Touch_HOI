# 文献—证据映射

核查日期：2026-10-12。全文事实与本项目研究建议分开。作者自报性能不作跨论文排名。

| ID | 文献/日期/来源 | 可支持的论点 | 条件与风险 | 证据位置 | 引用段落 |
|---|---|---|---|---|---|
| NF | [NeuralFeels with neural fields: Visuotactile perception for in-hand manipulation](https://arxiv.org/abs/2312.13469); 2023 preprint; 2024-11-13 Science Robotics | 手内视触觉形状/位姿联合估计；论文明确提出预训练形状先验和感知驱动规划作为后续方向 | 主系统融合外部RGB-D；S4为本体驱动旋转，非重建驱动动作规划 | 4.1, 4.6–4.7, Discussion, S4 | 现状比较/创新边界 |
| NFLOW | [NormalFlow: Fast, Robust, and Accurate Contact-based Object 6DoF Pose Tracking with Vision-based Tactile Sensors](https://arxiv.org/html/2412.09617v2); 2024 preprint; RA-L 10(1), January 2025 | 触觉法向图可用于未知对象6DoF相对跟踪，并支持几何重建 | 不是自主多指探索控制系统；不把论文低纹理例子解释为所有对称表面都可观 | Method and experiments | 现状比较/创新边界 |
| GEL | [GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System](https://arxiv.org/html/2508.15990v2); 2025 preprint; 2026 version/project T-RO | 纯触觉相对位姿跟踪、闭环检测、重定位和全局几何重建已存在 | 问题设置为手持传感器/物体；量化数据包含固定物体+运动传感器；不含机器人自主手内旋转策略 | III, IV, V | 现状比较/创新边界 |
| T2S | [Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction](https://arxiv.org/html/2505.13091v1); 2025-05 preprint | 触觉条件扩散补全和以生成潜变量指导RL触碰选择已经提出 | 仿真触碰采样、球面50个候选动作；不是连续运动物体的触觉SLAM验证 | 3.1–3.3, 4 | 现状比较/创新边界 |
| TA | [TouchAnything: Diffusion-Guided 3D Reconstruction from Sparse Robot Touches](https://arxiv.org/html/2604.08945v1); 2026-04 preprint | 预训练大规模视觉扩散先验可用于稀疏触觉约束的3D重建 | 明确假设机器人运动学给出接触位置，并提供类别级文本；不等价于无类别提示的未知动态位姿重建 | 3 Problem formulation and method | 现状比较/创新边界 |
| GSR | [Grasp, Slide, Roll: Comparative Analysis of Contact Modes for Tactile-Based Shape Reconstruction](https://arxiv.org/html/2602.23206v1); 2026-02 preprint | 扩散补全、信息导向接触选择、多种接触模式已有真实多指手验证 | 九个物体固定在机器人基座坐标系；palm-rolling为手掌在固定物体上滚动，不是物体在单手中自由翻转 | IV, V-B | 现状比较/创新边界 |
| TACTFUL | [TACTFUL: Tactile-Driven Exploration For Object Localization and Identification in Confined Environments](https://arxiv.org/html/2606.24712); 2026-06 preprint | 真实机器人无视觉主动探索、触觉点云补全和识别已有工作 | 已知三对象库，正文限制明确为静态对象；不是通用未知物体的动态手内SLAM | III, IV, Limitations | 现状比较/创新边界 |
| DPOSE | [Object Pose Estimation through Dexterous Touch](https://arxiv.org/html/2509.13591v1); 2025-09 preprint | 双手分工的纯触觉主动探索和物体位姿估计已有研究 | 一手稳定物体另一手探索；仿真实验；位姿后端明确使用FoundationPose model-based assumptions，不能按摘要宣称完全无模型SLAM | III, IV-A, V | 现状比较/创新边界 |
| AURORA | [AURORA: Active Uncertainty-Driven Re-Orientation for In-Hand Reconstruction](https://arxiv.org/html/2609.08493v1); 2026-09 preprint | 重建不确定性指导下一手内旋转轴的闭环系统已经提出 | 重建与CAD-free位姿跟踪使用RGB-D；并非纯触觉重建 | III, Limitations, Appendices A/D | 现状比较/创新边界 |
| PHYS | [Physically Grounded 3D Generative Reconstruction under Hand Occlusion using Proprioception and Multi-Contact Touch](https://arxiv.org/html/2604.09100v2); 2026-04 preprint; v2 June | 生成式形状先验与多接触/手部非穿透约束结合已有工作 | 依赖RGB与可见性mask；单次相机对齐表示不等价于连续纯触觉对象SLAM | 3, Limitations | 现状比较/创新边界 |
| PTLD | [PTLD: Sim-to-Real Privileged Tactile Latent Distillation for Dexterous Manipulation](https://arxiv.org/html/2603.04531v3); 2026-03 preprint; v3 June | NeuralFeels部分共同作者仍研究真实触觉操作和潜变量蒸馏 | 控制工作；真实特权数据采集有相机/标记和已知形状，不是未知物体重建 | Method, A.1 | 现状比较/创新边界 |
| TSLAM | [Curiosity Driven Self-supervised Tactile Exploration of Unknown Objects](https://arxiv.org/html/2204.00035); 2022 preprint | 纯触觉主动探索加形状推断早于NeuralFeels已有研究 | 仿真ADROIT；探索后重建两阶段；局限中明确运动物体接触配准困难 | Method, limitations | 现状比较/创新边界 |
| ACT | [AcTExplore: Active Tactile Exploration on Unknown Objects](https://arxiv.org/html/2310.08745v2); 2023 preprint; revised 2024 | RL主动触觉表面探索和重建已有研究 | 邻接的触觉探测任务，不能据此视为解决动态手内抓持与联合SLAM | Method, experiments | 现状比较/创新边界 |
