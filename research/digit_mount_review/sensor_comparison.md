# 手内操作传感器与NeuralFeels装配核查

本次读取下列论文正文/附录及官方代码；硬件类别不等于策略观测。源码/网页下载副本位于忽略的sources目录。

| 工作 | 硬件与部署感知 | 触觉用途和边界 |
| --- | --- | --- |
| [HORA (2022)](https://haozhi.io/hora/) | Allegro，控制依赖本体感知历史 | 部署策略不需要额外触觉传感器；不能称为DIGIT触觉策略 |
| [NeuralFeels (2023/2024)](https://arxiv.org/html/2312.13469v1) | Allegro四个末端各一个商用DIGIT，手装在Franka Panda上；主系统另用RGB-D | DIGIT图像→接触深度/局部几何→形状与位姿估计。S4旋转策略由本体感知驱动，非图像直接控制手指 |
| [RotateIt (2023)](https://arxiv.org/html/2309.09979) | Allegro四指上的全向光学触觉指尖，以及RealSense D435 | 使用触觉图像提取低维接触位置，与视觉深度、本体感知融合；正文未将使用的硬件标作标准DIGIT，不把相关工作中的传感器列表当作实用硬件 |
| [Rotating without Seeing (2023)](https://arxiv.org/html/2303.10880) | Allegro上16个FSR，分布于手指和手掌，外覆指套增摩擦 | FSR电压阈值化后的16路binary接触；不是16个光学相机，也不是四个DIGIT |
| [AnyRotate (2024)](https://arxiv.org/html/2405.07391v2) | Allegro四指，自研DigiTac/TacTip衍生曲面光学传感器，手装在UR5上 | 图像观察器提取接触特征，策略每指使用接触标志、两个接触角度和合力大小；仿真使用刚体接触信息，不渲染策略训练图像 |
| [DexHier (2025)](https://arxiv.org/html/2501.05439) | 多指手、本体感知、RGB-D及物体状态估计 | 论文结论明确将加入触觉视作后续改善滑移/漂移的方向，不能列为已使用DIGIT的工作 |
| [PTLD (2026)](https://arxiv.org/html/2603.04531v3) | Allegro上的Xela uSkin，18块pad、368个三轴感知单元 | 用去除无接触基线的三轴原始读数学习触觉潜变量；不是光学触觉图像，也非仅4路binary。见附录A.1 |
| [DexScrew (2025预印本)](https://arxiv.org/html/2512.02011) | 12-DoF XHand原生五指触觉阵列，每指120个三轴单元 | 原始张量5×120×3；以真实示范训练含触觉的行为克隆策略，不是Allegro+DIGIT |

之前还讨论的工作不能都归为触觉操作：
[DexterityGen](https://arxiv.org/html/2502.04307) 使用关节状态/跟踪误差等隐式接触线索，
[REGRIND](https://arxiv.org/html/2607.11874v1) 实机用动作捕捉提供物体位姿，
[POISE](https://arxiv.org/html/2609.13761v1) actor使用本体感知和RGB-D/FoundationPose估计，
[ArtManip](https://arxiv.org/html/2609.12498v1) 学生使用本体历史与初始观测。
这些观测通道不能说成直接使用高分辨率触觉皮肤。

## NeuralFeels装配能确认到什么程度

§4.2原文：“The Allegro hand is retrofit with four DIGIT vision-based tactile sensors, at each of the distal ends.”
官方 `neuralfeels/modules/allegro.py` 使用
`data/assets/allegro/allegro_digit_left_ball.urdf`，通过16关节运动学定位
`link_3.0_tip`、`link_7.0_tip`、`link_11.0_tip`、`link_15.0_tip`，
并在 `_hora_to_neural` 中增加URDF传感器参考系至SLAM参考系的固定变换。
此固定坐标系变换不能直接理解为传感器相对指腹的安装倾斜角。

确认是左手模型/四个末端DIGIT；本次未成功获取FeelSight assets.tar.gz内URDF，
因此不能确认全部支架尺寸、螺孔及每指安装RPY，也不能称当前模型是它的装配复现。
官方[DIGIT适配器](https://github.com/facebookresearch/digit-design/tree/de292229d43eed1deba58b856b2fe1ee04097631/DIGIT/adaptors/Allegro)
可证明存在配套装配方案，但不证明NeuralFeels使用的正是该版本支架。

## AnyRotate安装角的关键更正

附录J：传感器基于DigiTac/TacTip，包含标记针、软皮肤、相机以及重设计的紧凑基座。
Sensor Placement明确写出(thumb, index, middle, ring)=(-45°, -45°, 0°, +45°)，
用于最大化前向感测区域在旋转中的接触覆盖。
故“所有传感器必须与指腹共面”并非普遍原则；本项目采用共面直线基线是当前明确的设计选择。
这些论文偏置不能直接移植成当前标准DIGIT代理的RPY，尤其不能据此推导旧模型的额外俯仰与拇指-90°。
