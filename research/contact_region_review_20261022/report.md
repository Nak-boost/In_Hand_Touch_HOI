# 指定触觉区域接触的手内操作：文献与当前项目分析

日期：2026-10-22（Asia/Shanghai）。本次仅调研及只读检查，没有改变训练、模型、奖励、缓存或进程。

## 结论与检索边界

“从整个指尖允许接触，改为希望主要通过 DIGIT 胶面接触”可能显著增加训练难度。它同时限制接触位置、可用接触法向、传力方向，以及换指过程中的接触迁移。当前还不能由训练差直接证明这一项是唯一原因，因为手部几何、控制基准、奖励和初始抓取也不同于原 HORA。

最直接证据来自 AnyRotate：作者明确讨论单向传感器侧面外壳接触的不稳定性，调整安装方向；同时明确指出接触与稳定性奖励会导致稳定握持而不旋转，使用自适应奖励课程。

本轮尝试 arXiv 检索和外部连接，但配置的 HTTP 代理断连、直接连接超时。因此下列结论基于工作区论文 PDF、此前已保存的 arXiv 全文，以及项目源码/冻结配置/已有日志。这不是截至今日的完整新增文献检索。网络原始尝试记录在 /tmp/contact_literature_search.json。

## 已核对全文的相关工作

### AnyRotate（2024）

- 原文：https://arxiv.org/html/2405.07391v3
- 本地已保存全文：../long_horizon_rl_20261019/anyrotate.html
- 关键原文摘录：anyrotate_excerpts.txt

这是最直接对应“指定传感器区域操作”的工作。模拟触觉从 sensing surface 获取接触位置和力；奖励鼓励至少两个指尖接触、惩罚非指尖接触。

Sensor Placement 段明确指出，单向传感器主要在正面感知，侧面外壳接触可能打滑并导致不稳定。作者给各指传感器设置方向偏置（拇指、食指、中指、无名指分别 -45°、-45°、0°、45°），增加感知表面接触机会。其硬件和安装坐标系不同，不能把角度直接照搬到我们的 DIGIT。

Adaptive Curriculum 段明确写道：接触与稳定性项虽有利于最终迁移，但会妨碍学习，产生“稳定抓住却不旋转”的局部最优。其主旋转奖励保留，接触/稳定性项通过课程系数逐步引入；附录 B.3 根据每回合达成的连续目标数调整系数，范围参数为 1 到 2。不要把它误读成先强制所有指尖持续接触。

稳定初始抓取还经过多重力方向检验，并筛选指尖接触及非指尖接触，但静态库只是起点，不能取代动态换指。

参考启示：安装几何、接触奖励课程、动态换指必须共同设计。AnyRotate 的附录若干公式存在符号/条件疑点，不能不核对实现就逐字抄成代码。

### RotateIt / General In-Hand Object Rotation with Vision and Touch（CoRL 2023）

- https://arxiv.org/abs/2309.09979
- 本地 PDF：../../In_Hand_Manip/Papers/General In-Hand Object Rotation with Vision and Touch.pdf

采用全向视觉触觉指尖，触觉抽象为指尖坐标系中 8 个离散接触位置，而非仅二值接触。其适用触觉覆盖与一个局部 DIGIT 胶面不等价。

附录 Reward Hyperparameter 明确说：偏轴惩罚从训练开始设为 -0.1 会使策略仅学会稳定持物，因此从 0 逐渐调到 -0.1。这与“从一开始同时施加所有最终要求，探索会受阻”的判断一致，但不能说它单独证明了胶面约束造成失败。

参考启示：能读取接触位置，才更容易区分即将滚出感知区域与位于区域中央。可先给教师接触局部坐标作诊断，评估能否学会；学生实机感知仍需单独实现，不能把不可测量信号直接当部署输入。

### Rotating without Seeing: Towards In-hand Dexterity through Touch（RSS 2023）

- 本地 PDF：../../In_Hand_Manip/Papers/Rotating without Seeing： Towards In-hand Dexterity through Touch.pdf

使用覆盖指尖、指节、手掌的 16 个二值传感器。方法通过扩大接触覆盖，支持手掌与手指共同操作，而不是强迫全部操作发生在四个小胶面内。

原文对 x/y 轴旋转的讨论指出，物体与指节侧面之间会出现关键接触，当前传感器布局不能覆盖这些接触可能导致失败。

参考启示：能读取二值触觉不等于具备同样的感知能力；传感器覆盖区域是关键条件。该工作任务包含掌部支撑，不能作为四个 DIGIT 的直接可比基线。

### Robot Synesthesia: In-Hand Manipulation with Visuotactile Sensing（2024）

- https://arxiv.org/abs/2312.01853
- 本地 PDF：../../In_Hand_Manip/Papers/Robot Synesthesia：In-Hand Manipulation with Visuotactile Sensing.pdf

用分布式 FSR 触觉，把激活传感器的几何位置表示成点云，再与视觉/手部点云融合。Related Work 对 RotateIt 的讨论明确把持续保持指尖触觉接触描述为对象/操作范围的限制；这是作者对前作的归纳，不是相同条件下改变感知面积的独立因果实验。

参考启示：一种解决方向是扩大有效感知覆盖，并用空间表示表达“哪里接触”，而不只是加大接触奖励。适用于允许更广泛手部接触的任务；不是建议当前项目立即更换硬件。

### Grasp, Slide, Roll: Comparative Analysis of Contact Modes for Tactile-Based Shape Reconstruction（2026）

- https://arxiv.org/abs/2602.23206
- 已核对全文：/remote-home/ums_hanminghao/Projects/TouchFM/research/neuralfeels_followup_20261011/grasp_slide_roll_full.txt

显式设计抓取释放、手指擦过引发的滑动、手掌滚动三类接触模式。先采样候选接触位姿并检查可达/碰撞，再执行带力控制的交互动作。说明为了获得触觉数据，可直接规划有目的的接触模式，而不是只奖励物体旋转速度。

重要边界：真实实验中的物体相对机器人基座固定，不需要同时解决悬空持物和未知物体运动，不能把其效果直接外推到自由手内旋转。

### 后续可追的接触规划文献线索

以下题录已在上述论文参考文献核对，但本轮未取得并精读全文，不将具体性能作为本报告证据：

- Han & Trinkle, Dextrous Manipulation by Rolling and Finger Gaiting, ICRA 1998。
- Sundaralingam & Hermans, Geometric In-Hand Regrasp Planning: Alternating Optimization of Finger Gaits and In-Grasp Manipulation, ICRA 2018。
- Morgan et al., Complex In-Hand Manipulation via Compliance-Enabled Finger Gaiting and Multi-Modal Planning, RA-L 2022，https://doi.org/10.1109/LRA.2022.3145961 。
- Khandate et al., Sampling-Based Exploration for Reinforcement Learning of Dexterous Manipulation, https://arxiv.org/abs/2303.03486 。

检索方向：contact-constrained manipulation、contact-region constraints、finger-gait planning、contact-mode scheduling、tactile servoing、contact-aware reward curriculum。

## 当前实现与日志证据

已核对 2026-10-19 四组实验的冻结解析配置，以及当前对应源码。它们从头训练的是特权教师 PPO。因此当前教师本身未学好时，不应先把失败全部归因于最终学生仅有二值触觉。

当前是胶面接触的软偏好，而不是严格“非胶面一碰就失败”：

- 至少两个有效胶面—物体接触时给 0.05 接触奖励。
- 外壳接触有 0.05 系数的代价；当前该项使用外壳净接触力，不是目标物体专属接触对统计。
- 正向连续旋转奖励在 0/1/≥2 个胶面接触时分别乘 0.1/0.5/1.0；反向旋转惩罚不打折。
- 离散过角度目标奖励没有相同接触门控。
- 悬空任务失触不直接终止；桌面冻结配置 terminate_on_contact_loss=false。
- 胶面有效接触当前按“胶面刚体—目标物体”接触对和法向载荷判断，并未进一步检查具体接触点是否位于光学有效区域、是否接近边缘。因此胶面刚体接触不能完全等同于有效触觉图像。

源码：hora/tasks/manipulation_math.py 的 contact_conditioned_rotation；hora/tasks/allegro_digit_two_axis.py 的 _failure_conditions/_compute_reward；hora/utils/object_contacts.py。

2026-10-22 17:16 已有汇总的最后 1000 万训练步窗口：

| 任务 | 有效胶面—物体接触数均值（最多4） | 平均接触倍率 | 有符号轴向角速度 rad/s |
|---|---:|---:|---:|
| 悬空单球 | 0.958 | 0.462 | 0.113 |
| 悬空六物体 | 0.595 | 0.331 | 0.074 |

这些是训练日志均值，不是独立评估成功率，也不证明姿态几何无法支持胶面旋转。它们表明训练中仍未持续保持较多有效胶面接触。原始摘录见 training_snapshot_excerpt.json。

10月18日抓取缓存经30秒稳定保持、多种随机种子筛选：至少两路有效胶面接触占比≥90%，非胶面支撑≤10%。该验收不保证存在能循环返回、且不越关节限位的换指轨迹。静态抓紧与动态可操作必须分开评估。

10月17日同一已有策略轨迹的奖励反事实审计显示，加入门控后连续旋转项均值由0.761降到0.244，总奖励由0.357降到-0.160。这说明新目标确实大幅改变回报，不是关于重训成功率的因果实验。

## 为什么指定胶面会困难

1. 可用接触区域缩小。目标表面可触达不等于胶面能以合理法向触达；指尖位置与朝向由同一组关节耦合控制。
2. 可施加的接触力/力矩集合改变。胶面只能压不能拉；摩擦和法向限制可用切向力，原来依赖侧面或其他部位的轨迹可能失效。
3. 连续旋转必须换指。接触会从中心向边缘迁移，阶段性失触并重新接触可能是必需动作；持续接触计数奖励可能偏向静态夹紧。
4. 正向旋转奖励随接触变少而衰减，但其他代价未同比衰减，可能压低探索动作回报，形成不动或很慢的局部最优。
5. 部署策略的接触观测还需区分位置/边界；仅二值信号不能区分传感器中央和即将滑出的边缘。但这应与教师学习能力分开检验。

## 建议的验证顺序（尚未执行）

先在同一个 DIGIT 资产、同一组抓取、同一控制器和随机种子下，针对单球做可比对照，避免用原 HORA 的不同资产直接作为因果证据。

| 条件 | 用于奖励的允许接触区域 | 正向旋转接触门控 |
|---|---|---|
| A | 较宽的末端指尖区域 | 关闭 |
| B | 胶面区域 | 关闭，保留相同系数的区域偏好 |
| C | 较宽的末端指尖区域 | 开启 |
| D | 胶面区域 | 开启（当前思路） |

接触奖励/非法区域代价应使用一致的目标物体接触对定义；区域与门控以外的内容保持相同。宽区域只作为诊断，不作为最终满足触觉目标的结果。

- A也失败：优先查 DIGIT 几何、抓取、控制与基础优化问题。
- A成功、B明显失败：支持感知区域限制是重要困难。
- B成功、D失败：支持当前奖励耦合方式阻碍探索。
- 四组同时报告总转角、有效胶面接触下转角、掉落率、连续完成一圈/多圈比例、失触时长和重新接触成功率；不能只比不同定义的总reward。

之后才考虑：按 AnyRotate 思路对接触/稳定性项做课程；为关键指明确支撑—推进—回撤—重接触的阶段；增加同一指尖内接触位置与边缘距离监督；给缓存增加局部可操作性而非只看静态稳定。若所有合理轨迹都必须经过传感器无法感知的区域，则奖励无法修复硬件几何，需要调整安装/感知覆盖或放宽允许短暂辅助支撑的任务定义。

对于最终重建任务，应区分“永远只靠胶面持物”和“稳定操作过程中持续获得足够有效且有覆盖的胶面观测”。二者不是同一任务；应依据重建算法允许的失触长度、接触重叠和覆盖要求选择约束。
