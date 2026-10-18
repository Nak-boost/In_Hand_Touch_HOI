# 触觉手内旋转：公开奖励与实现核查

核查日期：2026-10-15。范围：官方项目主页、论文正文与附录、主页链接的 GitHub 仓库及其奖励函数/配置。本文是源码与文献核查，没有运行这些外部项目，也没有修改当前训练策略或启动实验。

## 结论与此前回答的纠正

1. HORA 的实机策略不使用触觉；不能用它说明触觉接触/换指奖励如何设计。
2. **AnyRotate 确实有“至少两个指尖接触才奖励”的设计**，见最新 arXiv v3 附录 B.1 式 (6)。此前说这种形式完全没有论文依据，是错误的。但这不证明我们此前写代码时已经按论文复现：当前项目讨论中的固定 +0.05、连续失触 2 秒和失败 -250，并非该论文原设置。
3. **Visual Dexterity 使用深度视觉和本体信息，不是触觉策略。** 但其桌面重定向任务、接近物体的奖励、相对实际关节角的控制值得参考。
4. 真实使用触觉的论文没有统一要求“接触数越多越奖励”：有至少两点接触的阈值奖励，也有不使用接触数奖励、改用距离引导和任务奖励的方案。
5. AnyRotate 与 RotateIt 都明确报告过“稳定握持但不旋转”的训练局部最优，并用奖励课程缓解。这是当前现象的相关文献依据，不能据此断定我们的失败只由奖励造成；动作裁剪、接触语义和物理模型仍需独立诊断。

## 公开情况

| 工作 | 实机策略的感知 | 任务 | 公开实现核查 |
|---|---|---|---|
| Rotating without Seeing / Touch Dexterity，RSS 2023 | Allegro，16 路分布式 FSR 二值触觉＋本体，无外部视觉 | 多物体手内旋转，分别训练轴向策略 | 官方主页的 Code 指向下述 Robot Synesthesia 共享代码库，README 明确包含 PS/Rotating without Seeing 基线。不能把共享库当前全部实现当作原论文逐项不变的版本 |
| Robot Synesthesia，ICRA 2024 | 16 路二值 FSR＋深度视觉＋本体；把接触位置编码为触觉点云 | 三轴旋转、双球、轮形扳手等 | 官方训练环境、教师训练、数据收集和学生 BC 代码公开 |
| AnyRotate，2024 | 四个改进 DigiTac/TacTip 光学指尖；每指接触标志、两个接触角度、合力大小＋本体 | 多轴、不同重力方向的持续手内旋转 | 论文附录公开奖励、课程与参数；本次核查到的作者同名仓库是网页资源，未核实到公开训练实现 |
| RotateIt，2023 | 四个全向光学指尖的离散接触位置＋深度视觉＋本体历史 | 通用手内旋转 | 论文公开奖励及部分超参数；官方项目页未找到训练代码链接，本次未核实到官方训练实现 |
| Visual Dexterity，Science Robotics 2023 | 深度点云＋关节编码器，无触觉输入 | 桌面支撑及空中目标姿态重定向 | 官方 dexenv 代码、训练配置公开；发布的是 D’Claw 版本，不是 Allegro＋DIGIT 现成实现 |

“未核实到”是本次搜索范围内的结论，不等于断言任何地方都没有代码。第三方同名仓库不能作为官方实现依据。

## 1. Rotating without Seeing：接触作为观测，距离作为引导

原论文 arXiv:2303.10880v4，IV-A 与附录 D：

\[
r=20r_{rot}+0.1r_{vel}+r_{fall}+0.0003r_{work}+0.0003r_{torque}+0.1r_{dist}.
\]

- 旋转项：目标旋转角增量裁剪到 ±0.157 rad。
- 平移项：负的物体线速度范数。
- 掉落项：-50。
- 功率项：\(-\langle|\tau|,|\dot q|\rangle\)。
- 力矩项：\(-\|\tau\|\)。
- 距离项：

\[
r_{dist}=\frac14\sum_{i=1}^4\operatorname{clip}\left(\frac{0.1}{0.02+4\|p_{tip,i}-p_{obj}\|},0,1\right).
\]

距离是指尖位置到物体位置的几何距离，不是胶面接触计数，也不是物体表面最短距离。它在尚未接触时就能提供接近物体的学习信号。物体位置来自仿真，可以用于训练奖励，不意味着实机 actor 必须获得物体真值位置。

附录所列奖励没有单独的“换一次手指就加分”，也没有按 16 路二值接触数量递增的奖励。换指行为由旋转任务、保持物体不掉落、距离引导及能耗约束共同形成。

该论文动作是对上一关节目标累加，经 EMA 平滑，η=0.8，策略控制频率 10 Hz。不能把 Visual Dexterity 的相对实际关节角控制归到这篇论文名下。

## 2. Robot Synesthesia：可直接审阅的触觉基线代码

官方主页与 Touch Dexterity 官方主页均链接：
<https://github.com/YingYuan0414/in-hand-rotation>

本次检查 commit `1e5cc92864b87fbe60482a923002b5499ec5e799`。

- `readme.md`：仓库是 Robot Synesthesia 代码，并提供 Rotating without Seeing 的 PS 基线启动方式；三轴 PS 使用 `observationType=partial_stack`。
- `isaacgymenvs/tasks/allegro_arm_morb_axis.py:1987`：`compute_hand_reward_finger`。
- 同文件 `:2013` 附近：与上文同形的距离奖励，乘 `finger_coef`。
- 同文件 `:2040`：接触数奖励为 `clip(finger_contacts.sum(), 0, 5) * contact_coef`。
- `isaacgymenvs/cfg/task/AllegroArmMOAR.yaml`：`finger_coef: 0.1`，**`contact_coef: 0.0`**，`fallPenalty: -50.0`，`work_coef: -0.0003`，`torque_coef: -0.0003`。
- `scripts/teacher_axis.sh` 没有覆盖上述系数；用户通过额外命令行参数覆盖时另当别论。

因此不能只看到函数中有 `contact_reward` 就说训练开启了它。该三轴默认启动配置开启距离引导，关闭接触数奖励。动作和控制误差项也存在，但该 YAML 中系数均为 0。

共享代码与原论文不是字面完全相同：例如代码力矩项使用平方和，原 Rotating without Seeing 附录写范数。代码将 `theta*20` 传入后裁剪到 ±3.14；原论文写先裁剪角增量 ±0.157 再乘 20。参数来源应标清楚。

另一个容易误读的地方：YAML 有 `usePrevTarget: False`，但本次检查的相对动作执行路径在 `pre_physics_step` 实际使用 `prev_targets + relative_scale * actions`。**不能仅凭配置字段断定控制行为。**

触觉点云是由已知传感器位置、机器人运动学和二值信号构造的空间表示，不是从 FSR 恢复出物体的稠密表面点云。

## 3. AnyRotate：两指阈值奖励有依据，但课程才是关键配套

以最新 arXiv:2405.07391v3（2024-11-03）为准。本目录还保存较早 v2，不能混用其中参数。已同时核对 v3 HTML 和 PDF。

附录 B.1：

\[
r_{gc}=\mathbf1[n_{tip}\ge2],\qquad
r_{contact}=\lambda_{rew}(0.1r_{gc}+0.2r_{bc}).
\]

因此两指、三指、四指在该好接触奖励上相同；它没有要求用更大接触数量直接促进换指。

其任务奖励还包含辅助目标关键点距离、目标轴旋转增量和达到阶段目标的奖励。阶段目标达成后继续更新目标，实现连续旋转。论文对应权重为关键点项 1、旋转增量项 5、目标成功项 10；旋转增量裁剪 ±0.025 rad。终止项权重为 50，条件是关键点距离超过 0.1 m 或轴偏差超过 45°，并非“失触 2 秒扣 250”。

附录 B.3 与正文说明：接触和稳定性项由 `λrew` 随每回合成功目标数逐渐提高权重，课程区间为 [1,2] 个目标。作者明确指出，这些项会阻碍早期探索，使策略停留在“抓稳但不转”的局部最优。不能只摘取接触奖励而忽略其余任务奖励和课程。

v3 的其余稳定性参数是 `λω=0.5, λpose=0.5, λwork=0.1, λtorque=0.05, ωmax=0.6`；与 v2 不同。本报告不建议直接迁移这些数值。

**公开公式存在需澄清的歧义**，且 PDF 中也存在：坏接触项条件印作 `n_non_tip_contact >= 0`（按计数定义永远满足）；角速度惩罚的 `-min` 与“惩罚超速”文字不一致；关键点公式变量和 work/torque 符号也有不一致。课程公式没有清楚交代区间外裁剪。没有训练代码来消歧时，应注明按文字意图重实现，不能宣称逐项忠实复现。

坏接触在文字中明确定义为物体与非指尖部位的接触。好接触项则使用触觉二值向量 `c` 的和；没有核实作者如何排除手指自身接触，不能假定其实现已经做了 object-only 过滤。

## 4. RotateIt：论文明确报告“惩罚过早会只握不转”

论文 arXiv:2309.09979v2，§3.1 与 Reward Hyperparameter 段：任务由绕目标轴的角速度奖励、非目标轴角速度惩罚、偏离初始关节姿态、物体平移、功率和力矩项组成。所列公式没有接触数奖励，也没有逐次换指奖励。

作者明确写到：若非目标轴惩罚系数从一开始就设为 -0.1，策略只会学习稳定握持；因此先设为 0，再线性调整到 -0.1。这与当前项目“先学会转，再逐渐约束动作”的思路直接相关，但不是对我们失败原因的实证诊断。

仿真触觉使用接触位置投影到指尖坐标系二维平面，离散为 8 个位置；实机从全向光学传感器图像中提取接触位置代理。最终结合深度视觉、本体/动作历史，通过 visuotactile transformer 推断隐变量。不是四路 binary，也不是输入完整触觉图像端到端训练策略。

论文正文与附录对部分奖励项正负号的书写也不完全一致；本报告只迁移明确的设计意图及课程结论，不将公式逐字照抄到项目。

## 5. Visual Dexterity：与当前桌面实验最相关的公开对照

官方仓库 <https://github.com/Improbable-AI/dexenv>，commit `ad9634e9d26cf5555d18728244a58f5b412c1eb2`。

论文的真实输入是关节编码器和 RealSense 深度图，转成统一点云。没有触觉传感输入。论文确实包括桌面支撑的重定向任务，然后增加桌面接触惩罚以学习空中操作；我们的桌面任务不应照搬后一个惩罚。

`dexenv/envs/rewards.py` 的公开实现：

- 姿态项 `rot_reward_scale / (abs_rot_dist + rot_eps)`。
- 指尖项 `ftip_reward_scale * mean(||p_tip-p_object||)`，负系数使指尖靠近物体中心。
- 能耗项 `-energy_scale * sum(abs(qdot * torque))`，可对能耗截顶。
- 成功 bonus、远离目标位置的失败 penalty。
- 可选桌面接触 penalty。

论文明确说，没有指尖接近物体这一项时，手指训练中几乎不接触物体。其消融还显示，能耗约束过大时学习会失败；系数为零时学得最快，但缺少能耗约束。这支持检查辅助惩罚对探索的影响，而不是盲目整体放大旋转奖励。

论文式 (3) 写平方距离求和，公开代码实际用欧氏距离均值；报告实现时必须保留这个差别。

默认 task YAML 中 `ftipRewardScale=-1`、`rotRewardScale=1`、`rotEps=0.1`、`reachGoalBonus=800`、`fallPenalty=-100`、`energy_scale=20`、`pen_tb_contact=False`。`fallDistance` 在 task YAML 是 0.24，但标准教师 `dclaw.yaml -> hand_default.yaml` 覆盖为 0.15。均非适合我们控制周期和硬件的推荐数值。

对当前“目标积累到限位”的问题，更值得做对照的是控制方法：

\[
q^{target}_{t+1}=\operatorname{clip}(q^{actual}_t+s\bar a_t,q_{min},q_{max}).
\]

`dexenv/envs/dclaw_base.py:882` 根据 `relativeToPrevTarget` 选择上一目标或当前实际关节角；默认设为 False，实际使用后者。论文 Action space 也明确给出相对当前关节角的公式。

它可以避免目标在关节被接触阻挡时仍不断积累到很远，但**不能保证关节永不到限位，也不能保证更适合 Allegro**。在 PD 控制下它还会改变保持接触力的方式，必须重新训练、检查接触力/力矩和旋转性能，不能把旧策略权重直接切换控制语义后比较。

## 面向当前项目的建议（待实验验证，非已经完成的修改）

按顺序处理，避免同时改多个机制后无法归因：

1. **先核实奖励与评估使用的接触语义。** 胶面有合力不必然代表接触目标物体。训练奖励可使用仿真的胶面—目标物体接触真值；actor 仍只能接收实机可获得的触觉/本体信息。object-only 真值不能偷偷成为部署输入。接触对提取在当前 Isaac Gym GPU 管线中的成本与可用性需实现前验证。
2. **控制方式对照优先于继续堆奖励。** 同一物体、种子、物理参数和任务目标，比较原目标累加与 Visual Dexterity 式实际关节角增量；记录真实关节移动、目标跟踪误差、限位裁剪率、胶面—物体接触和净旋转角。动作软边界 ±1.1 不会解决关节目标积累问题。
3. **用有出处的连续距离引导替代拍脑袋新增失触罚分。** Rotating without Seeing 与 Visual Dexterity 都支持这一思路。先作为独立对照，不能与接触奖励、控制器和失败罚分同时更换。针对 DIGIT，应明确用胶面有效区域的位置；若改用到物体表面的距离，就是我们的适配，而非原论文原式。不同尺寸应检查距离归一化与数值尺度。
4. **接触数量不是首要优化目标。** 若测试 AnyRotate 的两指阈值，应配套小权重/课程，并与不加接触奖励的对照比较。三四指额外奖励是否有益需消融，不能宣称文献已经证明“越多越好”。桌面支撑时一指推、其余指回撤也可能是有效中间动作。
5. **课程可借鉴，但保留用户已选任务定义。** AnyRotate/RotateIt 提供早期弱化稳定性约束的依据。用户当前不要求翻倒/非目标轴惩罚，因此不应因论文用了就添加。具体调度、接触距离系数及失败代价需按当前控制周期、典型角增量、回合长度重新标定。
6. **用任务指标选择模型。** 净目标轴转角、完整转圈数、物体保持在操作区的时间、真实物体接触率、裁剪率、运动/能耗分项，比总回报更能判断是否真的学会旋转。最终还要单独评估只有触觉和本体输入的学生。

最值得先借鉴的组合：以公开 Robot Synesthesia/Touch Dexterity 代码核对触觉基线，以 Visual Dexterity 设计桌面接近引导和控制器对照，以 AnyRotate/RotateIt 的课程研究解释并测试“抓住不动”局部最优。它们构成有来源的实验路线，不是某一篇方法的完整复现。

## 原始来源与复核入口

- Touch Dexterity 官方主页：<https://touchdexterity.github.io/>
- Rotating without Seeing 论文：<https://arxiv.org/html/2303.10880v4>，IV-A、附录 D。
- Robot Synesthesia 官方主页：<https://yingyuan0414.github.io/visuotactile/>
- Robot Synesthesia 论文：<https://arxiv.org/html/2312.01853v3>
- 共享库奖励函数：<https://github.com/YingYuan0414/in-hand-rotation/blob/1e5cc92864b87fbe60482a923002b5499ec5e799/isaacgymenvs/tasks/allegro_arm_morb_axis.py#L1987>
- 共享库默认配置：<https://github.com/YingYuan0414/in-hand-rotation/blob/1e5cc92864b87fbe60482a923002b5499ec5e799/isaacgymenvs/cfg/task/AllegroArmMOAR.yaml#L33>
- AnyRotate 官方主页：<https://maxyang27896.github.io/anyrotate/>
- AnyRotate 最新论文：<https://arxiv.org/html/2405.07391v3>，§3、§5.1、附录 B、F、H、I、J。
- AnyRotate PDF：<https://arxiv.org/pdf/2405.07391v3>；已保存并用 pdftotext 核对附录 B 的公式疑点。
- AnyRotate 网页仓库：<https://github.com/maxyang27896/anyrotate>，核查树 SHA `a6616fcb63d07d8c224d3077dfe7fba7a45bc604`。
- RotateIt 官方主页：<https://haozhi.io/rotateit/>
- RotateIt 论文：<https://arxiv.org/html/2309.09979v2>，§3.1、§3.2、Reward Hyperparameter。
- Visual Dexterity 官方主页：<https://taochenshh.github.io/projects/visual-dexterity>
- Visual Dexterity 论文：<https://arxiv.org/html/2211.11744v3>，Reward、Action space、Real-world observation、Ablation on the reward terms。
- Visual Dexterity 奖励代码：<https://github.com/Improbable-AI/dexenv/blob/ad9634e9d26cf5555d18728244a58f5b412c1eb2/dexenv/envs/rewards.py>
- Visual Dexterity 控制代码：<https://github.com/Improbable-AI/dexenv/blob/ad9634e9d26cf5555d18728244a58f5b412c1eb2/dexenv/envs/dclaw_base.py#L882>

HTML、论文文本、关键源码、仓库元数据与目录树保存在本目录。`hongiyoyo/rotating-without-seeing` 是第三方改版；`DeXTeRGT/RotateIt` 是无关同名项目；搜索引擎噪声结果均未作为论文结论依据。
