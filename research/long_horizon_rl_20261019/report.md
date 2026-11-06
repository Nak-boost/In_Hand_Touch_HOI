# 长程回撤与换指：强化学习方法调研

核查日期：2026-10-19。当前任务是桌面 Allegro＋DIGIT 释放、回撤、重新接触与持续旋转。此次仅调研，没有修改控制、奖励、网络或启动实验。本文选取直接相关方法，并非截至该日期的穷尽文献列表。

## 核心判断

回撤难点需要拆分：探索是否访问过有用的完整动作片段；延迟回报能否归因给早期动作；观测是否足够区分卸载/重新接近阶段；动作在当前接触几何中是否可行。延长 rollout 或加入历史分别只能改善部分问题，不能创造缺失的成功轨迹。

桌面操作可以由短周期“推动—释放—重新接近—再次推动”组成，不必预先规划整圈所有关节动作。训练应评估周期是否在真实连续回合内重复，而不是只在重置到有利姿态后成功一次。

## 原始来源与核实内容

1. **Learning Dexterous In-Hand Manipulation，2018（OpenAI）**
   - https://arxiv.org/abs/1808.00177
   - 全文：https://ar5iv.labs.arxiv.org/html/1808.00177
   - §4.2、§7：PPO、LSTM、非对称 actor-critic，无人工示范，出现 finger gaiting、多指协调和利用重力。不是纯触觉方案；这证明可行性，不保证当前感知/硬件/奖励与预算可直接复现。
2. **DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots，2024**
   - https://arxiv.org/html/2409.06613v2
   - §III-A：用示范中的状态构造不同难度起始条件；Zero-Variance Filtering 选择策略有时成功、有时失败的条件；优先目标分布与较早的示范状态，逐步减少对示范起点的依赖。
   - 原算法使用 MPO；教师的示范用途是初始化课程，不是必须克隆示范动作；后续视觉学生采用 BC 蒸馏。不可把视觉学生步骤与教师探索步骤混同。
   - 多指任务含 cube reorientation，但不是我们的 Allegro＋DIGIT 纯触觉连续旋转。完整 ZVF 丢弃大量探测数据，有计算代价；在 PPO 项目中先借鉴起始状态选择，不宣称完整复现。
3. **Reverse Curriculum Generation for Reinforcement Learning，2017**
   - https://arxiv.org/abs/1707.05300
   - 摘要确认：给定一个成功状态，自动生成越来越远的起始状态，不要求完整专家轨迹。接触动力学中的可行状态采样并不容易，不能随意扰动到穿透状态。
4. **Backplay，2018**
   - https://arxiv.org/abs/1807.06919
   - 摘要确认：沿示范从靠近结束的位置启动，逐步把起点往前移。主要实验是网格世界和 Pommerman，不是灵巧手实证。可作为解释倒序课程的参考，不能说已验证当前任务。
5. **Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations / DAPG，RSS 2018**
   - https://arxiv.org/abs/1709.10087
   - 全文：https://ar5iv.labs.arxiv.org/html/1709.10087
   - §IV-C：先 BC 初始化，再在自然策略梯度训练中加入随迭代衰减的示范项。不是 PPO 原实现；PPO＋BC 是方法适配。
   - 官方代码：https://github.com/aravindr93/hand_dapg （本次链接返回200）。
   - 能解决从未出现有用动作片段的问题；需要与项目控制接口一致的有效动作示范，小幅关节撤动检查不等于成功换指示范。
6. **Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning，CoRL 2019**
   - https://arxiv.org/abs/1910.11956
   - 全文：https://ar5iv.labs.arxiv.org/html/1910.11956
   - 从未分段的示范学习目标条件的分层策略，再用 RL 优化；高层子目标，低层执行固定步数。不要求每个最终任务的完整示范。
   - 主要证据来自厨房多阶段操作，不是换指专项实验。适合确实需要多个长阶段时；当前单轴周期任务优先试课程，避免过早增加层级。
7. **Learning by Playing — Solving Sparse Reward Tasks from Scratch / SAC-X，2018**
   - https://arxiv.org/abs/1802.10567
   - 摘要确认：同时训练辅助任务策略，用学习到的调度器执行辅助策略，促进稀疏奖励任务探索，使用 off-policy RL。
   - 对我们的借鉴是持续执行有意义的释放、接近等技能，而不是各关节每步加大独立噪声。辅助任务定义需要设计；并非直接在 PPO 中加一个奖励即可复现。
8. **First return, then explore / Go-Explore，Nature 2021（预印本2020）**
   - https://arxiv.org/abs/2004.12919
   - 摘要确认：记住有前景状态，先返回，再继续探索；另含稀疏奖励机器人抓放实验。可借鉴过程状态库，但全面迁移到随机接触物理不是小改动。
9. **AnyRotate，2024**
   - https://arxiv.org/html/2405.07391v3
   - §3.1、附录B.3：接触和稳定性项可能导致“抓稳但不转”，按成功进展逐步提高其课程权重。
   - §5.3/Fig.7：连续接触姿态/力有助于检测边缘接触与反应式换指；文中该反应未在其 proprioception/binary 条件下出现。仅说明其实验结论，不能推成二值触觉无法换指。
10. **RotateIt / General In-Hand Object Rotation with Vision and Touch，2023**
    - https://arxiv.org/html/2309.09979v2
    - Reward Hyperparameter：非目标轴惩罚从一开始设为−0.1会只学稳定握持，采用从0到−0.1的课程。当前桌面任务该项已关闭，因此不能照搬为本项目新的解决措施。
11. **Hindsight Experience Replay，2017**
    - https://arxiv.org/abs/1707.01495
    - 将达到的状态重新标为目标，有助于目标条件 off-policy RL；当前 on-policy PPO 不能直接混入旧轨迹当成新采样，且重标目标不自动产生未访问过的回撤状态。

## 当前项目建议（待实验，未实施）

第一优先：保留 PPO、动作接口和网络，测试 DemoStart/Backplay 思路的起始状态课程。需要先得到物理有效的完整“推动—释放—再接近—再次有效推动”轨迹。来源可以是仿真手动示范、经验证的脚本或规划轨迹，不必先采集大量实机数据。

训练状态从“重新接触后即将有效推动”开始，再扩展到“接近过程中”“已经释放”“回撤之前”；持续混入原任务起点。根据各组成功率调整采样；不指定未经验证的固定比例。接触多本身不等于阶段完成，重新接触后必须产生目标方向进展。

不能把任意几帧q值写进原静态抓取库就当成过程课程。需恢复物体与关节速度、PD目标、动作平滑器状态，必要时恢复观测历史；同时保持初始姿态惩罚的参考和目标定义一致，避免课程重置偷偷改变优化目标。PhysX接触缓存不保证完全可恢复，应检查短时重放与状态稳定性。

第二优先：若有完整动作示范而课程仍不足，增加 BC 初始化/衰减示范辅助项，作为单独实验。示范必须是当前actual_position增量和平滑后的控制语义，不能将关节轨迹直接当归一化动作。

第三优先：若短技能分别成功但无法组合，再研究 Relay/SAC-X式分层/技能调度。若使用技能状态，部署学生必须能从触觉/本体或自身指令历史获得它，不能隐式依赖物体仿真真值。

奖励课程可独立做对照，但当前保留的姿态项、两指接触奖励和门控不能在调研中未经授权删除。桌面无接触旋转门控0.1可能不利于推后自由转动，应根据目标动作是否包含此阶段决定消融。

## 时序参数边界

按此前冻结实验gamma=.99、tau=.95、20Hz：2秒后的折扣约.669，5秒后约.366；GAE系数乘积(.99*.95)^40约.086。后者是TD残差的trace权重，不等于未来奖励只剩8.6%，价值自举仍会传递收益。8步rollout=.4秒不是硬规划上限。增加rollout或历史不能替代探索；动作块/持续技能也需要正确的概率与学习实现，不能随便在PPO输出后叠加噪声而仍沿用旧log-prob。

## 必须用连续评估判断是否学会

从原始起点、不进行中途有利状态重置，统计完整推动循环次数、后半段旋转、释放后重新接触成功率、物体操作区保持率；覆盖多初始状态与随机种子。课程中某一后段成功不等于完整长程任务成功。

## 核查范围

主要正文已保存为 *_full.txt / HTML；AnyRotate和RotateIt全文单独保存。Reverse Curriculum、Backplay、Go-Explore、SAC-X、HER的本次核查以原始摘要为主。PDF下载发生截断，改用arXiv HTML/ar5iv全文，不宣称已核查那些未成功下载的PDF。通用搜索存在无关结果/验证码，因此不宣称穷尽2025—2026文献；以上方法的身份与关键内容均以原始论文页面核实。
