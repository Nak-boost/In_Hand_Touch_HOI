# 强遮挡下手内物体位姿跟踪调研

核对日期：2026-10-16。任务为文献调研，未修改训练配置或进程。

## 结论

ECCV 2026 官方列表确认收录 Point2Pose 和 GenHOI。Point2Pose 最接近“视觉下手持旋转、强遮挡、物体位姿跟踪”的描述，但尚不能确定用户记忆中的论文就是它。它使用 RGB-D，不是仅 RGB；完全遮挡后的恢复不代表遮挡期间准确观测到物体运动。GenHOI 研究单张 RGB 的手—物体姿态估计，不等同于机器人连续手内旋转闭环跟踪。

## Point2Pose

完整标题：Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for Multiple Unknown Objects Via 2D Point Trackers。

作者：Tzu-Yuan Lin, Ho Jae Lee, Kevin Doherty, Yonghyeon Lee, Sangbae Kim。

- 官方会议页：https://eccv.ecva.net/virtual/2026/poster/5929
- 项目：https://point2pose.github.io/
- 论文：https://arxiv.org/abs/2604.10415
- 已读全文版本：https://arxiv.org/html/2604.10415v1
- 公开代码：https://github.com/tzuyuan/point-to-pose

方法：少量点击初始化，SAM2 分割，长时 2D 点跟踪（论文实现为 causal BootsTAPIR），深度反投影建立物体坐标系关键点地图，多假设三维配准、在线图优化和 TSDF 几何约束，边跟踪边重建。不同对象的点可在同一次点跟踪中处理。

无需预先 CAD 或类别先验。论文假设固定相机参考系；并非无学习方法，使用预训练分割、点跟踪等模型。

项目页 2026 年 9 月更新声称 RTX 4090 上达到 30 Hz。仓库实时配置和论文高精度配置不同（例如 SAM2 small/large、点跟踪分辨率），不可把实时速度与论文所有精度结果视为同一设置。未在本机复现速度。

评估包括 HO3D、YCBInEOAT、新增 YCBMultiTrack；项目有手持旋转和遮挡后恢复演示。未据此确认在 Allegro+DIGIT 持续换指旋转中闭环验证。

限制：弱纹理/重复纹理使点跟踪退化；依赖深度和分割质量；TSDF 重建精度存在取舍。官方摘要明确承认以部分单物体精度换取多物体和遮挡恢复能力。不要将其概括为全面优于 BundleSDF。

## 其他相关方法

### GenHOI（ECCV 2026）

GenHOI: Generalized Hand-Object Pose Estimation with Occlusion Awareness。

官方：https://eccv.ecva.net/virtual/2026/poster/4706

已下载会议 PDF：https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/7114.pdf

单张 RGB 输入，利用层级语义描述、RGB/预测点云/文本的掩码建模，以及人手关节与旋转先验。DexYCB、HO3Dv2 实验。预测点云不等于传感器实测深度；人手先验也不能直接作为 Allegro 模型。适合作为遮挡推理相关工作，不应当直接认定是实时连续机器人位姿跟踪器。

### ComPose（2026 预印本；未确认 ECCV 收录）

ComPose: When to Trust Hands for Object Pose Tracking。

https://arxiv.org/abs/2605.23523
https://arxiv.org/html/2605.23523v1

仅 RGB 视频。将物体几何估计与人手运动结合，选择可靠手关节并自适应融合，时序校正减轻漂移。使用 Flow3R、WiLoR 等基础模型；SAM3D-Object 初始化位姿和物体网格，不要求用户提供真实 CAD。论文有周期性重新锚定，不能仅由“视频跟踪”推断完整系统为严格因果低延迟闭环。

评估平移轨迹使用全轨迹 Umeyama 对齐消除尺度和坐标差异，不能将这些结果直接等同于相机标定坐标系下的在线绝对公制位姿精度。论文列出重抓、未见抓握分布、长序列漂移等限制。

### FoundationPose（CVPR 2024）

https://nvlabs.github.io/FoundationPose/

RGB-D 6D 位姿估计/跟踪；支持给定 CAD 或通过参考图像建立物体表示。无需对每个新物体微调，不等于完全不需要物体先验。适合作为已知几何条件下的视觉基线。

### BundleSDF（CVPR 2023）

https://bundlesdf.github.io/

RGB-D 未知物体位姿跟踪和神经隐式重建，首帧分割初始化。与本项目“位姿和形状共同估计”的问题更接近，是重要视觉基线。完全遮挡后的恢复仍是后续方法重点改进方向。

### KineFuse（IROS 2026）

https://arxiv.org/abs/2607.14842
https://arxiv.org/html/2607.14842v1
https://cold-young.github.io/kine-fuse/

FoundationPose 视觉表示融合本体、二值接触和稀疏近端力/力矩信息，按手指运动学组织特征。已知物体 CAD 和手眼标定。不是纯视觉或纯触觉方法。

证据边界：论文量化实验为单个铅笔形物体的仿真，真实实验为定性展示；顺序跟踪报告约 47.6 度平均角误差。不能把“相对视觉有提升”解释为任意物体的高精度位姿问题已解决。

## 对 Allegro + DIGIT 项目的判断

1. 优先评估 Point2Pose 和 BundleSDF 作为未知物体视觉基线；已知 CAD 的条件下加入 FoundationPose。
2. 必须区分人手拿着物体整体运动与物体相对机器人手掌旋转；后者存在换指、滚动、滑动，手运动不能直接代替物体运动。
3. 对完全不可见物体的任意未知运动，视觉没有新测量约束；预测和重现后的恢复均不等于持续准确观测。
4. 对无纹理完全球体，自转几何不可辨识；视觉或几何触觉都不能凭空恢复唯一朝向。需要非对称特征、表面纹理、切向运动信息及相应假设，或使用对称性合理的评估。
5. 建议实验分别评估部分遮挡、完全遮挡及恢复、持续换指、弱纹理/对称物体；报告旋转角误差、漂移、丢失率、恢复时间及实际闭环成功率。ADD-S 可掩盖对称轴旋转误差，不能单独用于证明转角跟踪可靠。
6. 视觉跟踪器可以提供基线或有置信度的辅助标签，不应未经独立验证当作强遮挡条件下的真值。

以上为资料核对与方法分析，未安装或运行这些算法。
