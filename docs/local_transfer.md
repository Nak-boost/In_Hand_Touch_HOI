# 将代码与当前训练结果迁移到本地

当前代码分支为 `feature/allegro-digit-two-axis`。代码基于 HORA，保留其
LICENSE 与上游历史。`scripts/vis_s1.sh` 是上游任务入口，不能直接用于当前
Allegro + DIGIT 教师。本项目的 `stage=evaluate` 入口只接受学生权重；
当前三组正式结果均为特权教师，尚无收敛的学生策略。

## 代码与结果

Git 保存代码、手模型、配置、依赖清单及验证文档。
训练权重与初始抓取缓存单独迁移，避免将不断增长的训练输出加入 Git 历史。

本次结果包位于服务器：

```
/root/In_Hand_Reconstruction/transfers/manip_training_20261013.tar.gz
/root/In_Hand_Reconstruction/transfers/manip_training_20261013.tar.gz.sha256
```

包含球体、立方体、胶囊体三组训练的 `best.pth`、`last.pth`、有效配置、
训练源码快照和资产元数据，三个抓取缓存，以及完整训练分析目录。
包内 `transfer_manifest_20261013.json` 记录各文件 SHA256。
没有包含 Isaac Gym SDK、Conda 环境、全部中间权重或 TensorBoard 事件文件。

这些权重在动作边界损失修正之前训练，不能描述为采用修正公式训练的策略。
本次 Git 代码的损失修正不会改变旧权重的推理行为。

## 本地目录

在本地创建父目录 `In_Hand_Reconstruction`，从自己的 GitHub 仓库克隆此分支到
`In_Hand_Reconstruction/In_Hand_Manip`。下载结果包及校验文件到父目录后：

```bash
cd In_Hand_Reconstruction
sha256sum -c manip_training_20261013.tar.gz.sha256
tar -xzf manip_training_20261013.tar.gz
```

解压后代码仍在 `In_Hand_Manip/`，缓存位于其 `cache/`，训练配置/权重位于
`runs/manip/`，分析报告位于 `research/training_review_20261013/`。
请在新建的本地目录解压，避免覆盖本地同名实验。

## 本地运行依赖

直接运行当前 Isaac Gym Preview 4 仿真回放需要兼容的 Linux x86_64、
NVIDIA GPU/驱动，以及已安装的 Isaac Gym SDK。Windows/macOS 不能按当前
服务器启动命令直接原生运行；图片/PDF 训练曲线和 Markdown 报告可直接查看。

环境版本与安装约束见 `environment.md`、`requirements-manip.txt` 和
`constraints-manip.txt`。`environment/` 是服务器环境审计记录，含本机路径，
不是本地一键安装文件。

在本地仓库根目录创建不提交 Git 的 `.env.local`，填入本地真实路径：

```bash
ISAACGYM_PATH=/your/local/path/isaacgym
MANIP_CONDA_PREFIX=/your/local/path/conda/envs/inhand_manip
```

先用 `scripts/python_manip.sh -m pytest -q tests` 检查环境。当前迁移准备
不意味着已经验证本地 GUI、相机视角或教师可视化入口；后续回放应加载结果包
中保存的训练配置，保留手模型与碰撞签名检查，并明确标记特权教师身份。
