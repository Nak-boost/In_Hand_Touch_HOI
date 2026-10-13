# In-Hand Touch HOI

本仓库统一管理手内操作、触觉位姿/形状重建及后续闭环集成。
Git 根目录为 `In_Hand_Reconstruction/`；各模块是普通子目录，不是嵌套仓库或 submodule。

| 目录 | 内容 |
|---|---|
| [In_Hand_Manip](In_Hand_Manip/README.md) | 基于 HORA 的 Allegro + DIGIT 操作策略、仿真资产与验证 |
| [Shape_Pose_Reconstruction](Shape_Pose_Reconstruction/README.md) | 动态位姿与形状联合估计，尚未实现 |
| [research](research/) | 文献调研、训练分析脚本、报告与精简评估结果 |
| `runs/` | 各模块的运行产物；本地保留，不提交 Git |
| `transfers/` | 本地迁移包与备份；不提交 Git |

当前操作模块已实现教师 PPO、触觉/本体历史学生蒸馏和自碰撞模型。
三组单物体 Z 轴教师已完成训练，但尚未学出可靠的持续旋转；详见
[训练复盘](research/training_review_20261013/report.md)。
最新代码已修正 PPO 动作边界惩罚，现有训练权重来自修正之前的运行。
完整操作—重建闭环尚未实现。

## 克隆与运行

```bash
git clone https://github.com/Nak-boost/In_Hand_Touch_HOI.git In_Hand_Reconstruction
cd In_Hand_Reconstruction/In_Hand_Manip
```

环境配置与运行入口见 [操作模块 README](In_Hand_Manip/README.md) 和
[环境说明](In_Hand_Manip/docs/environment.md)。
克隆只获取代码、配置、资产和文档；训练权重、抓取缓存与完整日志不会上传。
本地回放还需另行取得所需权重/缓存，并准备兼容的 Isaac Gym 环境。

## 代码来源

操作模块基于 [HORA](https://github.com/HaozhiQi/hora)，保留原有提交历史、
[MIT License](In_Hand_Manip/LICENSE) 和版权声明。
`upstream` 指向 HORA，`origin` 指向本项目 GitHub 仓库。
