# 本地回放现有教师策略

当前三组正式权重是特权教师，使用物体真值输入；这不是纯触觉学生部署。
`scripts/vis_manip.sh` 只加载模型推理，默认一个环境，不启动 PPO 或蒸馏。
`scripts/vis_s1.sh` 属于原 HORA 任务；`stage=evaluate` 只接受本项目学生。

## 先确认本地 SDK 和环境

在本地操作模块目录运行以下命令。`ISAACGYM_PATH` 必须改为本地实际的
Isaac Gym Preview 4 根目录（包含 `python/setup.py`），不是服务器的路径。

```bash
conda activate inhand_hoi
cd ~/Desktop/In_Hand_Reconstruction/In_Hand_Manip
export ISAACGYM_PATH=/your/local/path/isaacgym
python -s -m pip install -c constraints-manip.txt -e "$ISAACGYM_PATH/python"
cat > .env.local <<EOF
MANIP_CONDA_PREFIX="$CONDA_PREFIX"
ISAACGYM_PATH="$ISAACGYM_PATH"
EOF
scripts/run_manip.sh stage=smoke num_envs=1 headless=false evaluation_steps=100
```

## 获取文件

服务器已准备 `transfers/teacher_replay_20261013.tar.gz` 及同名 `.sha256` 文件。
该包只在服务器本地保存，不上传 GitHub。它包括：

- 新增的 `In_Hand_Manip/replay_teacher.py`、`scripts/vis_manip.sh` 和本文档；
- 三种物体的 best/last 教师权重与各自的有效训练配置、资产元数据；
- 三个对应抓取缓存及用于比对的推理源码快照；
- 包内 `teacher_replay_manifest_20261013.json`，记录每个文件的 SHA256。

可通过 VS Code Remote 文件浏览器下载，或用本地能连接服务器的 SSH 地址执行 scp。
将压缩包和校验文件放在本地项目根目录（与 `In_Hand_Manip/` 同级），执行：

```bash
cd ~/Desktop/In_Hand_Reconstruction
sha256sum -c teacher_replay_20261013.tar.gz.sha256
tar -xzf teacher_replay_20261013.tar.gz
```

只有校验显示 OK 才解压。包按整个项目的目录结构保存，不能在 `In_Hand_Manip/`
内部解压；解压前确认没有需要保留的同名本地回放文件。包已包含新增脚本，因此
不依赖 GitHub 连接恢复后才能开始回放。它不会包含或覆盖 `.env.local`。

## 打开窗口

```bash
cd ~/Desktop/In_Hand_Reconstruction/In_Hand_Manip
bash scripts/vis_manip.sh --object sphere55 --checkpoint best
bash scripts/vis_manip.sh --object box50 --checkpoint last
bash scripts/vis_manip.sh --object capsule70 --checkpoint last
```

每次运行一条命令，退出窗口后再换物体。默认采用动作均值并持续回放，失败或
30 秒回合结束时自动重置。Esc 或 Ctrl-C 退出。`--num-envs 4` 可显示四个环境；
GTX 1650 4 GB 建议先用默认的一个环境。球体外观缺少纹理，默认绘制随物体姿态
变化的红/绿/蓝局部 X/Y/Z 坐标轴；这只是显示辅助，不改变碰撞或惯量。

其他参数：`--sampled` 使用训练时的高斯动作采样；`--steps 600` 在600个控制步后
退出；`--no-axes` 隐藏坐标轴；`--run-dir /path/to/saved_run` 指定其他完整实验目录。
没有图形桌面时可先检查 `--headless --steps 620`，但这不会弹出窗口。

脚本读取训练时保存的配置，不会套用当前默认任务参数。模型/碰撞签名、抓取
物体类别和关节顺序由加载检查验证；若包内有对应推理源码快照，也会逐文件比对。
本轮教师在动作惩罚修正前训练，尚未学出可靠的持续旋转；出现停转不是环境安装成功
与否的判据。单环境/新硬件的轨迹也不保证与服务器256回合评估逐帧一致。

服务器验证覆盖三个物体的一环境、620步无窗口推理和自动重置。
服务器没有图形桌面，实际 GUI 显示与本地 GTX 1650 仍需在本地验证。
