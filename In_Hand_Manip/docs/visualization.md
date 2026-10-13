# 可复用的策略回放与文件同步

`scripts/vis_manip.sh` 使用 `replay_policy.py`，根据检查点自动识别特权教师或触觉/本体学生。
它读取实验保存的配置，只做推理，不启动训练。`replay_teacher.py` 保留为兼容入口。
当前三组正式实验是特权教师，不能作为纯触觉学生部署；学生入口仅通过短程验证权重测试。

## 查看当前官方来源装配（不需要权重）

Git已包含网格、URDF和检查脚本。配置好本地环境后执行：

```bash
bash scripts/python_manip.sh scripts/inspect_digit_mount_sim.py --pose opposition
# 还可选择 open / half_closed / all。
```

来源、胶体弧面、碰撞近似和三姿态检查见[官方几何装配](official_digit.md)。
现有旧权重使用旧几何；新资产需要重新生成抓取和训练，不能用旧checkpoint验证新装配的旋转能力。
本次任务源码增加了外部网格哈希核验，历史权重的源码快照检查可能报差异。
严格复现旧实验请使用匹配的Git版本；理解该变更后可显式使用`--allow-source-mismatch`，
但不要将旧实验配置里的资产替换成新URDF，该参数也不会绕过资产哈希检查。

## 本地环境（首次配置）

在本地 Linux + NVIDIA 电脑上执行；假定已安装项目依赖。SDK 使用 Isaac Gym Preview 4。
如果本地仓库叫 `In_Hand_Touch_HOI`，将下面的项目目录改为实际克隆目录。

```bash
conda activate inhand_hoi
cd ~/Desktop/In_Hand_Reconstruction/In_Hand_Manip
# 替换为本地实际 SDK 根目录，其中应有 python/setup.py。
export ISAACGYM_PATH=/your/local/path/isaacgym
python -s -m pip install -c constraints-manip.txt -e "$ISAACGYM_PATH/python"
cat > .env.local <<ENV
MANIP_CONDA_PREFIX="$CONDA_PREFIX"
ISAACGYM_PATH="$ISAACGYM_PATH"
ENV
```

`.env.local` 位于 `In_Hand_Manip/`，不会提交到 Git；服务器路径不能直接复制到本地。
启动器隔离用户级 Python 包并配置 Isaac Gym 所需动态库。

## 按实验同步（在本地终端执行）

代码走 Git，权重和缓存走 rsync。只复制一个权重还不够：需要训练时的配置和对应抓取缓存。
清单脚本自动选择这些文件，并附带资产元数据与源码快照；不传日志和其他检查点。

```bash
cd ~/Desktop/In_Hand_Reconstruction
git pull --ff-only origin main

MANIP_RUN=teacher_sphere55_z_self_collision_v1
MANIP_CKPT=best

ssh -o ClearAllForwardings=yes TouchFM-vpn \
  "/root/conda_envs/inhand_hoi/bin/python /root/In_Hand_Reconstruction/In_Hand_Manip/scripts/replay_files.py --run-dir runs/manip/$MANIP_RUN --checkpoint $MANIP_CKPT" \
  > /tmp/inhand_replay_files.txt && \
rsync -avP -e 'ssh -o ClearAllForwardings=yes' \
  --files-from=/tmp/inhand_replay_files.txt \
  TouchFM-vpn:/root/In_Hand_Reconstruction/ ./

cd In_Hand_Manip
bash scripts/vis_manip.sh \
  --run-dir "runs/manip/$MANIP_RUN" --checkpoint "$MANIP_CKPT"
```

SSH 别名 `TouchFM-vpn` 使用 `root@10.13.21.41:20000`。`ClearAllForwardings=yes`
只关闭本次传输连接的额外端口转发，避免重复绑定17898端口，不影响已存在的代理连接。
若同步失败，请先处理错误，不要继续回放可能未完整更新的实验。

现有可选实验：

| 物体 | MANIP_RUN |
| --- | --- |
| 55 mm 球体 | teacher_sphere55_z_self_collision_v1 |
| 50 mm 立方体 | teacher_box50_z_self_collision_v1 |
| 70 mm 胶囊体 | teacher_capsule70_z_self_collision_v1 |

后续教师或学生实验只需替换 `MANIP_RUN` 和 `MANIP_CKPT`。
`best`/`last` 自动搜索实验根目录、`stage1_nn/`、`stage2_nn/`；同名有歧义时必须指定
`stage1_nn/best.pth` 或 `stage2_nn/last.pth`。也可指定其他已保存的检查点文件名。
清单生成要求文件位于项目根目录内，外部缓存会报错，需要先放入项目目录并调整配置。
运行中的实验建议选已完成写入、不会被覆盖的编号检查点，避免同步期间文件发生变化。
不要继续解压旧的 `teacher_replay_20261013.tar.gz` 覆盖新脚本；它是旧版教师专用快照。

## 回放参数

`--run-dir` 相对整个项目根目录，也接受绝对路径，与启动时的工作目录无关。
省略时可用现有实验的快捷入口：

```bash
bash scripts/vis_manip.sh --object sphere55 --checkpoint best
bash scripts/vis_manip.sh --object box50 --checkpoint last
bash scripts/vis_manip.sh --object capsule70 --checkpoint last
```

- 默认检查点为 `last`；默认一个环境对应一种物体，单物体实验就是一个环境，适合 GTX1650 4GB 起步。
- `--num-envs 4` 显示四个环境；数量不能少于保存配置中的物体类别数。
- 默认使用动作均值；`--sampled` 从高斯策略采样。
- `--steps 600` 限定600个控制步；默认持续回放，回合结束/失败自动重置，Esc或Ctrl-C退出。
- 球体没有纹理，默认叠加随物体转动的红/绿/蓝局部 X/Y/Z 坐标轴；`--no-axes` 隐藏。
- `--headless --steps 620` 检查无窗口推理；不会弹出窗口。
- `--device cuda:0` 和 `--graphics-device-id 0` 分别选择计算和图形设备。
- `--grasp-cache /absolute/path/cache.pt` 可重定位抓取缓存；兼容性校验仍执行。

终端输出回合时长、绕目标轴净旋转圈数和失败标志。净圈数不是完整姿态覆盖率。

## 兼容性与验证范围

读取训练时保存的任务配置，保留控制、物体、碰撞与随机化设置。检查资产和碰撞签名，
学生还检查关节顺序、限位、控制器与历史维度；环境检查抓取缓存的物体目录与关节信息。
若有对应推理源码快照，逐文件比较；缺失时打印提示。源码不一致默认中止，建议使用
匹配的代码版本。仅在明确理解后续改动时用 `--allow-source-mismatch` 显式允许差异；
它不跳过资产、碰撞和模型检查，也不能保证旧实验在新代码下行为完全相同。

服务器验证：15项测试通过；通用入口完成球体教师620步和学生80步的GPU无窗口推理及自动重置；
传输清单另做rsync副本逐文件SHA256检查。详见 `replay_validation.json`。
服务器没有图形桌面，GUI及本地GTX1650实际显示尚未验证。
当前三组正式权重训练于动作边界惩罚修正前，未学出可靠持续旋转，可能停转；
安装与推理成功不等于策略已收敛。
