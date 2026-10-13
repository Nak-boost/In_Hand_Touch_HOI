# 环境与依赖管理

## 当前环境

本项目使用 `/root/conda_envs/inhand_hoi`，是操作、重建及后续集成共用的独立 Conda 环境。
2026-10-13 从原名 `inhand_manip` 迁移，保留已安装包版本。完整重建链路仍需后续验证。
源环境是 `/root/conda_envs/active3d`；源环境和 TouchFM 代码未修改。
旧项目 `.venv` 已于 2026-10-12 删除，启动脚本使用独立 Conda 环境。

Linux x86_64；Python 3.8.20；PyTorch 2.0.1 + CUDA 11.8；NumPy 1.23.5。
Isaac Gym Preview 4 来自 `/remote-home/ums_hanminghao/Software/isaacgym`，
该厂商 SDK 外部路径是显式依赖，不代表 Python 环境继续依赖 active3d。
TACTO 没有安装到训练流程，原因见 `tactile_design.md`。

`scripts/run_manip.sh` 调用 `scripts/python_manip.sh`，默认选择以上 Conda 环境。
后者配置 Python/原生库路径并禁用 user site，不继承外部 PYTHONPATH。
本机 `.env.local` 指定 `ISAACGYM_PATH` 与 `MANIP_CONDA_PREFIX`，不提交 Git。
可用环境变量 `MANIP_CONDA_PREFIX` 或 `MANIP_PYTHON` 显式覆盖。

## 复制方式

原始 `conda create --clone --copy` 因已清理的包缓存和在线大文件下载中断失败。
最终使用 conda-pack 0.7.1，从本地安装文件打包，并重定位到新路径：

```bash
conda-pack -p /root/conda_envs/active3d \
  -o /tmp/inhand_manip_clone.tar --format tar \
  --dest-prefix /root/conda_envs/inhand_manip --ignore-editable-packages
mkdir -p /root/conda_envs/inhand_manip
tar -xf /tmp/inhand_manip_clone.tar -C /root/conda_envs/inhand_manip
```

以上仅记录本次复制过程，不要向已有环境重复解压。
使用 `--dest-prefix` 后无需 conda-unpack；新环境已注册到 Conda。
原环境的旧 pterotactyl editable 引用指向不存在的外部源码目录，未将该项目源码复制进来；
当前操作策略和已验证的 TouchFM 编码器不依赖该 editable 引用。

## 项目环境名称与本地改名

统一使用 `inhand_hoi`，表示操作、重建与集成共用的项目环境。
环境名不改变依赖覆盖范围：本地新装的操作依赖仍需后续补齐重建依赖。

本地按安装步骤新建的命名环境可以在退出该环境后改名：

```bash
conda deactivate
conda rename -n inhand_manip inhand_hoi
conda activate inhand_hoi
```

更新本地 `In_Hand_Manip/.env.local` 中的 `MANIP_CONDA_PREFIX` 为激活后的
`$CONDA_PREFIX`，`ISAACGYM_PATH` 保留本地 SDK 路径。改名后运行 `pip check`、
项目测试和少量环境的 smoke；旧绝对路径的 editable 安装可能需要重新注册。

服务器环境具有历史 Conda 元数据与 pip 覆盖层，因此此次使用 conda-pack
重定位实际存在的文件到 `/root/conda_envs/inhand_hoi`，不重新安装旧 Conda 包。
历史元数据中缺失的旧版本文件不会重新补回；迁移前后实际 Python 包列表必须一致。
原始复制命令与旧验证记录保留当时的路径，不能当成当前启动路径。

## 冲突处理原则与实际调整

优先保证当前 HORA/Isaac Gym 训练链路，同时保留 TouchFM 可兼容的依赖。
尽量维持共享的 Python/PyTorch/CUDA ABI；不为对齐上游历史 requirements 全面降级。
若只需局部、等价的兼容修改，应先评估该修改与依赖变更的影响范围。
若后续重建模型必须使用互不兼容的 Python/CUDA 栈，可以保留模块环境边界并通过
明确的观测/位姿/动作接口融合，不应强制改变两个系统的核心算法。

| 项目 | 最终处理 | 原因 |
|---|---|---|
| Python/PyTorch/CUDA/PyTorch3D | 保留 3.8.20 / 2.0.1 / 11.8 / 0.7.7 | 保留原有 CUDA 扩展兼容性 |
| NumPy | 1.24.4 → 1.23.5 | 厂商 Isaac Gym 使用已删除的 np.float；不直接改共享 SDK |
| typing-extensions | 统一为 4.13.2 | 满足原 cryptography 47.0.0，仍支持 Python 3.8 |
| cryptography | 保留原 47.0.0 | 无需降级；避免不必要的版本偏移 |
| setuptools / wheel | 60.10.0 / 0.37.1 | 保留 pyembree 0.2.11，满足它的依赖约束 |
| cmake | 补装 3.31.6 | 补齐原 triton 2.0.0 声明的依赖 |
| Hydra/Gym/TensorBoardX/Pytest | 安装 requirements-manip.txt 中的版本 | 新操作训练入口与验证所需 |

源环境有 setuptools、typing-extensions、pip 的重复版本元数据。
在新环境中重装统一 setuptools/typing-extensions；保留实际 pip 24.3.1，
将多余的 pip 25.0.1 元数据移到环境下 `.migration_backup/`，不参与包解析。
这些是新环境内部清理，不修改源环境。

## 日常添加依赖

```bash
cd /root/In_Hand_Reconstruction/In_Hand_Manip
conda activate /root/conda_envs/inhand_hoi
python -m pip install -c constraints-manip.txt 'package==version'
python -m pip check
```

也可不激活，使用 `scripts/python_manip.sh -m pip ...`。
将新包及版本写入 `requirements-manip.txt`；该文件自动引入 constraints。
约束冲突时先看冲突链，选择兼容版本；不要用 `--no-deps` 或任意放开 torch/numpy 版本掩盖问题。
对于需要新版编译工具的包，先在临时构建环境构建 wheel，再评估运行依赖；
不要为了构建一个包直接升级本项目 setuptools/wheel 而破坏 pyembree 的要求。

本环境包含 Conda 管理的底层包和 pip 安装覆盖层，避免 `conda update --all`。
确需修改 Python/PyTorch/CUDA 时，先复制测试环境并重新验证 CUDA 扩展。

依赖变更后：

```bash
scripts/python_manip.sh -m pytest -q tests/test_manipulation.py
scripts/run_manip.sh stage=smoke num_envs=6 evaluation_steps=30
scripts/python_manip.sh scripts/snapshot_environment.py
```

`environment/` 保存底层 Conda 清单、当前 Python 包版本和来源。
它们用于版本审计；pip-freeze 中含有源环境的本地编译路径，不能当成跨机器一键安装锁文件。
跨机器完整迁移需同时准备 Isaac Gym SDK、CUDA 驱动兼容性以及本地扩展源码/二进制。

## 验证范围

验证结果记录在 `environment/validation.json`。
TouchFM 检查覆盖触觉编码器前向、数据模块导入和 PyTorch3D CUDA 算子，
不等同于完整 TouchFM/TRELLIS 重建实验验证，也不代表操作策略已收敛。
