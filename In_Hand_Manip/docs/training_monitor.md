# 训练监测与受控恢复

入口 `scripts/monitor_manip.py`，适用于 `stage=teacher`。它是独立启动入口，不会自动接管已有训练或等待队列。

```bash
scripts/python_manip.sh scripts/monitor_manip.py \
  --config manip_hora16384_sphere55 --gpu 1 \
  --output ../research/hora16384_monitored_next \
  --run-prefix teacher_sphere55_hora16384_monitored \
  --max-steps 1500000000 --save-frequency 50 --max-restarts 2 \
  --checkpoint ../runs/manip/teacher_sphere55_z_hora16384_pairfilter_20261016/stage1_nn/best.pth
```

这是下次手动启动示例，本次实现不自动启动正式训练。省略 `--checkpoint` 为从头训练。恢复旧权重前仍须检查任务、手模型和控制兼容性。输出目录及每次尝试的 run 目录必须不存在，避免覆盖历史结果。终端断开时需用 tmux 或 nohup 保持 supervisor 存活；整个主机/容器退出后的自动拉起需要外部服务管理，不在此脚本范围内。

## 监测内容

- 默认每10秒记录整卡显存、GPU利用率、温度、可见的子进程显存、主机进程RSS；无法看到进程显存时记为null，不冒充0。
- 心跳记录初始化、采样、PPO更新、检查点保存、训练完成和销毁仿真阶段，以及实际步数和轮次。
- 默认初始化30分钟无进展、训练10分钟步数不增加触发停滞处理。阈值按真实每轮耗时调整，不能设得比一轮训练还短。
- 启动需至少12000 MiB可用显存；运行剩余显存低于2048 MiB或磁盘低于5 GiB时停止本实验。只管理本脚本创建的子进程会话。
- 启用 Python faulthandler；停滞时先发 SIGUSR1 请求所有Python线程栈，再依次SIGINT/TERM/KILL。同时记录退出码、末尾日志、nvidia-smi与内核日志（权限不足会如实记录）。TORCH_SHOW_CPP_STACKTRACES仅辅助PyTorch异常，不能保证原生Isaac Gym SIGSEGV得到C++完整栈；脚本不默认生成可能很大的core dump。

`status.json` 给出当前状态、PID及尝试历史；每次尝试有 `console.log`、`heartbeat.json`、`resources.jsonl`，异常时另有 `incident.json` 与恢复选择记录。

## 保存与恢复规则

监控入口默认每50轮保存，而原训练配置文件仍保留原来的保存间隔。16384×8规模下，50轮约655万环境步，以6700步/秒估算约16分钟。频率是轮数而非墙钟时间，极慢的一轮仍可能更久。

教师检查点包含模型、优化器、输入/价值归一化、已采样步数、轮次和自适应学习率。先写同目录临时文件并fsync，再原子替换目标文件；中断不会先覆盖原检查点。断电级目录元数据持久性不作保证。恢复不包含物理场景、完整随机数状态和回合轨迹，因此不是逐位精确续跑。

默认 `--max-restarts 0` 只记录；显式设置2表示最多自动重启两次。只对SIGSEGV、SIGABRT或无进度超时尝试恢复；OOM/Python普通错误、显存/磁盘不足、用户停止均不盲目重试。按检查点内部 `agent_steps` 选最新完整、有限值的教师检查点，不按文件名或mtime选择。坏文件跳过并记录。连续从同一步数恢复后再次失败且没有新检查点则停止，避免无限循环。

重启前检查训练源代码、配置和启动脚本摘要；发现编辑就停止自动恢复。手模型/碰撞/控制兼容性仍由训练加载校验。每次恢复写入独立run目录，最大步数是包含旧进度的累计预算。

只有训练完成心跳及完整的最终检查点均满足总预算时，才把销毁阶段异常单独标为 `completed_with_cleanup_failure`；这仍返回非零退出码，不冒充正常退出，也不无意义地重复训练。

监控降低丢失进度并保留诊断证据，不能修复底层内存错误，不能在SIGSEGV之后可靠地临时抢存损坏进程内的模型。
