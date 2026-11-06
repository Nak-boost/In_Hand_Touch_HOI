# 当前版本四组正式训练

启动核查：2026-10-19T03:37:39-05:00（America/Chicago）。

已停止本项目旧的三个训练进程及两个自动续训调度器，保存进程清单和历史检查点清单。四组均从头训练特权教师 PPO，不加载旧权重。

## 运行分配

| 任务 | 物理 GPU | 训练 PID | 核查时显存 MiB | 实时监控状态 |
|---|---:|---:|---:|---|
| 悬空六物体混合 | 0 | 585722 | 6422 | [air_six](air_six_monitor/status.json) |
| 悬空55 mm球 | 1 | 585715 | 5884 | [air_sphere55](air_sphere55_monitor/status.json) |
| 桌面55 mm球 | 2 | 585732 | 5916 | [table_sphere55](table_sphere55_monitor/status.json) |
| 桌面45 mm立方体 | 3 | 585706 | 5904 | [table_box45](table_box45_monitor/status.json) |

GPU编号已通过训练PID对应的GPU UUID与nvidia-smi物理编号交叉核验；子进程内cuda:0为CUDA_VISIBLE_DEVICES映射后的局部编号。

## 固定设置

- 每组16384环境；horizon_length=8，minibatch_size=32768，mini_epochs=5，初始学习率0.005，gamma/tau=0.99/0.95。
- 策略网络256/256/128，特权编码器128/64/8；当前actual_position动作基准、PD、物理参数、碰撞资产保持。
- 单组预算1,500,000,000环境步；保存频率沿用当前配置500轮，期间另保存reward-best。
- 四组均为当前+Z轴旋转配置；正向连续旋转接触倍率[0.1,0.5,1.0]；离散目标奖励仍未门控。
- 桌面terminate_on_contact_loss=false：失触只统计，不触发失败；越界失败和回合超时截断保留。
- 两指接触奖励保留；本次没有新增重新接触奖励、示范、课程或分层策略。
- 六物体是已有sphere55、sphere65、box45、box55、capsule70、capsule80组合；并行环境固定对象身份并平衡分配，重置随机采样抓取/配置规定的参数，不是随机生成任意形状。
- 使用10月18日复验缓存：悬空球225个；混合六物体225/217/77/70/207/221个；桌面球116个，立方体96个。

## 冻结与验证

495份文件（源码、配置、资产、缓存等）复制到frozen_project，哈希见frozen_sha256.json；工作区未提交的当前修改也已包含。后续工作区修改不会热加载到本次训练。runs目录链接到项目统一runs目录。
49项相关测试通过。四组缓存控制契约、碰撞签名、资产哈希、网络尺寸和解析配置均通过启动前核对。四组均已完成多轮真实PPO采样/优化、写出可读且数值有限的检查点；训练损失及接触对CPU耗时已记录。详见startup_verification.json。
启动核查约65万至79万步，各组已在运行；早期指标不能作为策略已经学会旋转的证据。
既有monitor_manip.py分别持续记录显存、GPU利用率、进度、异常和心跳。显存余量低于2048 MiB或长时间无进展会停止对应进程；自动重试次数为0，避免静默重复失败。

## 文件入口

- plan.json：本次配置和预算；launch.json：调度器PID与完整启动命令。
- *_monitor/status.json：各组持续更新的状态；attempt00/resources.jsonl：资源时序。
- *_monitor/attempt00/console.log：训练日志；同目录heartbeat.json：最新训练进度。
- *_resolved.yaml：启动前解析配置；实际训练目录config.yaml：生效配置。
- preserved_old_checkpoints.json：原训练已保存的检查点，不承诺中断时尚未保存的步数也被保留。
