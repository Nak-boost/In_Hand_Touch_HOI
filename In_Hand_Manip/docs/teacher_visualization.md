# 本地策略可视化

教师回放已扩展为教师/学生通用入口。请使用 [策略可视化与文件同步](visualization.md)。

```bash
bash scripts/vis_manip.sh --object sphere55 --checkpoint best
```

旧 `replay_teacher.py` 仍转发到通用入口。后续实验使用 `--run-dir runs/manip/实验名`，
路径相对整个项目根目录；代码通过Git更新，权重与匹配抓取缓存通过文档中的rsync清单同步。
旧 `teacher_replay_20261013.tar.gz` 是冻结快照，不要用它覆盖新的可视化脚本。
