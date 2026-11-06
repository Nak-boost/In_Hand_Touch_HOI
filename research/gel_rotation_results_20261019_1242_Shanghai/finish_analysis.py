from pathlib import Path
import json, math, hashlib, subprocess, datetime

P = Path(__file__).resolve().parent
OLD = P.parent / 'gel_rotation_results_20261018_0728_Chicago'
FIRST = P.parent / 'gel_rotation_results_20261018'
def read(p):
    return json.loads(p.read_text())
def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)
def aggregate(d):
    rows = list(d['per_object'].values())
    n = sum(x['n'] for x in rows)
    out = {'n': n, 'steps': d['checkpoint_steps']}
    for k in ['retained_30s', 'one_turn_retained', 'mean_turns']:
        out[k] = sum(x[k]*x['n'] for x in rows)/n
    survivors = sum(x['n']*x['retained_30s'] for x in rows)
    for k in ['last5s_speed_survivors', 'last5s_abs_speed_lt_005_survivors']:
        out[k] = sum(x[k]*x['n']*x['retained_30s'] for x in rows)/survivors
    return out

s = read(P/'training_summary.json')
prev = read(OLD/'training_summary.json')
last = read(P/'eval_multi_last_215.json')
best = read(P/'eval_multi_best_215.json')
previous = read(OLD/'eval_multi_last_215.json')
first = read(FIRST/'eval_multi_last_215.json')
processes = read(P/'evaluation_processes.json')
assert len(processes) == 6
for rec in processes:
    assert rec['result_saved']
    assert rec.get('returncode', 0) == 0
    d = read(P/f"eval_{rec['group']}_{rec['checkpoint']}_{rec['seed']}.json")
    assert finite(d)
    assert d['checkpoint_sha256'] == hashlib.sha256((P/f"{rec['group']}_{rec['checkpoint']}.pth").read_bytes()).hexdigest()
    assert sum(r['n'] for r in d['per_object'].values()) == (384 if rec['group']=='multi' else 64)
assert all(not v['nonfinite_tags'] for v in s.values())
comparison = {label: aggregate(d) for label,d in [('first',first),('previous',previous),('best',best),('last',last)]}
(P/'comparison.json').write_text(json.dumps(comparison,indent=2))
status = {'time':datetime.datetime.now().astimezone().isoformat()}
for key,cmd in {
    'multi_process':['ps','-p','224934','-o','pid,etime,args'],
    'gpu':['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],
    'multi_log_tail':['tail','-3','/root/In_Hand_Reconstruction/research/gel_rotation_pair_20261017/teacher_six_objects_gel16384_direct_20261017_152146.log'],
}.items():
    r = subprocess.run(cmd,capture_output=True,text=True)
    status[key] = {'returncode':r.returncode,'output':r.stdout,'stderr':r.stderr}
(P/'run_status_snapshot.json').write_text(json.dumps(status,indent=2))

w = s['multi']['windows']['last10M']; ow = prev['multi']['windows']['last10M']
lines = [
'# 手内旋转训练更新分析（2026-10-19，Asia/Shanghai）',
'',
'本次分析 GPU 0 六物体组和 GPU 1 单球体组。结论：六物体组继续训练到约9.23亿步后，仍未形成持续旋转；平均净转角有小幅增加，但近期训练回报回落。单球体没有新增训练结果。',
'',
'取样时六物体日志为923,271,168步，预算1,500,000,000步，完成61.6%；固定回放last为917,504,000步，best为656,015,360步。训练仍在运行。历史最高滚动回报21.11，不能当作旋转成功率；best按最近100个完成回合的回报选取。',
'',
'单球体日志停在613,416,960步，last为609,484,800步，best为488,243,200步，与上次一致。旧监测因memory_pressure发送SIGINT，随后destroy_sim清理崩溃；没有达到训练预算。本次核对检查点SHA256后复用原评估，未重复测量，也未重启训练。',
'',
'六物体独立回放比较如下。每检查点6×64=384次，相同seed215与抓取索引，最长30秒。成功定义为保持到30秒且沿目标轴累计净转角≥2π。',
'',
'| 检查点 | 训练步数M | 保持30秒 | 保持且净转≥1圈 | 平均净圈数 | 存活回合末5秒净轴向速度 rad/s |',
'|---|---:|---:|---:|---:|---:|',
]
for label,title in [('first','较早last'),('previous','上次last'),('best','当前best'),('last','当前last')]:
    r=comparison[label]
    lines.append(f"| {title} | {r['steps']/1e6:.2f} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['mean_turns']:.3f} | {r['last5s_speed_survivors']:.4f} |")
lines += ['', '最新last逐物体结果：', '',
'| 物体 | 上次平均圈数 | 当前平均圈数 | 保持30秒 | 保持且≥1圈 | ≥2有效胶面帧比例 | 末5秒净速度 rad/s |',
'|---|---:|---:|---:|---:|---:|---:|']
for obj,r in last['per_object'].items():
    lines.append(f"| {obj} | {previous['per_object'][obj]['mean_turns']:.3f} | {r['mean_turns']:.3f} | {r['retained_30s']:.1%} | {r['one_turn_retained']:.1%} | {r['frame_means']['two_gels']:.1%} | {r['last5s_speed_survivors']:.4f} |")
lines += ['',
'球体55mm为直径55mm；方块45/55mm为边长；胶囊70/80mm分别为总长×直径70×50、80×60mm。',
'',
'最新last所有物体的净角度主要在前15秒累积，末段几乎不再持续正转。这里的末5秒速度是有符号角增量的时间平均，接近零不排除局部往复运动，不能直接等同于所有关节完全静止。最新last与best均为0/384保持且满一圈；这不是对所有可能初始状态成功率为零的证明。',
'',
'训练窗口对比（各取最近1000万步；训练含动作采样，不能与均值动作回放直接等同）：',
'',
'| 指标 | 上次约6.02亿步 | 本次约9.23亿步 |',
'|---|---:|---:|']
for key,title,scale in [
('episode_rewards/step','平均回合回报',1),
('episode_lengths/step','平均回合时长（秒）',.05),
('rotation/speed','平均轴向速度（rad/s）',1),
('contact/task_gel_count','有效胶面数',1),
('control/target_increment_lost','目标增量裁剪损失（%）',100),
('reward/rotation','每步旋转奖励',1),
('reward/pose_penalty','每步姿态代价',1),
('reward/position_penalty','每步位置代价',1),
('reward/failure_penalty','每步失败代价',1),
('losses/entropy','高斯策略微分熵',1)]:
    lines.append(f'| {title} | {ow[key]*scale:.4f} | {w[key]*scale:.4f} |')
lines += ['',
'回报不是持续上升：约6.9–7.0亿步窗口均值为+0.75，随后回落，当前约−21.97。旋转奖励略增，但姿态、位置和失败代价增大，说明增加训练量并未稳定转化为任务能力。所有读取的TensorBoard标量有限；本次两次新回放均正常退出并完成清理，未观察到数值发散或评估清理崩溃。',
'',
'对原因的判断及证据边界：',
'',
'1. 主要行为仍是“初段调姿，随后保持”。从4.46亿、5.96亿到9.18亿步，独立回放始终没有保持且完成一圈，支持已经长期停留在不持续旋转的策略附近。不能仅凭此认定某一项奖励是唯一根因。',
'2. 胶面利用仍不足。最新各物体≥2有效胶面帧比例为约21%–40%；球体55mm约51%的有效帧没有目标物体胶面接触。非胶面与物体正载荷接触的采样帧比例约94%–97%，表明非胶面接触仍普遍存在，但这个指标不是承载力占比，也不能证明只靠手掌支撑。',
'3. 旧的严重目标饱和不是这次的主要证据。最新回放目标增量裁剪损失约4%–8%，训练窗口约5.6%。原始动作被[-1,1]裁剪的比例仍约33%–39%，与关节目标裁剪是不同指标，不能混为一谈；也不能由较低目标裁剪排除几何阻挡或动力学问题。',
'4. 探索逐渐收缩是值得做对照的线索。高斯策略微分熵从约−14.7降至−19.0；当前entropy_coef=0，没有显式熵奖励。微分熵为负本身正常，下降说明分布变窄；这只是伴随现象，不能证明增大熵奖励一定能学会换指。',
'',
'单球体对照（同文件SHA256，复用上次两seed共128次评估）：',
'',
'| 检查点 | 保持30秒 | 保持且≥1圈 | 平均净圈数 | 末5秒净速度 rad/s |',
'|---|---:|---:|---:|---:|',
'| best 488.24M | 94.5% | 71.9% | 2.068 | 0.376 |',
'| last 609.48M | 79.7% | 42.2% | 1.000 | 0.122 |',
'',
'建议优先保留单球488.24M作为有效基线。下一轮比继续单纯扩大并行环境或预算更有价值的实验，是从这套有效策略逐步加入一个尺寸/形状，并与六物体直接混训对照；以固定抓取集上的逐物体一圈成功率、末段持续正转和胶面使用率验收。探索强度或姿态代价分别作为后续单变量对照，避免一次改多项导致无法归因。本次没有执行这些改动，也没有停止当前训练。',
'',
'本次评估对象是有物体状态特权输入的教师，不是纯触觉学生或实机。圈数为目标轴上的角增量累计，不证明严格纯轴旋转、未知形状泛化或重建覆盖率。GPU物理仿真不保证位级确定性。',
'',
'证据文件：training_summary.json、multi_training_windows.json、comparison.json、eval_*.json、single_pooled.json、evaluation_processes.json、run_status_snapshot.json、固定pth快照及对应脚本。上次报告：../gel_rotation_results_20261018_0728_Chicago/report.md。'
]
(P/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(comparison,indent=2))
print('Verified evaluations and wrote',P/'report.md')
