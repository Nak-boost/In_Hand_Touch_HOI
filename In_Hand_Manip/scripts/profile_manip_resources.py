"""Measure per-process GPU memory during a short, separately named PPO run."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from auto_manip_queue import gpu_snapshot, now

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--steps', type=int, default=16384)
    p.add_argument('--config', default='manip_tabletop_box45')
    p.add_argument('--minibatch-size', type=int, help='Keep the production PPO minibatch during calibration')
    a = p.parse_args()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline = gpu_snapshot()[a.gpu]
    if baseline['total_mb']-baseline['used_mb'] < 6000:
        raise RuntimeError('Need at least 6000 MiB free for the resource probe')
    cmd = [str(ROOT/'scripts/run_manip.sh'), '--config-name', a.config,
           'stage=teacher', 'num_envs='+str(a.num_envs), 'graphics_device_id=-1',
           'task.env.numEnvs='+str(a.num_envs), 'train.ppo.num_actors='+str(a.num_envs),
           'run_root='+str(out/'runs'), 'run_name=resource_probe',
           'train.ppo.max_agent_steps='+str(a.steps),
           'train.ppo.minibatch_size='+str(a.minibatch_size or a.num_envs*16), 'seed=241']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    samples = []
    started = now()
    with (out/'console.log').open('x') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                stats = gpu_snapshot()[a.gpu]
                text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                                 '--format=csv,noheader,nounits'], text=True, timeout=10)
                memory = sum(int(row[2]) for row in csv.reader(text.splitlines()) if
                             len(row) == 3 and row[0].strip() == str(process.pid) and row[1].strip() == baseline['uuid'])
                samples.append(dict(time=now(), process_memory_mb=memory, **stats))
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    report = dict(started=started, finished=now(), command=cmd, gpu=a.gpu, pid=process.pid,
                  exit_code=process.returncode, baseline=baseline, num_envs=a.num_envs,
                  config=a.config, minibatch_size=a.minibatch_size or a.num_envs*16,
                  process_peak_mb=max((s['process_memory_mb'] for s in samples), default=0),
                  note='Utilization and used_mb are whole-GPU measurements and include other workloads.', samples=samples)
    temporary = out/'resources.tmp'
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(out/'resources.json')
    print(json.dumps({k: report[k] for k in ['exit_code', 'num_envs', 'process_peak_mb']}), flush=True)
    if process.returncode or report['process_peak_mb'] == 0:
        raise RuntimeError('Resource probe failed or per-process memory was unavailable')


if __name__ == '__main__':
    main()
