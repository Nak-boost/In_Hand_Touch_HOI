"""Wait for idle allowed GPUs, then run grasp generation, validation and training.

Uses existing free resources, never evicts another workload or allocates dummy
GPU tensors. GPU leases coordinate instances of this scheduler on this host.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def gpu_snapshot():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = [row for row in csv.reader(output.splitlines()) if row]
    if any(len(row) != 5 for row in rows):
        raise ValueError('Incomplete nvidia-smi GPU statistics')
    return {int(row[0]): dict(uuid=row[1].strip(), total_mb=int(row[2]),
                             used_mb=int(row[3]), utilization=int(row[4])) for row in rows}


def idle(gpu, plan):
    return (gpu['total_mb'] - gpu['used_mb'] >= plan['minimum_free_mb']
            and gpu['used_mb'] <= plan['idle_max_used_mb']
            and gpu['utilization'] <= plan['idle_max_utilization'])


def acquire_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def check_sources(plan):
    for path, expected in plan['source_sha256'].items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Queued implementation/config changed: '+path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gpus', help='Override allowlist, e.g. 2,3 or all (explicit opt-in)')
    parser.add_argument('--once', action='store_true', help='Read-only resource/plan preflight; do not launch jobs')
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    if args.gpus:
        plan['gpus'] = sorted(gpu_snapshot()) if args.gpus == 'all' else list(dict.fromkeys(int(v) for v in args.gpus.split(',')))
    if not plan['gpus'] or plan['idle_samples'] < 1 or plan['poll_seconds'] <= 0:
        raise ValueError('Require nonempty GPU allowlist and positive sampling settings')
    check_sources(plan)
    if args.once:
        snapshot = gpu_snapshot()
        print(json.dumps({i: dict(snapshot.get(i, {}), eligible=i in snapshot and idle(snapshot[i], plan))
                          for i in plan['gpus']}, indent=2))
        return
    plan_lock = acquire_lock(plan_path.with_suffix('.lock'))
    if plan_lock is None:
        raise RuntimeError('A supervisor already owns this plan')
    status_path = plan_path.with_name('queue_status.json')
    if status_path.exists():
        raise FileExistsError('Existing queue status; use a new plan directory to avoid duplicate training: '+str(status_path))
    jobs = [dict(j, status='waiting', stage_index=0) for j in plan['experiments']]
    active = {}
    streak = {i: 0 for i in plan['gpus']}
    stopping = False
    latest = {}
    monitor_error = None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(updated=now(), supervisor_pid=os.getpid(),
            allowed_gpus=plan['gpus'], gpu_snapshot=latest, monitor_error=monitor_error,
            jobs=jobs), indent=2)+'\n')
        temp.replace(status_path)

    def launch(job, gpu, lease):
        check_sources(plan)
        stage = job['stages'][job['stage_index']]
        log_path = Path(stage['log'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('x')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        try:
            process = subprocess.Popen(stage['command'], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
        except Exception:
            log.close()
            raise
        job.update(status='running', gpu=gpu, pid=process.pid, phase=stage['name'],
                   phase_started=now(), console_log=str(log_path))
        print(now(), job['run_name'], stage['name'], 'GPU', gpu, 'PID', process.pid, flush=True)
        return dict(process=process, log=log, lease=lease, job=job)

    save()
    try:
        while not stopping and any(j['status'] in ('waiting', 'running') for j in jobs):
            for gpu, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                item['log'].close()
                job = item['job']
                job.setdefault('history', []).append(dict(phase=job['phase'], exit_code=code,
                                                         finished=now(), log=job['console_log']))
                del active[gpu]
                if code != 0:
                    job.update(status='failed', exit_code=code, finished=now())
                    item['lease'].close()
                elif job['stage_index'] + 1 == len(job['stages']):
                    job.update(status='completed', exit_code=0, finished=now())
                    item['lease'].close()
                else:
                    job['stage_index'] += 1
                    # Every phase must pass the resource gate again. Another
                    # workload may have arrived while the previous phase ran.
                    job.update(status='waiting', phase=job['stages'][job['stage_index']]['name'])
                    item['lease'].close()
                streak[gpu] = 0
                save()
            try:
                latest = gpu_snapshot()
                monitor_error = None
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                latest = {}
                monitor_error = str(error)  # Unknown resources are never treated as free.
            for gpu, item in active.items():
                if gpu in latest:
                    peaks = item['job'].setdefault('phase_peak_gpu_used_mb', {})
                    phase = item['job']['phase']
                    peaks[phase] = max(peaks.get(phase, 0), latest[gpu]['used_mb'])
            for gpu in plan['gpus']:
                if stopping or gpu in active:
                    continue
                available = gpu in latest and idle(latest[gpu], plan)
                streak[gpu] = streak[gpu] + 1 if available else 0
                pending = next((j for j in jobs if j['status'] == 'waiting'), None)
                if pending is None or streak[gpu] < plan['idle_samples']:
                    continue
                lease = acquire_lock(Path(plan['lock_directory'])/(latest[gpu]['uuid']+'.lock'))
                if lease is None:
                    continue
                try:
                    # Recheck after acquiring the lease, closing the race between
                    # cooperating queue instances. External jobs do not honor it.
                    fresh = gpu_snapshot()
                    if gpu not in fresh or not idle(fresh[gpu], plan):
                        lease.close()
                        streak[gpu] = 0
                        continue
                    active[gpu] = launch(pending, gpu, lease)
                except Exception as error:
                    pending.update(status='failed', reason=str(error), finished=now())
                    lease.close()
                streak[gpu] = 0
                save()
            save()
            if any(j['status'] in ('waiting', 'running') for j in jobs):
                time.sleep(plan['poll_seconds'])
    finally:
        # Stop only process groups launched by this supervisor, never GPU owners
        # discovered through nvidia-smi.
        for item in active.values():
            proc = item['process']
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for item in active.values():
            proc = item['process']
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            item['job'].update(status='interrupted', exit_code=proc.returncode, finished=now())
            item['log'].close()
            item['lease'].close()
        for job in jobs:
            if job['status'] == 'waiting':
                job.update(status='cancelled', reason='Supervisor stopped before launch')
        save()
        plan_lock.close()


if __name__ == '__main__':
    main()
