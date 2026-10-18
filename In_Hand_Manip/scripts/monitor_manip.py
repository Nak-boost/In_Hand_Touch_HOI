"""Supervise one teacher experiment; record failures and optionally resume safely."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hora.utils.training_watchdog import atomic_json, checkpoint_info, newest_checkpoint


def source_digest(root):
    paths = [root/'train_manip.py']
    for folder in ('hora', 'configs', 'scripts'):
        paths.extend(p for p in (root/folder).rglob('*') if p.suffix in ('.py', '.yaml', '.sh'))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def runtime_info():
    packages = {}
    for name in ('torch', 'numpy', 'hydra-core'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), packages=packages)


def gpu_sample(gpu, pid=None):
    rows = csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines())
    row = next(r for r in rows if int(r[0]) == gpu)
    result = dict(gpu=gpu, uuid=row[1].strip(), total_mib=int(row[2]),
                  used_mib=int(row[3]), utilization=int(row[4]), temperature_c=int(row[5]))
    if pid:
        rows = csv.reader(subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'],
            text=True, timeout=10).splitlines())
        memory = [int(r[2]) for r in rows if r[0].strip() == str(pid)
                  and r[1].strip() == result['uuid'] and r[2].strip().isdigit()]
        result['process_mib'] = sum(memory) if memory else None
        try:
            result['proc_status'] = '\n'.join(line for line in Path('/proc/%s/status' % pid).read_text().splitlines()
                if line.startswith(('State:', 'VmRSS:', 'VmSize:', 'Threads:')))
        except OSError:
            pass
    return result


def incident(out, reason, state, log):
    data = dict(time=time.time(), reason=reason, state=state,
                log_tail=Path(log).read_text(errors='replace')[-20000:])
    for name, command in [('nvidia_smi', ['nvidia-smi']),
                          ('kernel_tail', ['dmesg', '--ctime'])]:
        try:
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=10)
            data[name] = dict(returncode=p.returncode, output=p.stdout[-20000:])
        except Exception as error:
            data[name] = dict(error=str(error))
    atomic_json(out, data)


def stop_child(proc):
    """Only signal the session created for this training child."""
    for sig, seconds in [(signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)]:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            continue


def finished_checkpoint(run, heartbeat, target):
    if not heartbeat.get('training_complete'):
        return False
    try:
        return checkpoint_info(run/'stage1_nn/last.pth')['agent_steps'] >= target
    except Exception:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', type=int, required=True, help='Physical nvidia-smi GPU index')
    p.add_argument('--output', type=Path, required=True, help='New monitor directory')
    p.add_argument('--run-prefix', required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--max-steps', type=int, required=True, help='Total budget, including restored progress')
    p.add_argument('--save-frequency', type=int, default=50)
    p.add_argument('--max-restarts', type=int, default=0, help='Explicit opt-in; only native crashes/stalls')
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--stall-seconds', type=float, default=600)
    p.add_argument('--startup-seconds', type=float, default=1800)
    p.add_argument('--minimum-free-mib', type=int, default=2048)
    p.add_argument('--start-free-mib', type=int, default=12000)
    p.add_argument('--minimum-disk-gib', type=float, default=5)
    p.add_argument('overrides', nargs='*', help='Additional Hydra settings, e.g. num_envs=16384')
    a = p.parse_args(argv)
    if min(a.max_steps, a.save_frequency, a.poll_seconds, a.stall_seconds, a.startup_seconds) <= 0 or a.max_restarts < 0:
        p.error('Budgets/timeouts must be positive and restarts nonnegative')
    if Path(a.run_prefix).name != a.run_prefix or a.run_prefix in ('.', '..'):
        p.error('run-prefix must be a single directory name')
    reserved = {'stage', 'run_root', 'run_name', 'checkpoint', 'train.ppo.max_agent_steps', 'train.ppo.save_frequency'}
    if any(x.lstrip('+').split('=')[0] in reserved or '=' not in x or x.startswith('-') for x in a.overrides):
        p.error('Use dedicated options for stage/run/checkpoint/budget/save settings')
    checkpoint = a.checkpoint.resolve() if a.checkpoint else None
    if checkpoint:
        checkpoint_info(checkpoint)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runroot = ROOT.parent/'runs/manip'
    fingerprint = source_digest(ROOT)
    state = dict(status='starting', supervisor_pid=os.getpid(), started=time.time(),
                 source_sha256=fingerprint, runtime=runtime_info(), attempts=[],
                 options={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
    atomic_json(out/'status.json', state)
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    initial_checkpoint = checkpoint
    last_restart_step = -1
    proc = None
    try:
        for attempt in range(a.max_restarts+1):
            if stopped:
                state['status'] = 'stopped_by_user'; break
            if source_digest(ROOT) != fingerprint:
                state['status'] = 'stopped_source_changed'; break
            sample = gpu_sample(a.gpu)
            if sample['total_mib']-sample['used_mib'] < a.start_free_mib:
                state.update(status='blocked_insufficient_gpu_memory', resource=sample); break
            if shutil.disk_usage(out).free < a.minimum_disk_gib*1024**3:
                state['status'] = 'blocked_insufficient_disk'; break
            run = runroot/(a.run_prefix+'_attempt%02d' % attempt)
            if run.exists():
                raise FileExistsError(run)
            folder = out/('attempt%02d' % attempt); folder.mkdir()
            heartbeat = folder/'heartbeat.json'; logpath = folder/'console.log'
            command = [str(ROOT/'scripts/run_manip.sh'), '--config-name', a.config,
                       'stage=teacher', 'run_root='+str(runroot), 'run_name='+run.name,
                       'train.ppo.max_agent_steps='+str(a.max_steps),
                       'train.ppo.save_frequency='+str(a.save_frequency)] + a.overrides
            if checkpoint:
                command.append('checkpoint='+str(checkpoint))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONFAULTHANDLER='1',
                       PYTHONUNBUFFERED='1', TORCH_SHOW_CPP_STACKTRACES='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MANIP_HEARTBEAT=str(heartbeat))
            with logpath.open('x') as log, (folder/'resources.jsonl').open('x') as resources:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                record = dict(attempt=attempt, pid=proc.pid, run_dir=str(run), command=command,
                              log=str(logpath), started=time.time())
                state['attempts'].append(record); state['status'] = 'running'
                previous_steps = None; progressed = time.monotonic(); reason = None
                while proc.poll() is None:
                    h = read_json(heartbeat)
                    steps = h.get('agent_steps', 0)
                    if steps != previous_steps:
                        previous_steps = steps; progressed = time.monotonic()
                    try:
                        resource = dict(time=time.time(), **gpu_sample(a.gpu, proc.pid))
                        resources.write(json.dumps(resource)+'\n'); resources.flush()
                        record['last_resource'] = resource
                        if resource['total_mib']-resource['used_mib'] < a.minimum_free_mib:
                            reason = 'memory_pressure'
                    except Exception as error:
                        record['monitor_error'] = str(error)
                    if shutil.disk_usage(out).free < a.minimum_disk_gib*1024**3:
                        reason = 'disk_pressure'
                    limit = a.startup_seconds if h.get('phase', 'initializing') in ('initializing','environment_ready') else a.stall_seconds
                    if time.monotonic()-progressed > limit:
                        reason = 'progress_stalled'
                    record.update(heartbeat=h, last_checked=time.time())
                    atomic_json(out/'status.json', state)
                    if stopped or reason:
                        reason = 'user_stop' if stopped else reason
                        # Registered only after main() starts: never send default-fatal USR1 at import time.
                        if h.get('pid') == proc.pid and proc.poll() is None:
                            try: os.kill(proc.pid, signal.SIGUSR1)
                            except ProcessLookupError: pass
                            time.sleep(min(1, a.poll_seconds))
                        stop_child(proc); break
                    time.sleep(a.poll_seconds)
                proc.wait()
                record.update(exit_code=proc.returncode, reason=reason, finished=time.time())
                h = read_json(heartbeat); record['heartbeat'] = h
            complete = finished_checkpoint(run, h, a.max_steps)
            if complete:
                state['status'] = 'completed' if proc.returncode == 0 else 'completed_with_cleanup_failure'
            elif reason == 'user_stop' or proc.returncode in (-signal.SIGINT, -signal.SIGTERM):
                state['status'] = 'stopped'
            else:
                state['status'] = 'failed'
            if proc.returncode != 0 or reason:
                incident(folder/'incident.json', reason or 'process_exit', record, logpath)
            atomic_json(out/'status.json', state)
            if complete or stopped or reason in ('memory_pressure', 'disk_pressure', 'user_stop'):
                break
            retryable = proc.returncode in (-signal.SIGSEGV, -signal.SIGABRT) or reason == 'progress_stalled'
            if not retryable or attempt >= a.max_restarts:
                break
            candidates = [p for r in state['attempts'] for p in (Path(r['run_dir'])/'stage1_nn').glob('*.pth')]
            if initial_checkpoint: candidates.append(initial_checkpoint)
            best, rejected = newest_checkpoint(candidates)
            atomic_json(folder/'recovery_selection.json', dict(selected=best, rejected=rejected))
            if best is None or best['agent_steps'] <= last_restart_step:
                state['status'] = 'stopped_no_recoverable_progress'; break
            checkpoint = Path(best['path']); last_restart_step = best['agent_steps']
            state['status'] = 'restarting'
            atomic_json(out/'status.json', state)
    except BaseException as error:
        state.update(status='supervisor_error', error=repr(error))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            stop_child(proc)
        state['finished'] = time.time()
        atomic_json(out/'status.json', state)
    print(json.dumps(dict(status=state['status'], output=str(out))))
    return 0 if state['status'] == 'completed' else 2


if __name__ == '__main__':
    sys.exit(main())
