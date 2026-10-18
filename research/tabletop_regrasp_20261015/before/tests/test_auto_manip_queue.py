"""Resource admission and failure gates without consuming a real GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/auto_manip_queue.py'
spec = importlib.util.spec_from_file_location('auto_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_zero_utilization_does_not_make_allocated_memory_idle():
    policy = dict(minimum_free_mb=12000, idle_max_used_mb=1024, idle_max_utilization=10)
    assert not queue.idle(dict(total_mb=24564, used_mb=23783, utilization=0), policy)
    assert not queue.idle(dict(total_mb=24564, used_mb=0, utilization=95), policy)
    assert queue.idle(dict(total_mb=24564, used_mb=0, utilization=0), policy)


def test_cooperative_lease_excludes_second_supervisor(tmp_path):
    first = queue.acquire_lock(tmp_path/'gpu.lock')
    assert first is not None
    assert queue.acquire_lock(tmp_path/'gpu.lock') is None
    first.close()
    second = queue.acquire_lock(tmp_path/'gpu.lock')
    assert second is not None
    second.close()


def fixture(tmp_path, fail_validation=False):
    gpu = tmp_path/'gpu.csv'
    gpu.write_text('0, GPU-other, 24564, 0, 0\n2, GPU-test, 24564, 24000, 0\n')
    binary = tmp_path/'nvidia-smi'
    binary.write_text('#!'+sys.executable+'\nfrom pathlib import Path\nprint(Path('+repr(str(gpu))+').read_text())\n')
    binary.chmod(0o755)
    marker = tmp_path/'executed'
    stages = []
    for phase in ['generate', 'validate', 'train']:
        code = "from pathlib import Path; import os; p=Path("+repr(str(marker))+"); p.open('a').write("+repr(phase)+"+':'+os.environ['CUDA_VISIBLE_DEVICES']+'\\n')"
        if phase == 'validate' and fail_validation:
            code += '; raise SystemExit(7)'
        stages.append(dict(name=phase, command=[sys.executable, '-c', code], log=str(tmp_path/(phase+'.log'))))
    plan = dict(gpus=[2], source_sha256={}, minimum_free_mb=12000, idle_max_used_mb=1024,
        idle_max_utilization=10, idle_samples=2, poll_seconds=.05, lock_directory=str(tmp_path/'locks'),
        experiments=[dict(run_name='test', stages=stages)])
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'])
    return path, env, gpu, marker


def wait_until(predicate, seconds=6):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('Timed out waiting for queue state')


def test_waits_for_allowed_gpu_then_runs_stages_in_order(tmp_path):
    path, env, gpu, marker = fixture(tmp_path)
    proc = subprocess.Popen([sys.executable, str(SCRIPT), '--plan', str(path)], env=env, stdout=subprocess.DEVNULL)
    try:
        status = tmp_path/'queue_status.json'
        wait_until(status.exists)
        duplicate = subprocess.run([sys.executable, str(SCRIPT), '--plan', str(path)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        assert duplicate.returncode != 0
        time.sleep(.2)
        assert not marker.exists()  # GPU 0 is idle but not allowed; GPU 2 is occupied.
        gpu.write_text('0, GPU-other, 24564, 0, 0\n2, GPU-test, 24564, 0, 0\n')
        assert proc.wait(timeout=8) == 0
        assert marker.read_text().splitlines() == ['generate:2', 'validate:2', 'train:2']
        assert json.loads(status.read_text())['jobs'][0]['status'] == 'completed'
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=5)


def test_validation_failure_never_launches_training(tmp_path):
    path, env, gpu, marker = fixture(tmp_path, fail_validation=True)
    gpu.write_text('2, GPU-test, 24564, 0, 0\n')
    result = subprocess.run([sys.executable, str(SCRIPT), '--plan', str(path)], env=env,
                            stdout=subprocess.DEVNULL, timeout=8)
    assert result.returncode == 0
    assert marker.read_text().splitlines() == ['generate:2', 'validate:2']
    job = json.loads((tmp_path/'queue_status.json').read_text())['jobs'][0]
    assert job['status'] == 'failed' and job['exit_code'] == 7


def test_readonly_preflight_and_changed_source_gate(tmp_path):
    path, env, gpu, marker = fixture(tmp_path)
    subprocess.run([sys.executable, str(SCRIPT), '--plan', str(path), '--once'], env=env,
                   check=True, stdout=subprocess.DEVNULL)
    assert not marker.exists() and not (tmp_path/'queue_status.json').exists()
    plan = json.loads(path.read_text())
    plan['source_sha256'] = {str(gpu): 'intentionally_invalid'}
    path.write_text(json.dumps(plan))
    result = subprocess.run([sys.executable, str(SCRIPT), '--plan', str(path)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode != 0 and not marker.exists()
