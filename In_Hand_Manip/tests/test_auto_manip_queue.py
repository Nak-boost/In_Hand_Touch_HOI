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


def test_memory_budget_reserve_and_compute_gate_are_both_required():
    policy = dict(minimum_free_mb=4000, job_memory_mb=5000, reserve_memory_mb=2000, idle_max_utilization=15)
    assert queue.idle(dict(total_mb=24000, used_mb=16000, utilization=10), policy)
    assert not queue.idle(dict(total_mb=24000, used_mb=18000, utilization=0), policy)
    assert not queue.idle(dict(total_mb=24000, used_mb=16000, utilization=50), policy)


def test_quality_gate_rejects_failed_or_old_controller_report(tmp_path):
    import pytest
    report = tmp_path/'quality.json'
    plan = dict(source_sha256={}, quality_gates=[dict(report=str(report), minimum_accepted=16,
                action_reference='actual_position')])
    for value in [dict(passed=False,accepted=64,action_reference='actual_position'),
                  dict(passed=True,accepted=15,action_reference='actual_position'),
                  dict(passed=True,accepted=64,action_reference='previous_target')]:
        report.write_text(json.dumps(value))
        with pytest.raises(ValueError): queue.check_sources(plan)
    report.write_text(json.dumps(dict(passed=True,accepted=16,action_reference='actual_position')))
    queue.check_sources(plan)


def test_training_waits_for_measurement_and_uses_peak_margin(tmp_path):
    import pytest
    path = tmp_path/'resources.json'
    job = dict(stage_index=0, stages=[dict(resource_report=str(path), num_envs=256)])
    policy = dict(job_memory_mb=4096, memory_peak_multiplier=1.25)
    assert queue.stage_resources(job, policy) is None
    path.write_text(json.dumps(dict(exit_code=0, process_peak_mb=4000, num_envs=256)))
    assert queue.stage_resources(job, policy)['job_memory_mb'] == 5000
    path.write_text(json.dumps(dict(exit_code=1, process_peak_mb=4000, num_envs=256)))
    with pytest.raises(ValueError): queue.stage_resources(job, policy)
    path.write_text(json.dumps(dict(exit_code=0, process_peak_mb=4000, num_envs=512)))
    with pytest.raises(ValueError): queue.stage_resources(job, policy)


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


def test_shared_profile_completes_before_both_training_jobs(tmp_path):
    path, env, gpu, marker = fixture(tmp_path)
    gpu.write_text('2, GPU-test, 24564, 0, 0\n')
    plan = json.loads(path.read_text())
    resource = tmp_path/'resources.json'
    code = "import json,sys; from pathlib import Path; assert sys.argv[1]=='2'; Path("+repr(str(resource))+ ").write_text(json.dumps(dict(exit_code=0,process_peak_mb=2000,num_envs=256)))"
    first = plan['experiments'][0]
    train = first['stages'][-1]
    train.update(resource_report=str(resource), num_envs=256, calibration_job='test')
    first['stages'] = [dict(name='profile',command=[sys.executable,'-c',code,'{gpu}'], log=str(tmp_path/'profile.log')),train]
    second_train = dict(train,log=str(tmp_path/'train2.log'))
    plan['experiments'].append(dict(run_name='test2', stages=[second_train]))
    path.write_text(json.dumps(plan))
    result = subprocess.run([sys.executable,str(SCRIPT),'--plan',str(path)],env=env,
                            stdout=subprocess.DEVNULL,timeout=8)
    assert result.returncode == 0
    assert marker.read_text().splitlines() == ['train:2','train:2']
    assert all(j['status']=='completed' for j in json.loads((tmp_path/'queue_status.json').read_text())['jobs'])


def test_failed_profile_marks_dependent_jobs_failed(tmp_path):
    path, env, gpu, marker = fixture(tmp_path)
    gpu.write_text('2, GPU-test, 24564, 0, 0\n')
    plan = json.loads(path.read_text())
    profile = dict(name='profile', command=[sys.executable,'-c','raise SystemExit(7)'],log=str(tmp_path/'profile.log'))
    train = plan['experiments'][0]['stages'][-1]
    train.update(resource_report=str(tmp_path/'missing.json'),num_envs=256,calibration_job='test')
    plan['experiments']=[dict(run_name='test',stages=[profile,train]),
                         dict(run_name='test2',stages=[dict(train,log=str(tmp_path/'train2.log'))])]
    path.write_text(json.dumps(plan))
    subprocess.run([sys.executable,str(SCRIPT),'--plan',str(path)],env=env,check=True,
                   stdout=subprocess.DEVNULL,timeout=8)
    assert not marker.exists()
    assert all(j['status']=='failed' for j in json.loads((tmp_path/'queue_status.json').read_text())['jobs'])
