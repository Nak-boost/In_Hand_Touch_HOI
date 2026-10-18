"""Fault-injection checks for checkpoint durability and bounded recovery."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from hora.utils.training_watchdog import atomic_torch_save, checkpoint_info, newest_checkpoint


def checkpoint(steps):
    return dict(model={'weight': torch.ones(1)}, optimizer={'state': {}},
                running_mean_std={}, value_mean_std={}, agent_steps=steps, epoch_num=steps,
                last_lr=.001, asset_sha256='asset', collision_sha256='collision',
                action_reference='actual_position', kind='privileged_teacher')


def test_recovery_uses_steps_and_rejects_corruption(tmp_path):
    torch.save(checkpoint(20), tmp_path/'best.pth')
    torch.save(checkpoint(10), tmp_path/'last.pth')
    (tmp_path/'partial.pth').write_bytes(b'incomplete')
    bad = checkpoint(30); bad['optimizer']['nan'] = torch.tensor(float('nan'))
    torch.save(bad, tmp_path/'nonfinite.pth')
    selected, rejected = newest_checkpoint(tmp_path.glob('*.pth'))
    assert selected['agent_steps'] == 20 and selected['path'].endswith('best.pth')
    assert len(rejected) == 2


def test_interrupted_save_preserves_old_checkpoint(tmp_path, monkeypatch):
    path = tmp_path/'last.pth'; atomic_torch_save(checkpoint(10), path)
    def fail(data, stream):
        stream.write(b'partial archive')
        raise OSError('injected write failure')
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'save', fail)
        with pytest.raises(OSError): atomic_torch_save(checkpoint(20), path)
    assert checkpoint_info(path)['agent_steps'] == 10
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('mode,expected,attempts', [
    ('recover', 'completed', 2), ('repeat', 'stopped_no_recoverable_progress', 2),
    ('cleanup', 'completed_with_cleanup_failure', 1), ('stall', 'completed', 2),
    ('ordinary_error', 'failed', 1), ('source_change', 'stopped_source_changed', 1)])
def test_supervisor_fault_injection(tmp_path, monkeypatch, mode, expected, attempts):
    spec = importlib.util.spec_from_file_location('watchdog_cli',
        Path(__file__).resolve().parents[1]/'scripts/monitor_manip.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    root = tmp_path/'project'; (root/'scripts').mkdir(parents=True)
    (root/'train_manip.py').write_text('# fake training entry\n')
    payload = '''import os,sys,time,json,signal,resource
from pathlib import Path
import torch
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
signal.signal(signal.SIGUSR1,lambda *args:None)
opts=dict(x.split('=',1) for x in sys.argv if '=' in x)
run=Path(opts['run_root'])/opts['run_name'];(run/'stage1_nn').mkdir(parents=True)
first=run.name.endswith('00');mode=MODE
if mode=='ordinary_error':sys.exit(1)
steps=10 if mode=='repeat' or (first and mode in ('recover','stall','source_change')) else 20
ck=dict(model={'w':torch.ones(1)},optimizer={},running_mean_std={},value_mean_std={},
 agent_steps=steps,epoch_num=steps,last_lr=.001,asset_sha256='a',collision_sha256='c',
 action_reference='actual_position',kind='privileged_teacher')
torch.save(ck,run/'stage1_nn/last.pth')
Path(os.environ['MANIP_HEARTBEAT']).write_text(json.dumps(dict(pid=os.getpid(),agent_steps=steps,
 phase='destroy_sim' if steps==20 else 'collect',training_complete=steps==20)))
if mode=='stall' and first:time.sleep(30)
if mode=='source_change':Path('train_manip.py').write_text('# changed while training')
if mode in ('cleanup','repeat','source_change') or (mode=='recover' and first):os.kill(os.getpid(),signal.SIGSEGV)
if not first:assert 'checkpoint' in opts
# A 0.3s synthetic stall threshold must not measure PyTorch interpreter teardown.
os._exit(0)
'''.replace('MODE', repr(mode))
    launcher = root/'scripts/run_manip.sh'
    launcher.write_text('#!'+sys.executable+'\n'+payload); launcher.chmod(0o755)
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m, 'gpu_sample', lambda *args:dict(total_mib=24000,used_mib=0))
    monkeypatch.setattr(m, 'incident', lambda path,*args:m.atomic_json(path,{'captured':True}))
    out=tmp_path/'monitor'
    m.main(['--config','fake','--gpu','1','--output',str(out),'--run-prefix','test',
            '--max-steps','20','--max-restarts','3','--poll-seconds','.05',
            '--stall-seconds','.3','--startup-seconds','30','--minimum-disk-gib','0'])
    state=json.loads((out/'status.json').read_text())
    assert state['status']==expected
    assert len(state['attempts'])==attempts
    if attempts==2:
        assert any(x.startswith('checkpoint=') for x in state['attempts'][1]['command'])
    assert (out/'attempt00/incident.json').exists()


def test_success_marker_without_valid_checkpoint_is_not_completion(tmp_path):
    spec=importlib.util.spec_from_file_location('monitor_cli',
        Path(__file__).resolve().parents[1]/'scripts/monitor_manip.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert not m.finished_checkpoint(tmp_path, {'training_complete':True}, 100)
