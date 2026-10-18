"""Unattended stage sequencing and failure gates without a GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/supervise_tabletop_stages.py'


def module():
    spec=importlib.util.spec_from_file_location('tabletop_stages',SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_failed_quality_never_passes_preflight(tmp_path):
    m=module();q=tmp_path/'quality.json';r=tmp_path/'resources.json'
    q.write_text(json.dumps(dict(passed=False,accepted=100,action_reference='actual_position')))
    r.write_text(json.dumps(dict(exit_code=0,num_envs=2560)))
    with pytest.raises(ValueError,match='grasp gate'):
        m.verify(dict(sha256={},experiments=[dict(quality_report=str(q),resource_report=str(r),num_envs=2560)]))


def test_teardown_exception_requires_complete_finite_checkpoint(tmp_path):
    import torch
    m=module(); folder=tmp_path/'stage1_nn';folder.mkdir()
    log=tmp_path/'console.log'
    log.write_text('max steps achieved\nFatal Python error: Segmentation fault\n'
                   'Current thread:\n  File "hora/tasks/allegro_digit_two_axis.py", line 513 in close\n')
    ck=dict(model={'weight':torch.ones(1)},optimizer={},agent_steps=100)
    torch.save(ck,folder/'last.pth')
    assert m.completed_before_teardown(tmp_path,log,100)
    assert not m.completed_before_teardown(tmp_path,log,101)
    ck['model']['weight'][0]=float('nan');torch.save(ck,folder/'last.pth')
    assert not m.completed_before_teardown(tmp_path,log,100)
    log.write_text('Fatal Python error: Segmentation fault\nCurrent thread: in step')
    assert not m.completed_before_teardown(tmp_path,log,100)


@pytest.mark.parametrize('fail_training',[False,True])
def test_training_then_two_evaluations_then_resume_and_failure_stops(tmp_path,monkeypatch,fail_training):
    m=module();root=tmp_path/'project';scripts=root/'scripts';scripts.mkdir(parents=True)
    marker=tmp_path/'calls.jsonl'
    code='''import sys,json
from pathlib import Path
args=sys.argv[1:]
with Path(MARKER).open('a') as f:f.write(json.dumps(args)+'\\n')
if 'stage=teacher' in args:
 if FAIL:sys.exit(7)
 opts=dict(x.split('=',1) for x in args if '=' in x)
 if 'checkpoint' in opts:assert Path(opts['checkpoint']).exists()
 p=Path(opts['run_root'])/opts['run_name']/'stage1_nn';p.mkdir(parents=True)
 (p/'last.pth').write_text('checkpoint')
 (p/'best.pth').write_text('checkpoint')
else:
 output=Path(args[args.index('--output')+1])
 output.write_text(json.dumps(dict(object='sphere55',agent_steps=10,checkpoint='ck',qualified_one_turn_fraction=0,one_turn_retained_fraction=0,late_signed_speed_all_starts=0,frame_means={})))
'''.replace('MARKER',repr(str(marker))).replace('FAIL',repr(fail_training))
    for name in ['run_manip.sh','python_manip.sh']:
        f=scripts/name;f.write_text('#!'+sys.executable+'\n'+code);f.chmod(0o755)
    q=tmp_path/'quality.json';q.write_text(json.dumps(dict(passed=True,accepted=128,action_reference='actual_position')))
    r=tmp_path/'resources.json';r.write_text(json.dumps(dict(exit_code=0,num_envs=2560,process_peak_mb=3000)))
    plan=dict(sha256={},milestones=[10000000,30000000],run_root=str(tmp_path/'runs'),experiments=[dict(
        object='sphere55',gpu=2,config='frozen',run_prefix='test',num_envs=2560,quality_report=str(q),resource_report=str(r))])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m,'gpu_stats',lambda:{2:dict(uuid='GPU-test',total_mb=24000,used_mb=0,utilization=0)})
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'')
    sleep=time.sleep;monkeypatch.setattr(m.time,'sleep',lambda _:sleep(.01))
    monkeypatch.setattr(sys,'argv',['supervisor','--plan',str(path)])
    m.main()
    job=json.loads((tmp_path/'queue_status.json').read_text())['jobs'][0]
    calls=[json.loads(x) for x in marker.read_text().splitlines()]
    if fail_training:
        assert job['status']=='failed' and len(calls)==1
    else:
        assert job['status']=='completed' and len(calls)==6
        assert [x['phase'] for x in job['history']]==['train','eval_last','eval_best']*2
        assert any(x.startswith('checkpoint=') for x in calls[3])
