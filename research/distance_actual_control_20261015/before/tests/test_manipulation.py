"""Behavioral checks that do not require Isaac Gym or physical hardware."""
import importlib.util
from pathlib import Path
import math
import torch

ROOT = Path(__file__).resolve().parents[1]
# Avoid hora.tasks.__init__ importing Isaac Gym during tensor-only tests.
spec = importlib.util.spec_from_file_location('manipulation_math', ROOT/'hora/tasks/manipulation_math.py')
maths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maths)


def test_quaternion_sign_and_axis_reward():
    q0 = torch.tensor([[0., 0., 0., 1.]])
    q1 = torch.tensor([[0., 0., math.sin(.1), math.cos(.1)]])
    expected = torch.tensor([[0., 0., .2]])
    assert torch.allclose(maths.rotation_increment(q1, q0), expected, atol=1e-6)
    assert torch.allclose(maths.rotation_increment(-q1, q0), expected, atol=1e-6)
    forward = maths.rotation_terms(expected, torch.tensor([[0., 0., 1.]]), .5)
    wrong = maths.rotation_terms(expected, torch.tensor([[1., 0., 0.]]), .5)
    assert forward[0] > wrong[0]
    assert forward[1] < wrong[1]


def test_pose_penalty_restores_each_environments_own_grasp():
    initial = torch.tensor([[0.2, -0.4], [1.1, 0.7]])
    q = (initial + torch.tensor([[0.1, -0.2], [0., 0.]])).requires_grad_()
    loss = maths.joint_pose_deviation(q, initial)
    torch.testing.assert_close(loss, torch.tensor([0.05, 0.]))
    loss.sum().backward()
    torch.testing.assert_close(q.grad, torch.tensor([[0.2, -0.4], [0., 0.]]))
    torch.testing.assert_close(maths.joint_pose_deviation(q.detach()+2, initial+2), loss)


def test_reset_history_does_not_leak_previous_episode():
    history = torch.randn(2, 30, maths.OBS_DIM)
    frame = torch.randn(2, maths.OBS_DIM)
    updated = maths.append_history(history, frame, torch.tensor([True, False]))
    assert torch.equal(updated[0], frame[0].expand_as(updated[0]))
    assert torch.equal(updated[1, :-1], history[1, 1:])


def test_student_no_privileged_leak_and_joint_gradients():
    from hora.algo.models.models import ActorCritic
    from hora.algo.models.tactile_student import TactileStudent, distillation_loss
    cfg = dict(actor_units=[64, 32], priv_mlp_units=[32, 8], actions_num=16,
               input_shape=(maths.OBS_DIM,), priv_info=True, proprio_adapt=False, priv_info_dim=23)
    teacher = ActorCritic(dict(cfg)).eval().requires_grad_(False)
    student = TactileStudent(cfg, maths.OBS_DIM)
    student.load_state_dict(teacher.state_dict(), strict=False)
    x = {'obs': torch.randn(4, maths.OBS_DIM), 'proprio_hist': torch.randn(4, 30, maths.OBS_DIM),
         'priv_info': torch.randn(4, 23)}
    first = student.act_inference(x)
    x['priv_info'] = torch.full((4, 23), float('nan'))
    assert torch.equal(first, student.act_inference(x))
    del x['priv_info']
    assert torch.equal(first, student.act_inference(x))
    labels = teacher._actor_critic({'obs': x['obs'], 'priv_info': torch.randn(4, 23)})
    before = student.mu.weight.detach().clone()
    opt = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=.001)
    loss, _, _ = distillation_loss(student._actor_critic(x), labels)
    loss.backward()
    for module in [student.adapt_tconv, student.actor_mlp, student.mu]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.env_mlp.parameters())
    opt.step()
    assert not torch.equal(before, student.mu.weight)


def test_asset_has_separate_gel_links_and_valid_meshes():
    import xml.etree.ElementTree as ET
    asset = ROOT/'assets/allegro/allegro_digit.urdf'
    tree = ET.parse(asset)
    joints = tree.findall('joint')
    assert sum(j.get('type') == 'revolute' for j in joints) == 16
    for finger in ['index', 'middle', 'ring', 'thumb']:
        assert tree.find("link[@name='digit_%s_gel']/collision" % finger) is not None
        assert tree.find("link[@name='digit_%s_housing']/collision" % finger) is not None
    for mesh in tree.iter('mesh'):
        assert (asset.parent/mesh.get('filename')).is_file()


def test_runtime_matches_training_observation_and_action_filter(tmp_path):
    from hora.algo.models.tactile_student import TactileStudent
    from hora.algo.models.running_mean_std import RunningMeanStd
    from hora.algo.padapt.tactile_distill import normalize_inputs
    from hora.algo.deploy.digit_runtime import DigitPolicyRuntime
    cfg = dict(actor_units=[64, 32], priv_mlp_units=[32, 8], actions_num=16,
               input_shape=(maths.OBS_DIM,), priv_info=True, proprio_adapt=False, priv_info_dim=23)
    model = TactileStudent(cfg, maths.OBS_DIM).eval()
    rms = RunningMeanStd((maths.OBS_DIM,)).eval()
    lower, upper = torch.full((16,), -.5), torch.full((16,), 1.5)
    path = tmp_path/'student.pt'
    torch.save(dict(kind='tactile_student', network=cfg, history_dim=maths.OBS_DIM, history_length=30,
                    model=model.state_dict(), running_mean_std=rms.state_dict(),
                    joint_names=list(range(16)), lower=lower, upper=upper,
                    action_scale=.026, action_smoothing=.5), path)
    runtime = DigitPolicyRuntime(path)
    q, qdot = torch.full((1, 16), .3), torch.zeros(1, 16)
    contacts = torch.tensor([[1., 1., 0., 1.]])
    previous, targets = torch.zeros_like(q), q.clone()
    history = torch.zeros(1, 30, maths.OBS_DIM)
    for step, axis in enumerate([torch.tensor([[0., 0., 1.]]), torch.tensor([[1., 0., 0.]])]):
        frame = maths.build_observation(q, qdot, targets, previous, contacts, axis, lower, upper)
        history = maths.append_history(history, frame, torch.tensor([step == 0]))
        with torch.no_grad():
            action = model.act_inference(normalize_inputs(rms, frame, history)).clamp(-1, 1)
        previous = .5*action+.5*previous
        targets = (targets+.026*previous).clamp(lower, upper)
        assert torch.allclose(runtime.step(q, qdot, contacts, axis), targets[0], atol=1e-6)
