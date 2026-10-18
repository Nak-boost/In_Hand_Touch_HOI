"""Tabletop collision and termination contracts, without a simulator."""
import importlib.util
import itertools
from pathlib import Path
import torch
from hora.utils.hand_collision import collision_spec, collision_signature

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tabletop_math', ROOT/'hora/tasks/tabletop_math.py')
maths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maths)


def test_shared_bit_changes_only_authorized_pairs():
    base = collision_spec(ROOT/'assets/allegro/allegro_digit_compact.urdf')
    result = maths.tabletop_collision_spec(base, {'friction': .1})
    masks = result['body_masks']
    gels = {n for n in masks if n.endswith('_gel')}
    for a, b in itertools.combinations(masks, 2):
        actual = bool(masks[a] & masks[b])
        expected = bool(base['body_masks'][a] & base['body_masks'][b]) or {a,b} <= gels
        assert actual == expected
    for body, mask in masks.items():
        assert bool(mask & result['table_filter']) == (body in gels)
        assert (mask & 0) == 0  # object mask is zero
    assert result['table_filter'] < 2**31
    changed = maths.tabletop_collision_spec(base, {'friction': .2})
    assert collision_signature(changed) != collision_signature(result)
    assert all(not (mask & result['table_filter']) for mask in base['body_masks'].values())


def test_contact_loss_duration_and_independent_reset():
    position = torch.zeros(2,3)
    center = torch.zeros(2)
    missing = torch.zeros(2,dtype=torch.long)
    contact = torch.zeros(2,4,dtype=torch.bool)
    for step in range(40):
        contact[1,2] = step == 20
        failed, missing, outside, lost = maths.tabletop_failures(position,contact,missing,center,.1,40)
        assert bool(failed[0]) == (step == 39)
        assert not bool(failed[1])
        assert not outside.any()
    contact[0,0] = True
    failed, missing, _, _ = maths.tabletop_failures(position,contact,missing,center,.1,40)
    assert not failed.any() and missing[0] == 0


def test_workspace_is_xy_circle_not_orientation_or_height():
    p = torch.tensor([[.099,0.,50.],[.101,0.,0.],[.08,.08,.5]])
    failed, missing, outside, lost = maths.tabletop_failures(
        p,torch.ones(3,4,dtype=torch.bool),torch.zeros(3,dtype=torch.long),torch.zeros(2),.1,40)
    assert failed.tolist() == [False,True,True]
    assert not lost.any()
