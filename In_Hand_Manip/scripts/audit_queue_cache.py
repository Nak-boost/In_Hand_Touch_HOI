"""Offline cache/config gate between physical grasp validation and training."""
import isaacgym  # must precede torch via train_manip
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.utils.asset_geometry import verify_geometry_manifest
from hora.utils.hand_collision import collision_spec, collision_signature
from hora.tasks.tabletop_math import tabletop_collision_spec


def audit(config, quality_path, minimum):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT/'configs')):
        cfg = compose(config_name=config)
    asset = ROOT/cfg.task.env.asset.handAsset
    verify_geometry_manifest(asset)
    cache = ROOT/cfg.task.env.grasp.cache
    bank = torch.load(cache, map_location='cpu')
    quality_path = Path(quality_path).resolve()
    quality = json.loads(quality_path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert bank['asset_sha256'] == quality['asset_sha256'] == digest(asset)
    tabletop = cfg.task.name == 'AllegroDigitTabletop'
    spec = collision_spec(asset)
    if tabletop:
        spec = tabletop_collision_spec(spec, OmegaConf.to_container(cfg.task.env.tabletop, resolve=True))
    assert bank['collision_sha256'] == quality['collision_sha256'] == collision_signature(spec)
    assert bank['quality_validation']['report_sha256'] == digest(quality_path)
    assert Path(bank['quality_validation']['report']).resolve() == quality_path
    assert bank['objects'] == OmegaConf.to_container(cfg.task.env.objects, resolve=True)
    assert len(bank['joint_names']) == len(set(bank['joint_names'])) == 16
    if tabletop:
        assert quality['hold']['seconds'] >= 30 - 1e-5
        assert quality['passed'] and quality['action_reference'] == cfg.task.env.manipulation.action_reference
        assert bank['quality_validation']['action_reference'] == quality['action_reference']
        assert bank['scene'] == OmegaConf.to_container(cfg.task.env.tabletop, resolve=True)
        assert all(quality['hold']['passed'][i] for i in quality['accepted_indices'])
    else:
        assert quality['hold_seconds'] >= 30 - 1e-5
    source_path = ROOT/quality['input_cache']
    assert digest(source_path) == quality['input_cache_sha256']
    source = torch.load(source_path, map_location='cpu')
    assert source['joint_names'] == bank['joint_names']
    limits = {j.get('name'): j.find('limit') for j in ET.parse(asset).getroot().findall('joint')
              if j.get('type') != 'fixed'}
    lower = torch.tensor([float(limits[n].get('lower')) for n in bank['joint_names']])
    upper = torch.tensor([float(limits[n].get('upper')) for n in bank['joint_names']])
    total = 0
    for obj in bank['objects']:
        states = bank['states'][obj['name']]
        targets = bank['targets'][obj['name']]
        assert states.ndim == 2 and states.shape[1] == 23
        assert targets.shape == (len(states), 16)
        assert torch.isfinite(states).all() and torch.isfinite(targets).all()
        assert len(torch.unique(states, dim=0)) == len(states) >= minimum
        assert ((states[:, :16] >= lower-1e-6) & (states[:, :16] <= upper+1e-6)).all()
        assert ((targets >= lower-1e-6) & (targets <= upper+1e-6)).all()
        indices = quality['accepted_indices']
        assert torch.equal(states, source['states'][obj['name']][indices])
        assert torch.equal(targets, source['targets'][obj['name']][indices])
        total += len(states)
    assert total == quality['accepted']
    expected = dict(off_axis_cost_cap=.5, overspeed_cost_cap=.25, pose_weight=.05,
                    drop_penalty=250., limit_push_weight=.25)
    if tabletop:
        expected.update(action_reference='actual_position', fingertip_distance_weight=.1,
                        contact_weight=.05, off_axis_weight=0., overspeed_weight=0.)
    assert all(cfg.task.env.manipulation[k] == v for k, v in expected.items())
    return dict(accepted=total, cache=str(cache), cache_sha256=digest(cache),
                asset_sha256=bank['asset_sha256'], collision_sha256=bank['collision_sha256'],
                rewards=expected, config=OmegaConf.to_container(cfg, resolve=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--quality', required=True)
    p.add_argument('--minimum', type=int, default=16)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = audit(a.config, a.quality, a.minimum)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print('CACHE AUDIT PASS', result['accepted'], flush=True)
