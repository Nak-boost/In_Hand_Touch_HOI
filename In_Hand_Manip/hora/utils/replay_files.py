"""Checkpoint and transfer-path selection without importing Isaac Gym or Torch."""
from pathlib import Path


def resolve_checkpoint(run, selector='last'):
    run = Path(run).resolve()
    name = Path(selector).expanduser()
    if name.is_absolute() or len(name.parts) > 1:
        candidates = [name if name.is_absolute() else run / name]
    else:
        filename = name.name if name.suffix else name.name + '.pth'
        candidates = [run / filename, run / 'stage1_nn' / filename, run / 'stage2_nn' / filename]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError('Ambiguous checkpoint; specify stage1_nn/file.pth or stage2_nn/file.pth')
    if not existing:
        raise FileNotFoundError('Checkpoint not found: ' + ', '.join(str(p) for p in candidates))
    return existing[0].resolve()


def replay_transfer_files(project_root, run, selector='last'):
    import yaml
    project_root, run = Path(project_root).resolve(), Path(run).resolve()
    config = run / 'config.yaml'
    with config.open() as stream:
        cfg = yaml.safe_load(stream)
    cache = project_root / 'In_Hand_Manip' / cfg['task']['env']['grasp']['cache']
    required = [config, resolve_checkpoint(run, selector), cache]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Required replay file is missing: ' + str(path))
    optional = [run / 'asset_metadata.json', run / 'allegro_digit.urdf']
    snapshots = [p for p in (run / 'source').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.yaml', '.yml')]
    paths = required + [p for p in optional if p.is_file()] + snapshots
    # A files-from list must stay within the rsync source root. Reject external
    # caches/checkpoints rather than silently syncing an unrelated directory.
    names = sorted({str(path.resolve().relative_to(project_root)) for path in paths})
    if any('\n' in name or '\r' in name for name in names):
        raise ValueError('Transfer paths must not contain newlines')
    return names
