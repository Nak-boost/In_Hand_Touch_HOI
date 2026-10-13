"""Validate externally referenced geometry before accepting an asset/cache hash."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def verify_geometry_manifest(asset):
    asset = Path(asset)
    robot = ET.parse(asset).getroot()
    relative = robot.get('geometry_manifest')
    if relative is None:
        return  # Historical assets retain their original URDF-only identity.
    manifest_path = asset.parent / relative
    expected = robot.get('geometry_manifest_sha256')
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Geometry manifest changed; rebuild the URDF and regenerate caches')
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files_sha256'].items():
        path = manifest_path.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Geometry mesh changed without regeneration: '+str(path))
    for name, digest in manifest['hand_visual_sha256'].items():
        path = asset.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Hand visual mesh changed without regeneration: '+str(path))
