"""Record the active project's environment for auditing, not as a portable pip lock."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'environment'
    output.mkdir(exist_ok=True)
    conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not conda:
        raise RuntimeError('Conda executable not found; set CONDA_EXE')
    # Run dependency validation before replacing the recorded snapshot.
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    commands = {
        'conda-explicit-linux-64.txt': [conda, 'list', '--explicit', '-p', sys.prefix],
        'python-packages.json': [sys.executable, '-m', 'pip', 'list', '--format=json'],
        'pip-freeze-provenance.txt': [sys.executable, '-m', 'pip', 'freeze', '--all'],
    }
    contents = {name: subprocess.check_output(command) for name, command in commands.items()}
    for name, content in contents.items():
        (output / name).write_bytes(content)
    (output / 'runtime.json').write_text(json.dumps({
        'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'prefix': sys.prefix,
        'base_prefix': sys.base_prefix,
        'isaacgym_path': os.environ.get('ISAACGYM_PATH'),
        'note': 'Conda explicit records its managed packages; pip overlays are listed separately. '
                'Local build URLs in pip-freeze are provenance, not portable download locations.',
    }, indent=2) + '\n')
    print('Environment snapshot:', output)


if __name__ == '__main__':
    main()
