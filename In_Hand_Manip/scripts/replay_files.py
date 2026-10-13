#!/usr/bin/env python3
"""Emit project-relative paths for rsync --files-from; no GPU/SDK imports."""
import argparse
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from hora.utils.replay_files import replay_transfer_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Absolute path or path relative to the project root')
    parser.add_argument('--checkpoint', default='last', help='best, last, or a saved checkpoint filename/path')
    args = parser.parse_args()
    run = args.run_dir.expanduser()
    if not run.is_absolute():
        run = MODULE.parent / run
    print('\n'.join(replay_transfer_files(MODULE.parent, run, args.checkpoint)))


if __name__ == '__main__':
    main()
