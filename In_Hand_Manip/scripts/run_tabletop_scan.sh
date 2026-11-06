#!/usr/bin/env bash
set -euo pipefail
MANIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$MANIP_ROOT/scripts/python_manip.sh" "$MANIP_ROOT/scripts/run_tabletop_scan.py" "$@"
