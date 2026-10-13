#!/usr/bin/env bash
set -euo pipefail
MANIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYDRA_FULL_ERROR=1
cd "$MANIP_ROOT"
exec "$MANIP_ROOT/scripts/python_manip.sh" "$MANIP_ROOT/train_manip.py" "$@"
