#!/usr/bin/env bash
# Project Python with the native-library setup required by Isaac Gym Preview 4.
set -euo pipefail
MANIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$MANIP_ROOT/.env.local" ]]; then source "$MANIP_ROOT/.env.local"; fi
: "${ISAACGYM_PATH:?Set ISAACGYM_PATH in .env.local or your environment}"
MANIP_CONDA_PREFIX="${MANIP_CONDA_PREFIX:-/root/conda_envs/inhand_manip}"
MANIP_PYTHON="${MANIP_PYTHON:-$MANIP_CONDA_PREFIX/bin/python}"
if [[ ! -x "$MANIP_PYTHON" ]]; then
    echo "Project Python missing: $MANIP_PYTHON. See docs/environment.md." >&2
    exit 1
fi
# Do not inherit Python packages from another activated environment or user site.
unset PYTHONHOME
export PYTHONNOUSERSITE=1
MANIP_BASE="$("$MANIP_PYTHON" -c 'import sys; print(sys.base_prefix)')"
export PATH="$(dirname "$MANIP_PYTHON"):$MANIP_BASE/bin:$PATH"
export ISAACGYM_PATH
export PYTHONPATH="$MANIP_ROOT:$ISAACGYM_PATH/python"
export LD_LIBRARY_PATH="$MANIP_BASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$MANIP_PYTHON" "$@"
