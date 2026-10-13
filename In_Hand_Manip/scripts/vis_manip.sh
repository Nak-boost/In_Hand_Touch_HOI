#!/usr/bin/env bash
set -euo pipefail
MANIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MANIP_ROOT"
exec "$MANIP_ROOT/scripts/python_manip.sh" "$MANIP_ROOT/replay_policy.py" "$@"
