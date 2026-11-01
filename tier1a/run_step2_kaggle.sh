#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$REPO"
exec "${GATE0_PYTHON:-python3}" -u tier1a/run_oracle_sweep.py "$@"
