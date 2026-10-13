#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/fixed_30_20_server32.yaml}"
OUTPUT="${OUTPUT:-outputs/fixed_30_20_server32_w72}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Prevent glibc from creating many memory arenas in each long-lived process.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

echo "[preflight] validating data, solver sources, parameters, and all 8,000 tasks"
"$PYTHON_BIN" run.py --config "$CONFIG" --dry-run

if [[ -f "$OUTPUT/manifest.json" ]]; then
  echo "[resume] $OUTPUT"
  exec "$PYTHON_BIN" -u run.py --config "$CONFIG" --resume "$OUTPUT"
fi

if [[ -e "$OUTPUT" ]]; then
  echo "[error] output exists but has no manifest: $OUTPUT" >&2
  exit 2
fi

echo "[start] $OUTPUT"
exec "$PYTHON_BIN" -u run.py --config "$CONFIG" --output "$OUTPUT"
