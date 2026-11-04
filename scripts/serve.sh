#!/bin/bash
# 5090: serve a checkpoint with openpi's websocket policy server.  bash scripts/serve.sh <checkpoint_dir> [port]
#   <checkpoint_dir> = <run_dir>/checkpoints/<exp_name>/<step>  (must contain params/ and assets/)
#   from the 4090 (background):  ssh Worksation_5090 'nohup bash ~/robot_learning_final_pj/scripts/serve.sh \
#       ~/robot_learning_final_pj/runs/r1/checkpoints/lora_5k/4999 8000 >/dev/null 2>&1 &'
#   stop:  ssh Worksation_5090 'pkill -f "rlpj.vla.openpi_entry serve"'
# Env: RLPJ_CONFIG (default pi05_kinova_lora); JAX does not preallocate so training can share the GPU.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <checkpoint_dir> [port]" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$(realpath "$1")"; PORT="${2:-8000}"
CONFIG="${RLPJ_CONFIG:-pi05_kinova_lora}"
[ -d "$CKPT/params" ] || { echo "no params/ in $CKPT" >&2; exit 2; }
case "$CKPT" in */checkpoints/*) LOG="${CKPT%%/checkpoints/*}/logs" ;; *) LOG="$ROOT/runs/logs" ;; esac
mkdir -p "$LOG"
TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
cd "$ROOT/third_party/openpi"
echo "[serve.sh] $CKPT on port $PORT (config $CONFIG), log $LOG/serve_${TAG}.log"
uv run python -m rlpj.vla.openpi_entry serve policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT" --port "$PORT" \
  2>&1 | tee "$LOG/serve_${TAG}.log"
