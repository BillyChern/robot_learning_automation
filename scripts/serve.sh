#!/bin/bash
# Serve a checkpoint with openpi's websocket policy server (4090 by default, or 5090).  bash scripts/serve.sh <checkpoint_dir> [port]
#   <checkpoint_dir> = <run_dir>/checkpoints/<exp_name>/<step>  (must contain params/ and assets/)
#   rlpj eval does this itself (hosts.policy_host). On the 5090 by hand:  ssh Worksation_5090 'nohup bash ~/robot_learning_final_pj/scripts/serve.sh \
#       ~/robot_learning_final_pj/runs/r1/checkpoints/lora_5k/4999 8000 >/dev/null 2>&1 &'
#   stop:  ssh Worksation_5090 'pkill -f "rlpj.vla.openpi_entry serve"'
# Env: RLPJ_CONFIG / RLPJ_RECIPE (default: from <exp>/train_info.json, else pi05_kinova_lora), RLPJ_SERVE_PROGRAM
# (agent 3's server: --checkpoint --port --config); JAX does not preallocate so training can share the GPU.
set -euo pipefail
export UV_NO_SYNC=1    # keep the venv as set up (torch cu128, transformers patch for PyTorch checkpoints)
[ $# -ge 1 ] || { echo "usage: $0 <checkpoint_dir> [port]" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$(realpath "$1")"; PORT="${2:-8000}"
INFO="$(dirname "$CKPT")/train_info.json"      # written by training: config name and agent 3's recipe, if any
info() { [ -f "$INFO" ] && python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$INFO" "$1" || true; }
CONFIG="${RLPJ_CONFIG:-$(info config)}"; CONFIG="${CONFIG:-pi05_kinova_lora}"
RECIPE="${RLPJ_RECIPE:-$(info recipe)}"; [ -n "$RECIPE" ] && export RLPJ_RECIPE="$RECIPE"
[ -d "$CKPT/params" ] || { echo "no params/ in $CKPT" >&2; exit 2; }
case "$CKPT" in */checkpoints/*) LOG="${CKPT%%/checkpoints/*}/logs" ;; *) LOG="$ROOT/runs/logs" ;; esac
mkdir -p "$LOG"
TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
cd "$ROOT/third_party/openpi"
echo "[serve.sh] $CKPT on port $PORT (config $CONFIG), log $LOG/serve_${TAG}.log"
if [ -n "${RLPJ_SERVE_PROGRAM:-}" ]; then        # agent 3's own server (e.g. real-time chunking); its workspace on PYTHONPATH
  export PYTHONPATH="$(dirname "$(dirname "$RLPJ_SERVE_PROGRAM")"):$PYTHONPATH"
  echo "[serve.sh] agent server $RLPJ_SERVE_PROGRAM (config $CONFIG${RECIPE:+, recipe $RECIPE})"
  uv run python "$RLPJ_SERVE_PROGRAM" --checkpoint "$CKPT" --port "$PORT" --config "$CONFIG" 2>&1 | tee "$LOG/serve_${TAG}.log"
else
  uv run python -m rlpj.vla.openpi_entry serve --port "$PORT" policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT" \
    2>&1 | tee "$LOG/serve_${TAG}.log"
fi
