#!/bin/bash
# Full fine-tuning on the 5090 (run there) with agent 3's train_5090.py (torch cu128 for sm_120), resuming from the
# newest checkpoint after a crash or power-off, up to RETRIES times. The 5090 runs with ~/power-mitigation installed
# (400 W, GPU clock <= 2200 MHz, CPU 125/200 W): it powered off 3 times under training load on 2026-11-07 without it.
#   bash scripts/train_guard_5090.sh <run_dir> <dataset> <exp_name> [train-pytorch overrides]
# Norm stats must exist (SKIP_NORM=1). Env: RLPJ_CONFIG (default pi05_kinova_full), RETRIES (default 6).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(realpath -m "$1")"; NAME="$2"; EXP="$3"; shift 3
CKPT="$RUN/checkpoints/$EXP"; LOG="$RUN/logs/train_${EXP}.log"; RETRIES="${RETRIES:-6}"
CONFIG="${RLPJ_CONFIG:-pi05_kinova_full}"
PROG="$RUN/agents/learn/programs/train_5090.py"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT/third_party/openpi"
export PYTHONPATH="$ROOT" UV_NO_SYNC=1 SKIP_NORM=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
for n in $(seq 0 "$RETRIES"); do
  if ls -d "$CKPT"/[0-9]* >/dev/null 2>&1; then MODE=--resume; else MODE=--overwrite; fi
  echo "[train_guard_5090] $(date '+%F %T') start $n ($MODE, $CONFIG)" | tee -a "$LOG"
  nice -n 10 .venv/bin/python "$PROG" --run-dir "$RUN" --dataset "$NAME" --exp-name "$EXP" --config "$CONFIG" "$MODE" "$@" >> "$LOG" 2>&1
  [ -f "$CKPT/train_info.json" ] && { echo "[train_guard_5090] $(date '+%F %T') finished" | tee -a "$LOG"; exit 0; }
  echo "[train_guard_5090] $(date '+%F %T') stopped early; resuming in 60 s" | tee -a "$LOG"; sleep 60
done
echo "[train_guard_5090] gave up after $RETRIES restarts" | tee -a "$LOG"; exit 1
