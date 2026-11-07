#!/bin/bash
# Full fine-tuning on this machine with automatic resume: rerun train-pytorch with --resume from the newest checkpoint
# when it stops before train_info.json is written (crash, OOM from another GPU job), up to RETRIES times.
#   bash scripts/train_guard.sh <run_dir> <dataset> <exp_name> [train-pytorch overrides, e.g. --num-train-steps 30000 --batch-size 4]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(realpath -m "$1")"; NAME="$2"; EXP="$3"; shift 3
CKPT="$RUN/checkpoints/$EXP"; LOG="$RUN/logs/train_${EXP}.log"; RETRIES="${RETRIES:-6}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" UV_NO_SYNC=1
cd "$ROOT/third_party/openpi"
for n in $(seq 0 "$RETRIES"); do
  if ls -d "$CKPT"/[0-9]* >/dev/null 2>&1; then MODE=--resume; else MODE=--overwrite; fi
  echo "[train_guard] $(date '+%F %T') start $n ($MODE)" | tee -a "$LOG"
  nice -n 10 uv run python -m rlpj.vla.openpi_entry train-pytorch --run-dir "$RUN" --dataset "$NAME" pi05_kinova_full \
    --exp-name "$EXP" "$MODE" "$@" >> "$LOG" 2>&1
  [ -f "$CKPT/train_info.json" ] && { echo "[train_guard] $(date '+%F %T') finished" | tee -a "$LOG"; exit 0; }
  echo "[train_guard] $(date '+%F %T') stopped early (exit $?); resuming in 60 s" | tee -a "$LOG"; sleep 60
done
echo "[train_guard] gave up after $RETRIES restarts" | tee -a "$LOG"; exit 1
