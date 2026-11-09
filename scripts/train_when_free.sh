#!/bin/bash
# Full fine-tuning on this machine's GPU when it is free: wait until NEED_MB is free for STABLE_S seconds, run one
# train_guard.sh attempt (it resumes from the newest checkpoint), and if that stops early (another job took the GPU,
# OOM) wait for the GPU again. The 4090 is shared with other users' jobs; training never moves to another machine.
#   bash scripts/train_when_free.sh <run_dir> <dataset> <exp_name> [train-pytorch overrides]
# Env: NEED_MB (free GPU memory to start, default 23000), STABLE_S (default 120), RLPJ_CONFIG / RLPJ_RECIPE as train_guard.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(realpath -m "$1")"; NAME="$2"; EXP="$3"; shift 3
CKPT="$RUN/checkpoints/$EXP"; LOG="$RUN/logs/train_${EXP}.log"
NEED="${NEED_MB:-23000}"; STABLE="${STABLE_S:-120}"
CONFIG="${RLPJ_CONFIG:-pi05_kinova_full}"
NORM="$RUN/assets/$CONFIG/rlpj/$NAME/norm_stats.json"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
mkdir -p "$(dirname "$LOG")"
until [ -f "$NORM" ]; do sleep 30; done
while [ ! -f "$CKPT/train_info.json" ]; do
  ok=0
  echo "[train_when_free] $(date '+%F %T') waiting for $NEED MB free on the GPU (now $(free_mb) MB)" | tee -a "$LOG"
  while [ "$ok" -lt "$STABLE" ]; do
    if [ "$(free_mb)" -ge "$NEED" ]; then ok=$((ok + 10)); else ok=0; fi
    sleep 10
  done
  echo "[train_when_free] $(date '+%F %T') GPU free ($(free_mb) MB): starting $EXP" | tee -a "$LOG"
  RETRIES=0 bash "$ROOT/scripts/train_guard.sh" "$RUN" "$NAME" "$EXP" "$@"
  [ -f "$CKPT/train_info.json" ] || sleep 60
done
echo "[train_when_free] $(date '+%F %T') $EXP finished" | tee -a "$LOG"
