#!/bin/bash
# 5090: norm stats + fine-tune inside the openpi venv.  bash scripts/train.sh <run_dir> <dataset_name> <exp_name> [tyro overrides]
#   e.g. bash scripts/train.sh runs/r1 bee_v1 lora_5k --num-train-steps 5000 --batch-size 16 --overwrite
#   from the 4090: ssh WorkStation_5090 'bash ~/robot_learning_final_pj/scripts/train.sh ~/robot_learning_final_pj/runs/r1 bee_v1 lora_5k --overwrite'
# Env: RLPJ_CONFIG (default pi05_kinova_full: PyTorch full fine-tuning; *lora* configs use the JAX trainer), RLPJ_WANDB=1 enables wandb, XLA_PYTHON_CLIENT_MEM_FRACTION (default 0.9).
# Output: <run_dir>/checkpoints/<exp_name>/<step>/, <run_dir>/assets/<config>/rlpj/<dataset>/norm_stats.json, logs in <run_dir>/logs/.
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <run_dir> <dataset_name> <exp_name> [tyro overrides]" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(realpath -m "$1")"; NAME="$2"; EXP="$3"; shift 3
CONFIG="${RLPJ_CONFIG:-pi05_kinova_full}"   # operator 2026-11-07: full fine-tuning only (PyTorch trainer)
export UV_NO_SYNC=1                          # keep the venv as set up (torch cu128 for the 5090, transformers patch)
TRAINER=train; case "$CONFIG" in *full*) TRAINER=train-pytorch;; esac
LOG="$RUN/logs"; mkdir -p "$LOG"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
cd "$ROOT/third_party/openpi"
echo "[train.sh] norm stats: run=$RUN dataset=$NAME config=$CONFIG"
uv run python -m rlpj.vla.openpi_entry norm-stats --run-dir "$RUN" --dataset "$NAME" --config-name "$CONFIG" \
  2>&1 | tee "$LOG/norm_stats_${NAME}.log"
echo "[train.sh] train: exp=$EXP overrides: $*"
uv run python -m rlpj.vla.openpi_entry "$TRAINER" --run-dir "$RUN" --dataset "$NAME" "$CONFIG" --exp-name "$EXP" "$@" \
  2>&1 | tee "$LOG/train_${EXP}.log"
echo "[train.sh] done: $RUN/checkpoints/$EXP"
