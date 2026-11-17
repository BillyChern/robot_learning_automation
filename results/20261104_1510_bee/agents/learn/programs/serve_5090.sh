#!/bin/bash
# Start (or restart) the openpi policy server for one checkpoint on this GPU host (5090), detached, then wait until
# the port answers. Usage: bash programs/serve_5090.sh <checkpoint_dir> [port]
set -uo pipefail
ROOT=/home/r2r/robot_learning_final_pj
WS="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$1"; PORT="${2:-8000}"
LOG="$WS/notes/serve5090_${PORT}.log"
pkill -f "[o]penpi_entry serve --port $PORT" ; sleep 3
cd "$ROOT/third_party/openpi"
PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false setsid nohup uv run python -m rlpj.vla.openpi_entry serve \
  --port "$PORT" policy:checkpoint --policy.config pi05_kinova_lora --policy.dir "$CKPT" > "$LOG" 2>&1 < /dev/null &
echo "started pid $! serving $CKPT on port $PORT, log $LOG"
for i in $(seq 1 120); do
  if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then echo "port $PORT up after $((i*5)) s"; tail -5 "$LOG"; exit 0; fi
  sleep 5
done
echo "server did not come up in 600 s"; tail -30 "$LOG"; exit 2
