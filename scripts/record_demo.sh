#!/bin/bash
# Record one human demonstration for the one_demo condition with the lab recorder (4090, system python3).
#   bash scripts/record_demo.sh [task]      Enter = start/stop the episode, q + Enter = save and quit
# Output: ~/Repositories/kinova_realsense_recorder/data/<task>/episode_000 (what config/tasks/<task>.yaml:one_demo names).
set -euo pipefail
TASK="${1:-$(cd "$(dirname "$0")/.." && .venv/bin/python -c 'from rlpj import config; print(config.load()["task"]["name"])')}"
cd ~/Repositories/kinova_realsense_recorder
if [ -d "data/$TASK" ]; then
  echo "data/$TASK exists; episodes continue numbering. The one_demo path points at episode_000." >&2
fi
echo "Recording one demonstration of '$TASK' with side_left + front. Keep the hand visible from both cameras."
python3 record.py demo --task "$TASK"
ls -la "data/$TASK"
