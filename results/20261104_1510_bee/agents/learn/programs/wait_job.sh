#!/bin/bash
# Wait (max ~19 min, under the orchestrator's 1200 s attempt timeout) for the detached training of <exp>; report state.
EXP="$1"; RUN=/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee
end=$((SECONDS + ${2:-1140}))
while pgrep -f "openpi_entry train.*--exp-name $EXP" >/dev/null && [ $SECONDS -lt $end ]; do sleep 20; done
if pgrep -f "openpi_entry train.*--exp-name $EXP" >/dev/null; then echo "STILL RUNNING $(date +%T)"; else echo "FINISHED (process gone) $(date +%T)"; fi
ls "$RUN/checkpoints/$EXP" 2>&1 | tr "\n" " "; echo; grep -E "Progress on|loss=|rror" "$(dirname "$0")/../notes/train_$EXP.out" 2>/dev/null | tail -3 | cut -c1-140
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
