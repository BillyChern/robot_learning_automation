#!/bin/bash
# When <exp> finishes training on the agent host (the 5090): copy its final checkpoint here (weights, assets,
# train_info; no optimizer), tell the learn agent to evaluate it, and switch run-all to learn at its next safe point.
# Serving is left as it is (agent 3 serves from the 5090 through its proxy while the 4090's GPU is taken).
#   bash scripts/after_training_remote.sh <run_id> <exp_name> [host]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; RUN="$ROOT/runs/$1"; EXP="$2"; HOST="${3:-WorkStation_5090}"
CK="$RUN/checkpoints/$EXP"
until ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "test -f $CK/train_info.json"; do sleep 120; done
LAST=$(ssh -o BatchMode=yes "$HOST" "ls -d $CK/[0-9]* | sort -V | tail -1")
for i in 1 2 3 4; do
  rsync -a --partial --mkpath --exclude optimizer.pt -e 'ssh -o ServerAliveInterval=15' "$HOST:$LAST/" "$LAST/" \
    && rsync -a "$HOST:$CK/train_info.json" "$CK/" && break
  sleep 30
done
cat >> "$RUN/notes/learn.md" <<NOTE

$EXP finished training on $HOST at $(date '+%F %T'): final checkpoint $LAST (weights also copied to the 4090).
Evaluate it now: 20 rollouts from the look pose (programs/rollout_look.py), DAgger on, 240 s, served as for evals
18/19 (5090 through your proxy unless the 4090's GPU is free). Compare with full_v5 (eval 18: 13/20, eval 19).
NOTE
echo "- $(date '+%m-%d %H:%M'): $EXP finished on $HOST ($LAST); weights copied; note to agent 3; PHASE=learn." >> "$RUN/OPERATOR_LOG.md"
echo learn > "$RUN/PHASE"; echo "evaluate $EXP" > "$RUN/RESTART"
echo "after_training_remote: done $(date '+%T')"
