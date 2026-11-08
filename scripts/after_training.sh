#!/bin/bash
# When <exp> finishes training on this machine (train_info.json written): serve policies locally (GPU now free),
# tell the learn agent to evaluate the final checkpoint, and switch run-all to the learn phase at its next safe point.
#   bash scripts/after_training.sh <run_id> <exp_name>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; RUN="$ROOT/runs/$1"; EXP="$2"; CK="$RUN/checkpoints/$EXP"
until [ -f "$CK/train_info.json" ]; do sleep 60; done
LAST=$(ls -d "$CK"/[0-9]* | sort -t/ -k1 -V | tail -1)
sed -i -E 's|^  policy_host: .*$|  policy_host: local               # 4090 GPU free after training (set by after_training.sh)|; s|^  policy_server: .*$|  policy_server: 127.0.0.1         # 100.122.29.80 when policy_host is the 5090|' "$ROOT/config/pipeline.yaml"
cat >> "$RUN/notes/learn.md" <<NOTE

$EXP finished training at $(date '+%F %T') on the 4090: final checkpoint $LAST.
Evaluate it now: rlpj eval $1 --checkpoint $LAST --n 20 --rollout programs/rollout_look.py (DAgger on): the
policy starts at the look pose, like the bee_v5 episodes, which start at the approach phase. The policy server
runs on the 4090 (hosts.policy_host local). If it is short of 70 %, build the next dataset the same way:
rlpj vla convert --modes collect,dagger --wrist normal --start-phase approach --drop-idle (newest collection
episodes and DAgger corrections included; only collection episodes are trimmed to the look pose). Train the next
full fine-tuning run on the 4090 from the best checkpoint so far: >= 30k steps, --batch-size 4.
NOTE
echo "- $(date '+%m-%d %H:%M'): $EXP finished ($LAST). after_training.sh: policy_host local, learn note, PHASE=learn." >> "$RUN/OPERATOR_LOG.md"
echo learn > "$RUN/PHASE"; echo "evaluate $EXP" > "$RUN/RESTART"
echo "after_training: done $(date '+%T')"
