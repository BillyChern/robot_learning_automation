#!/bin/bash
# After the first finished evaluation of <exp>, delete its listed intermediate checkpoints (not the deployable one,
# not one an unfinished evaluation uses).   bash scripts/cleanup_after_eval.sh <run_id> <exp> <step> [<step> ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; RUN="$ROOT/runs/$1"; EXP="$2"; shift 2
q() { python3 - "$RUN/state.json" "$EXP" "$@" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); exp = sys.argv[2]; ev = (s.get("evals") or {}).values()
if len(sys.argv) == 3:                            # done: a finished eval of exp with rollouts
    print(int(any(f"/{exp}/" in str(e.get("checkpoint")) and e.get("t_end") and e.get("n") for e in ev)))
else:                                             # in use: deployable, or an unfinished eval
    c = sys.argv[3]; d = str(s.get("deployable_checkpoint") or "")
    print(int(d.rstrip("/").endswith(c) or any(str(e.get("checkpoint")).rstrip("/").endswith(c) and not e.get("t_end") for e in ev)))
PY
}
until [ "$(q)" = 1 ]; do sleep 120; done
for step in "$@"; do
  c="$EXP/$step"; d="$RUN/checkpoints/$c"
  [ -d "$d" ] || continue
  if [ "$(q "$c")" = 1 ]; then echo "kept $c (in use)"; continue; fi
  rm -rf "$d" && echo "deleted $c"
done | tee -a /dev/stderr | sed "s/^/- $(date '+%m-%d %H:%M'): cleanup_after_eval: /" >> "$RUN/OPERATOR_LOG.md"
df -h / | tail -1
