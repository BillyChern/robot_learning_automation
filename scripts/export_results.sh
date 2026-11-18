#!/usr/bin/env bash
# Copy the publishable part of a run into results/<run>: reports, logs, notes, the agents' programs and notes, readable
# transcripts, the RL learning curve, post texts and figures (whatever the run has). Leaves out checkpoints, datasets,
# raw recordings, attempt folders and raw transcripts (too big, or they hold account details). Colleagues' names become
# roles. Fails if anything secret-looking is left.
#
#     bash scripts/export_results.sh [run]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-20261104_1510_bee}
SRC=runs/$RUN
DST=results/$RUN

cp_if() {   # cp_if <src> <dst dir>: copy a file or folder when it exists
    [ -e "$1" ] && mkdir -p "$2" && cp -r "$1" "$2"/ || true
}

rm -rf "$DST"
mkdir -p "$DST"
for f in REPORT.md OPERATOR_LOG.md events.jsonl usage.jsonl state.json rl_curve.jsonl notes logs evals resets; do
    cp_if "$SRC/$f" "$DST"
done
for a in execute reset learn; do
    cp_if "$SRC/agents/$a/programs" "$DST/agents/$a"
    cp_if "$SRC/agents/$a/notes" "$DST/agents/$a"
    cp_if "$SRC/agents/$a/PROMPT.md" "$DST/agents/$a"
    cp_if "$SRC/agents/$a/rl/learner_log.jsonl" "$DST/agents/$a/rl"
done
# readable transcripts (rlpj transcripts <run> renders them from the raw ones)
.venv/bin/rlpj transcripts "$RUN" >/dev/null 2>&1 || true
for t in "$SRC"/transcripts/*.md; do cp_if "$t" "$DST/transcripts"; done
cp_if "$SRC/videos/POSTS.txt" "$DST/videos"
cp_if "$SRC/videos/POSTS.md" "$DST/videos"
for p in "$SRC"/videos/thread/*.png; do cp_if "$p" "$DST/videos/thread"; done
if [ "$RUN" = 20261104_1510_bee ]; then
    cp_if runs/zeroshot_named/ANALYSIS.md "$DST/zeroshot"
    [ -f runs/zeroshot_blind/results.jsonl ] && mkdir -p "$DST/zeroshot" && cp runs/zeroshot_blind/results.jsonl "$DST/zeroshot/blind_results.jsonl"
fi

# colleagues' names -> roles
find "$DST" -type f \( -name '*.md' -o -name '*.log' -o -name '*.out' -o -name '*.txt' -o -name '*.json' -o -name '*.jsonl' \
    -o -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) -exec sed -i -E \
    "s/\baaron's/another user's/gI; s/\(Rai\)/(a lab member)/g; s/\bRai\b/a lab member/g; s#/aaron/#/colleague/#gI; s/\baaron\b/colleague/gI" {} +

# refuse secrets and personal emails
if grep -rIlE "sk-ant-[A-Za-z0-9_-]{10}|gh[pousr]_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30}|PRIVATE[ ]KEY|xox[bap]-|@gmail\.com|@cs\.cmu\.edu|@andrew\.cmu\.edu" "$DST"; then
    echo "export_results: secret-looking text in the files above; nothing exported" >&2
    rm -rf "$DST"
    exit 1
fi
echo "exported $SRC -> $DST ($(du -sh "$DST" | cut -f1))"
