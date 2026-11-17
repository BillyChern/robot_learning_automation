#!/usr/bin/env bash
# Copy the publishable part of a run into results/<run>: reports, logs, notes, the agents' programs, readable
# transcripts, post texts and figures. Leaves out checkpoints, datasets, raw recordings and raw transcripts (too big,
# or they hold account details). Colleagues' names become roles. Fails if anything secret-looking is left.
#
#     bash scripts/export_results.sh [run]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-20261104_1510_bee}
SRC=runs/$RUN
DST=results/$RUN

rm -rf "$DST"
mkdir -p "$DST"/{agents,transcripts,videos/thread,zeroshot}
cp "$SRC"/{REPORT.md,OPERATOR_LOG.md,events.jsonl,usage.jsonl,state.json} "$DST"/
cp -r "$SRC"/{notes,logs,evals} "$DST"/
for a in execute reset learn; do
    mkdir -p "$DST/agents/$a"
    cp -r "$SRC/agents/$a"/{programs,notes} "$DST/agents/$a"/
done
cp "$SRC"/transcripts/*.md "$DST"/transcripts/
cp "$SRC"/videos/{POSTS.txt,POSTS.md} "$DST"/videos/
cp "$SRC"/videos/thread/*.png "$DST"/videos/thread/
cp runs/zeroshot_named/ANALYSIS.md "$DST"/zeroshot/
cp runs/zeroshot_blind/results.jsonl "$DST"/zeroshot/blind_results.jsonl

# colleagues' names -> roles
find "$DST" -type f \( -name '*.md' -o -name '*.log' -o -name '*.out' -o -name '*.txt' -o -name '*.json' -o -name '*.jsonl' \
    -o -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) -exec sed -i -E \
    "s/\baaron's/another user's/gI; s/\(Rai\)/(a lab member)/g; s/\bRai\b/a lab member/g; s#/aaron/#/colleague/#gI; s/\baaron\b/colleague/gI" {} +

# refuse secrets and personal emails
if grep -rIlE "sk-ant-[A-Za-z0-9_-]{10}|gh[pousr]_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30}|PRIVATE KEY|xox[bap]-|@gmail\.com|@cs\.cmu\.edu|@andrew\.cmu\.edu" "$DST"; then
    echo "export_results: secret-looking text in the files above; nothing exported" >&2
    rm -rf "$DST"
    exit 1
fi
echo "exported $SRC -> $DST ($(du -sh "$DST" | cut -f1))"
