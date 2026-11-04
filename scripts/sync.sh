#!/bin/bash
# Mirror this repo to/from the agent machine (same path there; excludes .venv, runs/, third_party/, .git).
#   bash scripts/sync.sh push|pull [path]      path: a file or folder inside the repo (default: the whole repo)
# Host: hosts.agent in config/pipeline.yaml, or $RLPJ_AGENT_HOST.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${RLPJ_AGENT_HOST:-$(sed -n 's/^ *agent: *\([^ #]*\).*/\1/p' "$ROOT/config/pipeline.yaml" | head -1)}"
HOST="${HOST:-Worksation_5090}"
EXCLUDES=(--exclude /.venv --exclude /runs --exclude /third_party --exclude /.git --exclude __pycache__ --exclude '*.pyc')

usage() { echo "usage: bash scripts/sync.sh push|pull [path]"; exit 1; }
[ $# -ge 1 ] && [ $# -le 2 ] || usage
DIR="${1:-}"; SUB="${2:-}"

# resolve [path] to a path relative to the repo root
if [ -n "$SUB" ]; then
  case "$SUB" in /*) ;; *) SUB="$ROOT/$SUB" ;; esac
  SUB="$(python3 -c 'import os,sys; print(os.path.relpath(os.path.normpath(sys.argv[1]), sys.argv[2]))' "$SUB" "$ROOT")"
  case "$SUB" in ..*) echo "path is outside the repo: $2"; exit 1 ;; esac
fi
LOCAL="$ROOT/${SUB:+$SUB}"; REMOTE="$HOST:$ROOT/${SUB:+$SUB}"
# a directory syncs its contents (trailing slash); a file syncs as is
if [ -z "$SUB" ] || [ -d "$LOCAL" ]; then LOCAL="${LOCAL%/}/"; REMOTE="${REMOTE%/}/"; fi

case "$DIR" in
  push) rsync -a --delete --mkpath "${EXCLUDES[@]}" "$LOCAL" "$REMOTE" ;;
  pull) rsync -a --delete --mkpath "${EXCLUDES[@]}" "$REMOTE" "$LOCAL" ;;
  *) usage ;;
esac
echo "sync $DIR ${SUB:-.} ok ($HOST)"
