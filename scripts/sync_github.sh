#!/usr/bin/env bash
# Keep github.com/BillyChern/robot_learning_automation up to date: export the runs' results (RUNS), commit code + results on
# the current branch, push it to the remote's main and to a branch of the same name, and upload changed showcase
# videos to the release "bee-run-videos". Repeats every EVERY seconds (ONCE=1 for a single pass).
#
#     tmux new -d -s github_sync 'bash scripts/sync_github.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-github}
EVERY=${EVERY:-1800}
RUNS=${RUNS:-"20261104_1510_bee 20261117_2026_drawer"}   # runs exported to results/
RUN=20261104_1510_bee                                    # the run whose showcase videos go to the release
TAG=bee-run-videos
export GH_TOKEN="$(gh auth token -u "${GH_USER:-BillyChern}")"   # this account only, the machine default stays
SECRETS="sk-ant-[A-Za-z0-9_-]{10}|gh[pousr]_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30}|PRIVATE[ ]KEY|@gmail\.com|@cs\.cmu\.edu|@andrew\.cmu\.edu"

sync_once() {
    for r in $RUNS; do bash scripts/export_results.sh "$r" || return 1; done
    git add -A
    if ! git diff --cached --quiet; then
        if git diff --cached | grep -E '^\+' | grep -qE "$SECRETS"; then      # added lines only
            echo "$(date '+%F %T') secret-looking text staged; nothing committed. Files:"
            git diff --cached --name-only | while read -r f; do git show ":$f" 2>/dev/null | grep -qE "$SECRETS" && echo "  $f"; done
            git reset -q; return 1
        fi
        git commit -q -m "Sync: $(date '+%F %H:%M')" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    fi
    branch=$(git rev-parse --abbrev-ref HEAD)
    git push -q "$REMOTE" "HEAD:refs/heads/main" "HEAD:refs/heads/$branch" || return 1
    gh release view "$TAG" -R BillyChern/robot_learning_automation >/dev/null 2>&1 || \
        gh release create "$TAG" -R BillyChern/robot_learning_automation --title "Bee run: showcase videos" \
            --notes "Videos of run $RUN (toy bee into a bowl, pi0.5 fine-tuned by Claude Code agents). The thread texts are in results/$RUN/videos/POSTS.txt." || return 1
    sums=.git/release_sums
    touch "$sums"
    for f in runs/"$RUN"/videos/0*.mp4; do
        s="$(md5sum "$f" | cut -c1-32) $(basename "$f")"
        grep -qxF "$s" "$sums" && continue
        gh release upload "$TAG" "$f" --clobber -R BillyChern/robot_learning_automation || return 1
        grep -v " $(basename "$f")$" "$sums" > "$sums.tmp" || true
        echo "$s" >> "$sums.tmp"; mv "$sums.tmp" "$sums"
    done
    echo "$(date '+%F %T') synced"
}

while :; do
    sync_once || echo "$(date '+%F %T') sync failed (will retry)"
    [ "${ONCE:-0}" = 1 ] && break
    sleep "$EVERY"
done
