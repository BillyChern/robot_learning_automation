"""RL progress plots for a run: runs/<run>/plots/*.png (return growth, success, evaluations, timing).

    third_party/openpi/.venv/bin/python scripts/rl_plots.py <run> [--every 600]

Reads runs/<run>/rl_curve.jsonl (one row per training/eval episode, written by the orchestrator) and events.jsonl
(resets, episodes). Plots: return per training episode with a rolling mean, rolling and cumulative success, fixed
evaluation success rate, episode and reset durations, reset success, wall-clock timeline. With --every it redraws
in a loop (the status page and the final video use the PNGs).
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def roll(x, w):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))]) if len(x) else x


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def arm_of(d, ep):
    """'scripted' or 'learned' from the episode's summary (the program writes arm=...); 'learned' when missing."""
    try:
        return json.loads((d / "episodes" / f"ep_{int(ep):06d}" / "summary.json").read_text()).get("arm") or "learned"
    except (OSError, ValueError, TypeError):
        return "learned"


def draw(run):
    d = ROOT / "runs" / run
    out = d / "plots"
    out.mkdir(exist_ok=True)
    cur = rows(d / "rl_curve.jsonl")
    ev = rows(d / "events.jsonl")
    tr = [r for r in cur if r.get("kind") == "train"]
    evals = [r for r in cur if r.get("kind") == "eval"]
    resets = [e for e in ev if e.get("kind") == "reset"]
    eps = [e for e in ev if e.get("kind") == "episode"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if tr:
        n = np.arange(1, len(tr) + 1)
        ret = [float(r.get("reward") or 0) for r in tr]
        ok = np.array([r.get("final") == "success" for r in tr])
        ax.plot(n, ret, "o", color="#9aa7b5", ms=4, label="return per episode")
        ax.plot(n, roll(ret, 5), "-", color="#1f5fa8", lw=2.2, label="rolling mean (5)")
        if ok.any():
            ax.plot(n[ok], np.asarray(ret)[ok], "*", color="#2e9e4f", ms=11, label="success")
        ax.legend(frameon=False, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no training episodes yet", ha="center", va="center", transform=ax.transAxes, color="#777")
    style(ax, "Return growth", "training episode", "episode return")
    fig.tight_layout(); fig.savefig(out / "return.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if tr:
        n = np.arange(1, len(tr) + 1)
        ok = np.array([r.get("final") == "success" for r in tr], float)
        ax.plot(n, roll(ok, 10) * 100, "-", color="#1f5fa8", lw=2.2, label="training success, rolling 10")
        ax2 = ax.twinx()
        ax2.step(n, np.cumsum(ok), where="post", color="#2e9e4f", lw=1.6, label="cumulative successes")
        ax2.set_ylabel("cumulative successes")
        ax2.spines["top"].set_visible(False)
    arms = [arm_of(d, r.get("episode")) for r in tr]
    if "scripted" in arms:
        for arm, col in (("learned", "#1f5fa8"), ("scripted", "#e67e22"), ("sim", "#8e44ad")):
            idx = [i for i, a in enumerate(arms) if a.startswith(arm)]
            if not idx:
                continue
            ok = np.array([tr[i].get("final") == "success" for i in idx], float)
            ax.plot(np.array(idx) + 1, np.cumsum(ok) / np.arange(1, len(ok) + 1) * 100, "o-", color=col, ms=3, lw=1.6,
                    label=f"{arm}: running success ({int(ok.sum())}/{len(ok)})")
    if evals:
        ax.plot([r["after"] for r in evals], [100 * float(r.get("rate") or 0) for r in evals], "s", color="#c0392b",
                ms=9, label="fixed evaluation")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, loc="upper left")
    style(ax, "Success rate", "training episode", "success (%)")
    fig.tight_layout(); fig.savefig(out / "success.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    if eps:
        axs[0].plot(np.arange(1, len(eps) + 1), [float(e.get("duration") or 0) / 60 for e in eps], "o-", color="#1f5fa8", ms=3)
    style(axs[0], "Episode duration", "episode", "minutes")
    if resets:
        dur = [float(e.get("duration") or 0) / 60 for e in resets]
        okr = [bool(e.get("ok")) for e in resets]
        axs[1].scatter(np.arange(1, len(resets) + 1), dur, c=["#2e9e4f" if o else "#c0392b" for o in okr], s=14)
        axs[1].set_title(f"Reset duration (green ok, red failed): {sum(okr)}/{len(okr)} ok", fontsize=12, loc="left")
    style(axs[1], axs[1].get_title() or "Reset duration", "reset", "minutes")
    fig.tight_layout(); fig.savefig(out / "timing.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 2.6))
    t = lambda e: datetime.fromisoformat(e["t"])                       # noqa: E731
    for e in resets:
        ax.plot([t(e)], [1], "|", color="#2e9e4f" if e.get("ok") else "#c0392b", ms=18)
    for e in eps:
        ax.plot([t(e)], [0], "|", color="#1f5fa8" if e.get("final") != "success" else "#2e9e4f", ms=18, mew=2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["RL episodes", "resets"]); ax.set_ylim(-0.6, 1.6)
    style(ax, "Timeline", "", "")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out / "timeline.png", dpi=140); plt.close(fig)
    (out / "summary.json").write_text(json.dumps(dict(
        t=time.strftime("%F %T"), train_episodes=len(tr), successes=sum(r.get("final") == "success" for r in tr),
        evals=[dict(after=r["after"], rate=r.get("rate")) for r in evals], resets=len(resets),
        resets_ok=sum(bool(e.get("ok")) for e in resets),
        arms={a: dict(n=sum(1 for r in tr if arm_of(d, r.get("episode")) == a),
                      successes=sum(1 for r in tr if arm_of(d, r.get("episode")) == a and r.get("final") == "success"))
              for a in ("learned", "scripted", "sim")}), indent=1))


def main():
    run = sys.argv[1]
    every = float(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 0
    while True:
        draw(run)
        if not every:
            break
        time.sleep(every)


if __name__ == "__main__":
    main()
