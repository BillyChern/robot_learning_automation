"""One video of a whole run: every recorded program run (dashboard.mp4: side + wrist cameras with overlays, status bar)
in time order, each after a title card (time, role, attempt, what it tried, outcome). Reads finished files only.

    python -m rlpj.orchestrator.timeline <run> [--robot-only] [--out PATH]      (also: rlpj timeline <run>)"""
import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

from rlpj import config

W, H = 1280, 480


def clips(run_dir, robot_only=False):
    """[(started, attempt_dir, attempt.json)] of every attempt with a dashboard.mp4 (agents' and policy rollouts), oldest first."""
    run_dir = Path(run_dir)
    out = []
    for d in list(run_dir.glob("agents/*/attempts/attempt_*")) + list(run_dir.glob("episodes/ep_*/attempt")):
        v, m = d / "dashboard.mp4", d / "attempt.json"
        if not (v.exists() and m.exists()):
            continue
        meta = json.loads(m.read_text())
        if "exit_code" not in meta or (robot_only and not meta.get("moves_robot")):     # still running, or a dry run
            continue
        out.append((float(meta.get("started") or v.stat().st_mtime), d, meta))
    return sorted(out, key=lambda c: c[0])


def label(d):
    """'execute attempt_3' for agents/<role>/attempts/attempt_N, 'ep_000012 policy rollout' for episodes/<ep>/attempt."""
    return f"{d.parents[1].name} {d.name}" if d.parent.name == "attempts" else f"{d.parent.name} policy rollout"


def card(t0, started, d, meta):
    """Black title frame: clock time and time since the run started, what ran, its outcome."""
    img = np.zeros((H, W, 3), np.uint8)
    kind = "ROBOT" if meta.get("moves_robot") else "dry run"
    lines = [(time.strftime("%H:%M:%S", time.localtime(started)) + f"   (+{(started - t0) / 60:.1f} min)", 1.0, (200, 200, 200)),
             (f"{label(d)}   [{kind}]", 1.3, (0, 255, 255) if meta.get("moves_robot") else (255, 255, 255))]
    lines += [(s, 0.8, (255, 255, 255)) for s in textwrap.wrap(str(meta.get("what") or ""), 90)[:4]]
    lines.append((f"exit {meta.get('exit_code')}, {meta.get('outcome')}, {float(meta.get('duration_s') or 0):.0f} s", 1.0,
                  (0, 200, 0) if meta.get("outcome") == "success" else (0, 0, 255)))
    y = 70
    for text, scale, col in lines:
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += int(55 * scale)
    return img


def build(run_dir, out=None, robot_only=False, card_s=2.0, fps=8.0, log=print):
    """Write the timeline video; returns (path, number of clips, seconds of video)."""
    run_dir = Path(run_dir)
    out = Path(out or run_dir / ("timeline_robot.mp4" if robot_only else "timeline.mp4"))
    cs = clips(run_dir, robot_only)
    st = run_dir / "state.json"
    t0 = time.mktime(time.strptime(json.loads(st.read_text())["created"], "%Y-%m-%dT%H:%M:%S")) if st.exists() else (cs[0][0] if cs else 0)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = 0
    try:
        for started, d, meta in cs:
            c = card(t0, started, d, meta)
            for _ in range(int(card_s * fps)):
                wr.write(c)
                n += 1
            cap = cv2.VideoCapture(str(d / "dashboard.mp4"))
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                wr.write(f if f.shape[:2] == (H, W) else cv2.resize(f, (W, H)))
                n += 1
            cap.release()
    finally:
        wr.release()
    log(f"timeline: {len(cs)} runs, {n / fps / 60:.1f} min of video -> {out}")
    return out, len(cs), n / fps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="run id (runs/<run>)")
    p.add_argument("--robot-only", action="store_true", help="only runs that moved the robot")
    p.add_argument("--out", help="output path (default runs/<run>/timeline.mp4)")
    a = p.parse_args(argv)
    build(config.RUNS / a.run, a.out, a.robot_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
