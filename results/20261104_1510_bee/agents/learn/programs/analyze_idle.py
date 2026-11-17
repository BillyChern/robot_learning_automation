"""Idle-frame and phase statistics of the collect/success training episodes (runs on the 5090).
usage: python analyze_idle.py <run_dir>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
fr, ph_t, ph_idle, speed = [], defaultdict(list), defaultdict(list), []
grasp = []
for d in sorted((run / "episodes").glob("ep_*")):
    try:
        m = json.loads((d / "meta.json").read_text())
    except Exception:
        continue
    if m.get("mode") != "collect" or (m.get("outcome") or {}).get("final") != "success":
        continue
    s = np.load(d / "steps.npz")
    qc, gc, t = s["q_cmd"], s["gripper_cmd"], s["t"]
    dq = np.abs(np.diff(qc, axis=0)).max(1)
    dg = np.abs(np.diff(gc))
    idle = (dq < 0.02) & (dg < 1e-3)
    fr.append(idle.mean())
    speed.append(np.percentile(dq[~idle], 90) * 15 if (~idle).any() else 0)
    phases = json.loads((d / "phases.json").read_text())
    for p in phases:
        i0, i1 = p["i0"], min(p["i1"], len(idle))
        if i1 > i0:
            ph_t[p["name"]].append((i1 - i0) / 15)
            ph_idle[p["name"]].append(idle[i0:i1].mean())
    # gripper reading after first close command
    c = np.where(gc > 0.5)[0]
    if len(c):
        grasp.append(float(s["gripper"][min(c[0] + 45, len(gc) - 1)]))
print(f"episodes {len(fr)}; idle fraction median {np.median(fr):.2f} (p10 {np.percentile(fr, 10):.2f}, p90 {np.percentile(fr, 90):.2f})")
print(f"moving joint speed p90 deg/s median {np.median(speed):.1f}")
print(f"grip reading 3 s after close: median {np.median(grasp):.1f}, min {np.min(grasp):.1f}, max {np.max(grasp):.1f}")
for k in ph_t:
    print(f"phase {k:14s} n={len(ph_t[k]):3d} dur median {np.median(ph_t[k]):6.1f}s idle {np.median(ph_idle[k]):.2f}")
