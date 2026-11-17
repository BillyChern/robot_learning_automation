"""One line per eval episode: duration, first close time, grip max/end, lowest tool z and its xy distance to the bee,
the last stop reason / decision. usage: python eval_brief.py <run_dir> <first> <last>"""
import json
import re
import sys
from pathlib import Path

import numpy as np

run, a, b = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
KEYS = ("stop record", "traceback", "exception", "ignored", "release_descend", "time limit", "stop_reason", "refus")
for i in range(a, b + 1):
    d = run / "episodes" / f"ep_{i:06d}"
    try:
        m = json.loads((d / "meta.json").read_text())
        s = np.load(d / "steps.npz")
    except Exception:
        continue
    if m.get("mode") != "eval":
        continue
    t, g, gc, tp = s["t"], s["gripper"], s["gripper_cmd"], s["tool_pose"]
    c = np.where(gc > 0.5)[0]
    bee = np.array((m.get("layout_target") or {}).get("bee", [np.nan, np.nan]))
    bowl = np.array((m.get("layout_target") or {}).get("bowl", [np.nan, np.nan]))
    if len(c):
        k = c[0]
        dclose = np.linalg.norm(tp[k, :2] - bee) * 100
        zc = tp[k, 2]
    else:
        dclose, zc = np.nan, np.nan
    lines = (d / "log.txt").read_text().splitlines() if (d / "log.txt").exists() else []
    why = [x for x in lines if any(k in x.lower() for k in KEYS)]
    why = re.sub(r"\s+", " ", why[-1])[:110] if why else ""
    held = np.where((g > 60) & (g < 92))[0]
    dbowl = np.min(np.linalg.norm(tp[held, :2] - bowl, axis=1)) * 100 if len(held) else np.nan
    print(f"ep{i} {(m.get('outcome') or {}).get('final')} dur={t[-1]-t[0]:.0f}s close@{(t[c[0]]-t[0]) if len(c) else -1:.0f}s "
          f"d_bee={dclose:.1f}cm z={zc:.3f} gmax={g.max():.0f} gend={g[-1]:.0f} held_min_d_bowl={dbowl:.0f}cm | {why}")
