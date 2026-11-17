"""Episode selection for bee_v4: successful collect/dagger episodes whose tool never goes past y = THR (the curtain
swing between Home and the bowl-look pose reaches y 0.29-0.30; the robot now refuses gripper points past y 0.30).
Writes symlinks <workspace>/sel/<name>/ep_* -> runs/<run>/episodes/ep_* for rlpj vla convert --episodes-glob.
usage: python select_episodes.py <run_dir> <name> [thr]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

run, name = Path(sys.argv[1]), sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.26
out = Path(__file__).resolve().parent.parent / "sel" / name
out.mkdir(parents=True, exist_ok=True)
for p in out.glob("ep_*"):
    p.unlink()
rows = []
for d in sorted((run / "episodes").glob("ep_*")):
    try:
        m = json.loads((d / "meta.json").read_text())
        if m.get("mode") not in ("collect", "dagger") or (m.get("outcome") or {}).get("final") != "success":
            continue
        tp = np.load(d / "steps.npz")["tool_pose"]
    except Exception:
        continue
    rows.append((d, m["mode"], float(tp[:, 1].max())))
ys = np.array([r[2] for r in rows])
for t in (0.22, 0.24, 0.26, 0.28, 0.30):
    print(f"max tool y <= {t:.2f}: {int((ys <= t).sum())} of {len(ys)}")
keep = [r for r in rows if r[2] <= thr]
for d, mode, y in keep:
    os.symlink(str(d.resolve()), out / d.name)
print(f"selected {len(keep)} (collect {sum(r[1] == 'collect' for r in keep)}, dagger {sum(r[1] == 'dagger' for r in keep)}) "
      f"with max tool y <= {thr} into {out}")
