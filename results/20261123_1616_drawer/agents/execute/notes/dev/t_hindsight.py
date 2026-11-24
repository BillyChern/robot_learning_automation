"""Which stored episodes are replayed as real first-descent entries? Read-only on rl/."""
import glob, json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "programs"))
import learner as L
src = open(os.path.join(ROOT, "programs", "execute.py")).read()
assert np.allclose(eval(re.search(r"^SC_DESC = np.array\((\[.*?\])\)", src, re.M).group(1)), L.H_SCALE)
assert abs(float(re.search(r"^HOVER = ([\d.]+)", src, re.M).group(1)) + float(re.search(r"^ENTERED = ([\d.]+)", src, re.M).group(1)) - L.H_DEPTH) < 1e-9
d = os.path.join(ROOT, "rl"); used = []
for f in sorted(glob.glob(os.path.join(d, "transitions", "train_ep*.npz"))):
    m = json.load(open(f[:-4] + ".json"))
    if m.get("version", 1) < 2: continue
    z = np.load(f); p = {k: z[k].copy() for k in ("obs", "act", "rew", "next_obs", "done")}
    h = L.first_entry(p, m, L.relabel(d, f, p))
    if h is not None:
        n = len(h["rew"]); assert all(np.array_equal(h[k], p[k][:n]) for k in h); used.append((m["episode"], n))
print("replayed (episode, steps):", used)
assert {u[0] for u in used} == {19, 27}, used
