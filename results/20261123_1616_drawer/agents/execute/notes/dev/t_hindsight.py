"""Which stored episodes give hindsight transitions, and are they consistent? Read-only on rl/."""
import glob, json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "programs"))
import learner as L, rlcommon as rc, touch
src = open(os.path.join(ROOT, "programs", "execute.py")).read()
for name, v in (("SC_DESC", L.H_SCALE), ("OBS_C", L.H_OBS)):
    assert np.allclose(eval(re.search(rf"^{name} = np.array\((\[.*?\])\)", src, re.M).group(1)), v), name
assert abs(float(re.search(r"^HOVER = ([\d.]+)", src, re.M).group(1)) + float(re.search(r"^ENTERED = ([\d.]+)", src, re.M).group(1)) - L.H_DEPTH) < 1e-9
d = os.path.join(ROOT, "rl"); used, skipped = [], []
for f in sorted(glob.glob(os.path.join(d, "transitions", "train_ep*.npz"))):
    m = json.load(open(f[:-4] + ".json"))
    if m.get("version", 1) < 2: continue
    z = np.load(f); p = {k: z[k].copy() for k in ("obs", "act", "rew", "next_obs", "done")}
    rel = L.relabel(d, f, p); h = L.hindsight(p, m, rel, L.HP["gamma"])
    if h is None: skipped.append(m["episode"]); continue
    assert np.abs(h["act"][:, :3]).max() <= L.H_MAX_A + 1e-6 and h["done"][-1] == 1 and h["done"][:-1].sum() == 0
    end = h["obs"][-1, 2:5] * L.H_OBS + L.H_SCALE * h["act"][-1, :3]
    used.append((m["episode"], m.get("arm"), len(h["rew"]), np.round(end[:2] * 1e3, 1).tolist(), round(float(h["rew"][-1]), 2), round(float(p["rew"].sum()), 2)))
for u in used: print("ep %d %s steps %d reaches a,b mm %s last reward %.2f (logged return %.2f)" % u)
print("used", len(used), "transitions", sum(u[2] for u in used), "skipped", skipped)
