"""Where did the first descent touch or enter, relative to the frozen start? And where does the deterministic mean go now? Read-only on rl/."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "programs"))
import rlcommon as rc, touch
RL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "rl"); c0 = touch.frozen_centre(); SC = np.array([.001, .001, .5]); OC = np.array([.008, .008, 4.0])
A = rc.Actor(os.path.join(RL, "policy", "actor.npz")); print("actor updates", A.info.get("updates"))
rows = []
for f in sorted(glob.glob(os.path.join(RL, "transitions", "train_ep*.npz"))):
    m = json.load(open(f[:-4] + ".json")); ev = m.get("events") or []
    if m.get("version", 1) < 2 or not ev or m["episode"] < 11: continue
    st = np.array(ev[0][5]) - SC * np.load(f)["act"][0, :3] - c0
    if np.abs(st[:2]).max() > 0.0006: continue          # other start offset (episode 14)
    z = np.load(f); F = [i for i, e in enumerate(ev) if e[1] == "force"]
    e = next((i for i, x in enumerate(ev) if x[4] <= ev[0][4] - .012), None)
    k = F[0] if F and (e is None or F[0] < e) else e    # step of the first touch, or of the entry
    c = np.array(ev[k - 1][5], float) + SC * z["act"][k, :3] - c0 if F and k == F[0] else np.array(ev[e][5], float) - c0
    in_first = not (F and F[0] < e)
    o = z["obs"]; d = np.zeros(3)
    for i in range(min(k + 1, 4)):
        x = o[i].copy(); x[2:5] = d / OC; d = d + SC * np.tanh(A.dist(x)[0][:3])
    rows.append((m["episode"], m.get("arm", "?")[:4], in_first, c * [1e3, 1e3, 1], d * [1e3, 1e3, 1], float(np.sum(np.array([r for r in [0]])))))
print("ep arm  first-descent  offset a b mm yaw | deterministic mean now a b mm yaw")
for r in rows:
    print(f"{r[0]:2d} {r[1]} {'ENTERED' if r[2] else 'touched'}  {r[3][0]:+5.1f} {r[3][1]:+5.1f} {r[3][2]:+5.2f} | {r[4][0]:+5.2f} {r[4][1]:+5.2f} {r[4][2]:+5.2f}")
M = np.array([r[4] for r in rows]); print("mean of deterministic offsets a b yaw", np.round(M.mean(0), 2), "sd", np.round(M.std(0), 2))
