"""Does the actor's first-descent mean move toward the offsets where the drawer actually entered? Read-only on rl/."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "programs"))
import rlcommon as rc, touch
RL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "rl")
c0 = touch.frozen_centre()
SC = np.array([.001, .001, .5])
A = rc.Actor(os.path.join(RL, "policy", "actor.npz"))
print("actor", A.info.get("updates"), "c0 mm", np.round(c0[:2] * 1000, 2), "yaw", round(c0[2], 3))
rows = []
for f in sorted(glob.glob(os.path.join(RL, "transitions", "train_ep*.json"))):
    m = json.load(open(f)); z = np.load(f[:-5] + ".npz"); ev = m["events"]
    if m.get("version", 1) < 2 or not ev: continue
    forces = [i for i, e in enumerate(ev) if e[1] == "force"]
    first = forces[0] if forces else len(ev)
    # entry: c at the last lateral change before the bottom went below the rim by >= 4 mm
    rim = float(ev[0][4]) - 0.008 + 0.0  # ev[0] is at the hover height RIM + HOVER
    ent = next((np.array(e[5]) for e in ev if e[4] <= rim - 0.004), None)
    if ent is None: continue
    # deterministic rollout of the mean over the logged first-descent observations (before the first contact)
    c = np.zeros(3); 
    for i in range(min(first, len(z["obs"]))):
        o = z["obs"][i].copy(); o[2:5] = c / np.array([.008, .008, 4.0])
        if float(ev[i][4]) <= rim: break
        mu, _ = A.dist(o); c = c + SC * np.tanh(mu[:3])
    rows.append((m["episode"], m.get("arm", "?"), len(forces), (ent - c0), c, np.array(ev[max(first - 1, 0)][5]) - c0))
print("ep arm  rim  entry-c0 (a b mm, yaw)     actor mean shift (a b mm, yaw)   logged c at 1st contact")
for r in rows:
    print(f"{r[0]:2d} {r[1]:8s} {r[2]}  {r[3][0]*1e3:+5.1f} {r[3][1]*1e3:+5.1f} {r[3][2]:+5.2f}    {r[4][0]*1e3:+5.2f} {r[4][1]*1e3:+5.2f} {r[4][2]:+5.2f}    {r[5][0]*1e3:+5.1f} {r[5][1]*1e3:+5.1f}")
R = [r for r in rows if r[0] >= 11]
E = np.array([r[3] for r in R]); M = np.array([r[4] for r in R])
print("episodes", len(R), "entry mean (a,b mm, yaw)", np.round(E.mean(0)[:2]*1e3,2), round(E.mean(0)[2],2), "median", np.round(np.median(E,0)[:2]*1e3,2))
print("actor mean shift mean", np.round(M.mean(0)[:2]*1e3,3), round(M.mean(0)[2],3))
cos = (E[:,:2]*M[:,:2]).sum(1)/(np.linalg.norm(E[:,:2],axis=1)*np.linalg.norm(M[:,:2],axis=1)+1e-12)
print("cosine(actor shift, entry) per ep: mean", round(float(cos.mean()),2), "n>0:", int((cos>0).sum()), "of", len(cos))
print("|shift| mm mean", round(float(np.linalg.norm(M[:,:2],axis=1).mean()*1e3),3), " |entry| mm mean", round(float(np.linalg.norm(E[:,:2],axis=1).mean()*1e3),2))
