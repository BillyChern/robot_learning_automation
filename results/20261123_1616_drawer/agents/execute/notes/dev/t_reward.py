"""The learner's relabelled return against the report's formula, episode by episode, on the real summaries and verdicts."""
import glob, json, os, re, sys, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = sys.argv[1]; os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
import learner as L
def fixed_return(r, s):
    ins = s.get("insert") or {}
    return (10.0 * (r.get("final") == "success") - 1.0 * (ins.get("contacts") or 0) - 0.1 * max(0.0, (ins.get("peak") or 0) - 15.0) - 0.05 * (ins.get("steps") or 0))
S = {}
for f in glob.glob(W + "/attempts/attempt_*/summary.json"):
    try:
        d = json.load(open(f))
        if d.get("transitions_file"): S[os.path.basename(d["transitions_file"])] = d
    except Exception: pass
bad = 0
for f in sorted(glob.glob(D + "/transitions/train_ep*.npz")):
    m = json.load(open(f[:-4] + ".json"))
    if int(m.get("version", 1)) < 2 or not m.get("events"): continue
    z = np.load(f); part = {k: z[k].copy() for k in ("rew",)}; rel = L.relabel(D, f, part); ep = m["episode"]; v = json.load(open(f"{D}/verdicts/{ep:06d}.json")); fr = fixed_return(v, S[os.path.basename(f)]); dv = float(part["rew"].sum()) - fr
    flag = "" if abs(dv) < 1e-3 else ("  (+10: unsure verdict, real insertion)" if abs(dv - 10) < 1e-3 else "  MISMATCH"); bad += "MISMATCH" in flag
    print(f"ep {ep:2d} {str(S[os.path.basename(f)].get('arm', '-')):8s} verdict {v['final']:8s} report {fr:6.2f} learner {part['rew'].sum():6.2f}{flag}")
print("mismatches:", bad)
