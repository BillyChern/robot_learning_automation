"""Do the real sideways forces at rim touches point toward where the drawer then entered? (operator 18:32, sim README)"""
import glob, json, os, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute/attempts/"; C = []
for n in (36, 42, 46, 48, 50, 52, 55, 57):
    d = json.load(open(f"{W}attempt_{n}/summary.json")); m = [x for x in d["insert"]["contact_map"] if x[3] > 90]; e = d["entry"]["c"] if "entry" in d else None
    if e is None: e = np.array(d["insert"]["c"]) * [1000, 1000, 1]; e = e[:2]
    else: e = np.array(e[:2]) * 1000
    for x in m:
        v = e - np.array(x[:2]); f = np.array(x[4:6]); C.append((n, np.round(f, 1).tolist(), np.round(v, 1).tolist(), float(f @ v / (np.linalg.norm(f) * np.linalg.norm(v) + 1e-9))))
print("rim touches", len(C), "| mean cosine(force, direction to the entry)", round(np.mean([c[3] for c in C]), 3), "| min", round(min(c[3] for c in C), 3), "| median |f_xy| N", round(float(np.median([np.linalg.norm(c[1]) for c in C])), 1))
