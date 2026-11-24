import json, os, sys, tempfile, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs"); import touch
d = tempfile.mkdtemp(dir=W + "/dev_out"); json.dump(dict(v=2, centre=[0.00501, -0.00285, 0.1271], entry=[0.00977, -0.00244, 0.4175]), open(d + "/touch_centre.json", "w"))
print("v2 file on disk -> start", np.round(touch.load_centre(d) * [1000, 1000, 1], 2).tolist())
print("after an entry at +3.1 ->", np.round(touch.save_centre(d, [0.0031, -0.0030, 0.0], dict(episode=8)) * [1000, 1000, 1], 2).tolist(), "reloaded", np.round(touch.load_centre(d) * [1000, 1000, 1], 2).tolist())
for k in range(12): touch.save_centre(d, [0.0098, -0.0024, 0.4], dict(episode=k))
print("after 12 entries at +9.8 ->", np.round(touch.load_centre(d) * [1000, 1000, 1], 2).tolist(), "kept", len(json.load(open(d + "/touch_centre.json"))["entries"]))
def run(true, seed, noise=0.0003):
    r = np.random.default_rng(seed); T = touch.TouchSearch(36); p = np.zeros(3); n = 0
    while p is not None:
        e = p - true; n += 1
        if abs(e[0]) < 0.0018 and abs(e[1]) < 0.0012 and abs(e[2]) < 1.0: return n
        T.report(p, 0.095 + 0.22 * abs(e[0]) + 0.15 * abs(e[1]) + 0.0002 * abs(e[2]) + r.normal(0, noise)); p = T.next()
for tr in ((0, 0, 0), (0.007, 0.0005, 0.4), (-0.006, 0, 0), (0.009, -0.002, 0), (0.012, 0.004, 1.0), (-0.006, 0.006, -1.5), (0.014, -0.012, 0)):
    R = [run(np.array(tr), s) for s in range(200)]; ok = [x for x in R if x]
    print("offset", tr, "found", round(len(ok) / 200, 2), "median touches", int(np.median(ok)) if ok else None)
