import json, os, sys, tempfile, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs"); import touch
E = np.array(json.load(open(W + "/rl/touch_centre.json"))["entries"]); f = lambda c: np.round(np.asarray(c) * [1000, 1000, 1], 2).tolist()
print("real entries a mm:", np.round(E[:, 0] * 1000, 1).tolist()); print("old median", f(np.median(E, axis=0)), "-> densest", f(touch.densest(E)))
d = tempfile.mkdtemp(dir=W + "/dev_out"); json.dump(dict(v=3, centre=E[-1].tolist(), entries=E.tolist()), open(d + "/touch_centre.json", "w")); print("load", f(touch.load_centre(d)))
for k in range(1, 7): c = touch.save_centre(d, [0.0060, -0.0031, 0.0], dict(episode=k)); print(" after", k, "more entries at +6.0 ->", f(c))
print("single entry", f(touch.densest(np.array([[0.004, -0.002, 0.1]]))), "| tie, latest wins", f(touch.densest(np.array([[0.003, -0.003, 0], [0.006, -0.003, 0]]))))
