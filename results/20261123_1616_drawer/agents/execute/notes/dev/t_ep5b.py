import json, os, sys, tempfile, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = tempfile.mkdtemp(dir=W + "/dev_out"); os.makedirs(D + "/verdicts"); os.makedirs(D + "/transitions")
os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs"); import learner as L
f = D + "/transitions/train_ep0005.npz"
for bottom, reason, want in ((0.0122, "the independent checks disagree (claim=none, perception=false, vlm=true)", 10.0), (0.095, "the independent checks disagree (claim=none, perception=false, vlm=true)", None), (0.0122, "disagree (perception=true, vlm=false)", None)):
    json.dump(dict(version=2, mode="train", episode=5, terminal=dict(bonus=0.0), events=[[0, "ok", [0], 0, bottom]]), open(f[:-4] + ".json", "w"))
    json.dump(dict(episode=5, mode="rl", final="unsure", reason=reason), open(D + "/verdicts/000005.json", "w"))
    part = dict(rew=np.array([0.0, 1.0])); print("bottom", bottom, reason[-32:], "->", L.relabel(D, f, part), "last reward", float(part["rew"][-1]), "want bonus", want)
