import json, os, sys, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = W + "/dev_out/rel_t"
os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs"); import learner as L
f = D + "/transitions/train_ep0007.npz"; json.dump(dict(version=2, mode="train", episode=7, terminal=dict(bonus=10.0)), open(f[:-4] + ".json", "w"))
for final, reason, want in (("fail", "no independent check says success", 0.0), ("fail", "drawer is upside down", -5.0), ("success", "ok", 10.0)):
    json.dump(dict(episode=7, mode="rl", final=final, reason=reason), open(D + "/verdicts/000007.json", "w"))
    part = dict(rew=np.array([0.0, 10.0])); r = L.relabel(D, f, part); print(final, reason[:28], "->", r["new"], "ok" if r["new"] == want else "WRONG", "last reward", float(part["rew"][-1]))
