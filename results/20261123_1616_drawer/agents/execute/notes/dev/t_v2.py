import json, os, sys, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = W + "/dev_out/rl_test"
os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs"); import rlcommon as rc
print("old actor rejected:", rc.Actor().info.get("source") == "prior")
r = np.random.default_rng(0); n = 24; o = r.normal(0, .3, (n, rc.OBS_DIM)); a = np.clip(r.normal(0, .3, (n, rc.ACT_DIM)), -1, 1)
tr = [dict(obs=o[i], act=a[i], rew=-.02, next_obs=o[min(i + 1, n - 1)], done=float(i == n - 1)) for i in range(n)]
rc.save_episode("train_ep0100", tr, dict(version=rc.PRIOR_VERSION, mode="train", episode=100))
