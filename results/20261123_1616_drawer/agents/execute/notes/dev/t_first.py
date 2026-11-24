import os, sys, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = W + "/dev_out/rl_tune"; os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
import learner as L, rlcommon as rc
data, _ = L.load_data(D); A = rc.Actor(); o = data["obs"]; first = (o[:, 0] < 0.5) & (o[:, 12] == 0) & (o[:, 1] < 0.01)
m = np.array([np.tanh(A.dist(x)[0]) for x in o[first]]); print("first-descent align steps", int(first.sum()), "| mean action", np.round(m.mean(0), 3).tolist(), "max |a|", np.round(np.abs(m).max(0), 3).tolist(), "-> largest shift over the 2 steps (mm):", np.round(2 * np.abs(m[:, :2]).max(0), 2).tolist())
