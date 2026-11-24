"""Critic health on a copy of the real data: Q(s, a) against the empirical discounted return-to-go, for a learner state."""
import glob, json, os, sys, numpy as np, torch
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = sys.argv[1]; os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
import learner as L, rlcommon as rc
data, eps = L.load_data(D); st = torch.load(D + "/policy/learner_state.pt", map_location="cpu", weights_only=False); c = L.Critic(L.HP["hidden"]); c.load_state_dict(st["critic"]); a = L.Actor(L.HP["hidden"]); a.load_state_dict(st["actor"])
G = np.zeros(len(data["rew"])); g = 0.0
for i in range(len(G) - 1, -1, -1): g = data["rew"][i] + L.HP["gamma"] * (1 - data["done"][i]) * g; G[i] = g
with torch.no_grad():
    o, ac = torch.tensor(data["obs"]), torch.tensor(data["act"]); Q = torch.min(*c(o, ac)).numpy(); mu, ls = a(o)
print(f"episodes {len(eps)} transitions {len(G)} | return-to-go {G.min():.2f}..{G.max():.2f} mean {G.mean():.2f} | Q {Q.min():.2f}..{Q.max():.2f} mean {Q.mean():.2f} | MAE {np.abs(Q - G).mean():.2f} corr {np.corrcoef(Q, G)[0, 1]:.2f} | actor |mean| {np.abs(np.tanh(mu.numpy())).mean():.3f} max {np.abs(np.tanh(mu.numpy())).max():.3f} std {np.exp(ls.numpy()).mean():.3f} | updates {st['updates']}")
