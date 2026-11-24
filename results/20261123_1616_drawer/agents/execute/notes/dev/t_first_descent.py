"""Is there a signal for a better first-descent offset, and does the actor follow it? All real v2 episodes."""
import glob, json, os, sys, numpy as np, torch
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = W + "/dev_out/rl_tune"; os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
import learner as L, rlcommon as rc
S = {}
for f in glob.glob(W + "/attempts/attempt_*/summary.json"):
    try:
        d = json.load(open(f))
        if d.get("transitions_file") and not d.get("dry"): S[os.path.basename(d["transitions_file"])] = d
    except Exception: pass
rows = []
for f in sorted(glob.glob(D + "/transitions/*.npz"), key=os.path.getmtime):
    m = json.load(open(f[:-4] + ".json")); ev = m.get("events") or []; s = S.get(os.path.basename(f))
    if int(m.get("version", 1)) < 2 or not ev or s is None or "face_used" not in s["objects"]["drawer"]: continue      # fixed pinch height (r2 on)
    first = next((e for e in ev if e[4] <= 0.0975 or e[1] == "force"), ev[0]); hit = not any(e[1] == "force" and e[4] > 0.09 for e in ev)
    rows.append((m["episode"], s.get("arm"), first[5][0] * 1000, first[5][1] * 1000, first[5][2], s["objects"]["drawer"]["face_h"] * 1000, int(hit)))
print("fixed-pinch episodes:", len(rows), "| entered on the first descent:", sum(r[6] for r in rows))
for r in rows: print(f"  ep {r[0]:2d} {str(r[1]):12s} first offset a {r[2]:5.1f} b {r[3]:5.1f} yaw {r[4]:5.2f}  top reading {r[5]:.1f}  first-try {r[6]}")
st = torch.load(D + "/policy/learner_state.pt", map_location="cpu", weights_only=False); c = L.Critic(L.HP["hidden"]); c.load_state_dict(st["critic"]); a = L.Actor(L.HP["hidden"]); a.load_state_dict(st["actor"])
data, _ = L.load_data(D); o = torch.tensor(data["obs"]); first = (o[:, 0] < 0.5) & (o[:, 12] == 0) & (o[:, 1] < 0.01)
with torch.no_grad():
    mu, ls = a(o[first]); m = torch.tanh(mu); print("first-descent align steps", int(first.sum()), "| actor mean (a, b, yaw, dz)", np.round(m.mean(0).numpy(), 3).tolist(), "std of the mean", np.round(m.std(0).numpy(), 3).tolist())
    x0 = o[first][:1]; G = np.linspace(-1, 1, 9); print("critic Q at the first step over action a (rows) x b (cols), yaw 0:")
    for u in G[::2]:
        print("  a %+.1f: " % u + " ".join("%6.2f" % float(torch.min(*c(x0, torch.tensor([[u, v, 0.0, 0.0]], dtype=torch.float32)))) for v in G[::2]))
