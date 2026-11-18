"""SAC learner for drawer_rl (notes/design.md). Reads $RLPJ_RL_DIR/transitions/*.npz, continues from
policy/learner_state.pt, writes policy/actor.npz (numpy, for execute.py) and a line to learner_log.jsonl.
--check: load, one update, write nothing, exit 0 when fine."""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlcommon as rc

HP = dict(gamma=0.95, tau=0.005, lr=3e-4, batch=256, hidden=128, target_entropy=-2.0, init_alpha=0.05,
          utd_per_transition=40, max_updates=3000, time_budget_s=200.0, bc_weight=1.0, bc_decay_episodes=30.0,
          min_transitions=8)


def mlp(i, o, h, ln=False):
    layers = [nn.Linear(i, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, o)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l0, self.l1 = nn.Linear(rc.OBS_DIM, h), nn.Linear(h, h)
        self.mu, self.ls = nn.Linear(h, rc.ACT_DIM), nn.Linear(h, rc.ACT_DIM)
        for l in (self.mu, self.ls):
            nn.init.uniform_(l.weight, -1e-3, 1e-3)
        with torch.no_grad():
            self.mu.bias.copy_(torch.tensor(rc.atanh(rc.PRIOR), dtype=torch.float32))
            self.ls.bias.fill_(math.log(rc.INIT_STD))

    def forward(self, o):
        h = F.relu(self.l1(F.relu(self.l0(o))))
        return self.mu(h), self.ls(h).clamp(rc.LOG_STD_MIN, rc.LOG_STD_MAX)

    def sample(self, o):
        mu, ls = self(o)
        std = ls.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1) - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp, torch.tanh(mu)

    def to_npz(self, path, updates):
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.state_dict().items()}
        arr = dict(W0=sd["l0.weight"], b0=sd["l0.bias"], W1=sd["l1.weight"], b1=sd["l1.bias"], Wmu=sd["mu.weight"],
                   bmu=sd["mu.bias"], Wls=sd["ls.weight"], bls=sd["ls.bias"], updates=np.array(updates))
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arr)
        os.replace(tmp, path)


class Critic(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q1, self.q2 = mlp(rc.OBS_DIM + rc.ACT_DIM, 1, h, ln=True), mlp(rc.OBS_DIM + rc.ACT_DIM, 1, h, ln=True)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def load_data(d):
    files = sorted(glob.glob(os.path.join(d, "transitions", "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz") and not os.path.basename(f).startswith("dry_")]
    parts, eps = [], []
    for f in files:
        try:
            z = np.load(f)
            if len(z["rew"]):
                parts.append({k: z[k] for k in ("obs", "act", "rew", "next_obs", "done")})
                eps.append(dict(file=os.path.basename(f), n=int(len(z["rew"])), ret=float(z["rew"].sum())))
        except Exception as e:
            print(f"skip {f}: {e}")
    if not parts:
        return None, eps
    D = {k: np.concatenate([p[k] for p in parts]).astype(np.float32) for k in parts[0]}
    return D, eps


class Learner:
    def __init__(self, dev):
        h = HP["hidden"]
        self.dev = dev
        self.actor, self.critic, self.target = Actor(h).to(dev), Critic(h).to(dev), Critic(h).to(dev)
        self.target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(math.log(HP["init_alpha"]), device=dev, requires_grad=True)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=HP["lr"])
        self.oc = torch.optim.Adam(self.critic.parameters(), lr=HP["lr"])
        self.oal = torch.optim.Adam([self.log_alpha], lr=HP["lr"])
        self.updates = 0
        self.prior_u = torch.tensor(rc.PRIOR, dtype=torch.float32, device=dev)

    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), target=self.target.state_dict(),
                    log_alpha=self.log_alpha.detach().cpu(), oa=self.oa.state_dict(), oc=self.oc.state_dict(),
                    oal=self.oal.state_dict(), updates=self.updates)

    def load(self, st):
        self.actor.load_state_dict(st["actor"]); self.critic.load_state_dict(st["critic"]); self.target.load_state_dict(st["target"])
        with torch.no_grad():
            self.log_alpha.copy_(st["log_alpha"].to(self.dev))
        self.oa.load_state_dict(st["oa"]); self.oc.load_state_dict(st["oc"]); self.oal.load_state_dict(st["oal"])
        self.updates = int(st["updates"])

    def update(self, B, bc_w):
        o, a, r, o2, d = B
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2, _ = self.actor.sample(o2)
            q1t, q2t = self.target(o2, a2)
            y = r + HP["gamma"] * (1 - d) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = self.critic(o, a)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        an, lp, mean_a = self.actor.sample(o)
        q1n, q2n = self.critic(o, an)
        la = (alpha * lp - torch.min(q1n, q2n)).mean() + bc_w * ((mean_a - self.prior_u) ** 2).sum(-1).mean()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        lal = -(self.log_alpha * (lp.detach() + HP["target_entropy"])).mean()
        self.oal.zero_grad(); lal.backward(); self.oal.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                pt.mul_(1 - HP["tau"]).add_(HP["tau"] * p)
        self.updates += 1
        return dict(lc=float(lc), la=float(la), q=float(q1.mean()), alpha=float(alpha), ent=float(-lp.mean()))


def main():
    check = "--check" in sys.argv
    d = rc.rl_dir()
    pol = os.path.join(d, "policy")
    dev = os.environ.get("RLPJ_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if dev.startswith("cuda"):
        try:                                  # the 5090 (sm_120) has no kernels in this torch build
            torch.randint(0, 2, (4,), device=dev).sum().item()
            (torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev)).sum().item()
        except Exception as e:
            print(f"device {dev} unusable ({str(e)[:60]}): cpu")
            dev = "cpu"
    t0 = time.time()
    torch.manual_seed(int(t0) % 100000)
    L = Learner(dev)
    st_path = os.path.join(pol, "learner_state.pt")
    prev = dict(n_trans=0)
    if os.path.exists(st_path):
        try:
            st = torch.load(st_path, map_location=dev, weights_only=False)
            L.load(st); prev = st.get("meta", prev)
        except Exception as e:
            print(f"learner state not loaded ({e}): starting fresh")
    D, eps = load_data(d)
    n = 0 if D is None else len(D["rew"])
    print(f"data: {len(eps)} episodes, {n} transitions; device {dev}; updates so far {L.updates}")
    if check:
        if D is None:
            D = dict(obs=np.zeros((4, rc.OBS_DIM), np.float32), act=np.zeros((4, rc.ACT_DIM), np.float32),
                     rew=np.zeros(4, np.float32), next_obs=np.zeros((4, rc.OBS_DIM), np.float32), done=np.ones(4, np.float32))
            print("check: no real data yet, synthetic batch")
        T = {k: torch.tensor(v, device=dev) for k, v in D.items()}
        idx = torch.randint(0, len(T["rew"]), (min(HP["batch"], len(T["rew"])),), device=dev)
        info = L.update([T[k][idx] for k in ("obs", "act", "rew", "next_obs", "done")], 0.5)
        ok = all(np.isfinite(list(info.values())))
        print(f"check: one update {info}; finite {ok}")
        sys.exit(0 if ok else 1)
    os.makedirs(pol, exist_ok=True)
    n_new = n - int(prev.get("n_trans", 0))
    stats = {}
    n_up = 0
    if D is not None and n >= HP["min_transitions"]:
        T = {k: torch.tensor(v, device=dev) for k, v in D.items()}
        n_up = int(min(HP["max_updates"], max(200, HP["utd_per_transition"] * max(n_new, 1))))
        bc_w = HP["bc_weight"] * math.exp(-len(eps) / HP["bc_decay_episodes"])
        acc = []
        for i in range(n_up):
            idx = torch.randint(0, n, (HP["batch"],), device=dev)
            acc.append(L.update([T[k][idx] for k in ("obs", "act", "rew", "next_obs", "done")], bc_w))
            if time.time() - t0 > HP["time_budget_s"]:
                n_up = i + 1; break
        stats = {k: round(float(np.mean([a[k] for a in acc[-100:]])), 4) for k in acc[0]}
        stats["bc_w"] = round(bc_w, 4)
        with torch.no_grad():
            _, _, m = L.actor.sample(T["obs"][-min(n, 200):])
            stats["mean_act"] = np.round(m.mean(0).cpu().numpy(), 3).tolist()
    st = L.state(); st["meta"] = dict(n_trans=n, episodes=len(eps), t=time.time())
    torch.save(st, st_path + ".tmp"); os.replace(st_path + ".tmp", st_path)
    L.actor.to_npz(os.path.join(pol, "actor.npz"), L.updates)
    rets = [e["ret"] for e in eps]
    line = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), episodes=len(eps), transitions=n, new=n_new, updates=n_up,
                total_updates=L.updates, device=dev, secs=round(time.time() - t0, 1),
                ret_last10=round(float(np.mean(rets[-10:])), 3) if rets else None, **stats)
    with open(os.path.join(d, "learner_log.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line))


if __name__ == "__main__":
    main()
