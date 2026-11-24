"""AWAC/SAC learner for drawer_lying (notes/design.md). Reads $RLPJ_RL_DIR/transitions/*.npz, continues from
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

PRIOR_VERSION = rc.PRIOR_VERSION
LEARNER_VERSION = 3                # 2: no entropy term in the critic target, bounded targets; 3: the report's reward (operator 20:21)
Q_RANGE = (-20.0, 20.0)            # step reward >= -1 at gamma 0.95, terminal reward <= 12

HP = dict(actor="awac", awac_lambda=0.3, awac_wmax=20.0, gamma=0.95, tau=0.005, lr=3e-4, batch=256, hidden=128, target_entropy=-2.0, init_alpha=0.05,
          utd_per_transition=40, max_updates=3000, time_budget_s=200.0, bc_weight=1.0, bc_decay_episodes=10.0,
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
                   bmu=sd["mu.bias"], Wls=sd["ls.weight"], bls=sd["ls.bias"], updates=np.array(updates), prior_version=np.array(PRIOR_VERSION))
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


def verdict_for(d, meta):
    """The orchestrator's verdict for this episode ($RLPJ_RL_DIR/verdicts/<episode>.json), or None."""
    ep = meta.get("episode", -1)
    if ep is None or int(ep) < 0:
        return None
    vd = os.path.join(d, "verdicts")
    for name in (f"{int(ep):06d}.json", f"{int(ep)}.json", f"{int(ep):04d}.json", f"ep_{int(ep):04d}.json", f"{meta.get('mode')}_{int(ep)}.json",
                 f"{meta.get('mode')}_ep{int(ep):04d}.json"):
        p = os.path.join(vd, name)
        if os.path.exists(p):
            try:
                v = json.load(open(p))
            except Exception:
                continue
            if v.get("mode") and meta.get("mode") and str(v["mode"]).replace("rl_", "") not in (meta["mode"], "rl"):
                if not str(v["mode"]).endswith(meta["mode"]):
                    continue
            return v
    return None


def any_seated(meta):
    """The episode's last logged drawer bottom is at the compartment floor (commanded height, m)."""
    ev = meta.get("events") or []
    return bool(ev) and float(ev[-1][4]) < 0.03


def relabel(d, f, part):
    """Rewards of a stored episode from the report's formula (rc.fixed_rewards), whatever reward it was logged with. Success: the verdict; an
    unsure verdict with the vision judge positive and the drawer at the floor (episodes 5, 6: the check's cabinet comparison misfired on a
    filled compartment) counts, as does my own check for test runs that have no verdict."""
    try:
        meta = json.load(open(f[:-4] + ".json"))
    except Exception:
        return None
    ev = meta.get("events") or []
    if len(ev) != len(part["rew"]) or not len(ev):
        return None
    v = verdict_for(d, meta)
    final = None if v is None else str(v.get("final", "")).lower()
    if v is None:
        success = meta.get("check") is True
    else:
        success = final == "success" or (final == "unsure" and "vlm=true" in json.dumps(v).lower() and any_seated(meta))
    part["rew"] = rc.fixed_rewards(len(ev), [i for i, e in enumerate(ev) if e[1] == "force"], ev[-1][6], success)
    return dict(final=final, success=bool(success))


def load_data(d):
    files = sorted(glob.glob(os.path.join(d, "transitions", "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz") and not os.path.basename(f).startswith("dry_")]
    parts, eps = [], []
    for f in files:
        try:
            try:                               # only data whose actions and observations mean what they mean now
                if int(json.load(open(f[:-4] + ".json")).get("version", 1)) < rc.PRIOR_VERSION:
                    continue
            except Exception:
                continue
            z = np.load(f)
            if len(z["rew"]):
                part = {k: z[k].copy() for k in ("obs", "act", "rew", "next_obs", "done")}
                rel = relabel(d, f, part)
                parts.append(part)
                eps.append(dict(file=os.path.basename(f), n=int(len(part["rew"])), ret=float(part["rew"].sum()), relabel=rel))
        except Exception as e:
            print(f"skip {f}: {e}")
    if not parts:
        return None, eps
    D = {k: np.concatenate([p[k] for p in parts]).astype(np.float32) for k in parts[0]}
    for k in ("obs", "next_obs"):                  # unobserved inputs (insert stage only; push uses 5/6 for its own data)
        ins = D[k][:, 0] < 0.5
        for j in rc.ZERO_OBS:
            D[k][ins, j] = 0.0
    D["prior"] = rc.prior(D["obs"]).astype(np.float32)
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

    def bc_fit(self, T, steps=600):
        """Fit the actor's mean to the scripted prior on the stored observations (fresh learner state)."""
        n = len(T["obs"])
        for _ in range(steps):
            idx = torch.randint(0, n, (min(HP["batch"], n),), device=self.dev)
            mu, ls = self.actor(T["obs"][idx])
            loss = ((torch.tanh(mu) - T["prior"][idx]) ** 2).sum(-1).mean() + 0.1 * ((ls - math.log(rc.INIT_STD)) ** 2).mean()
            self.oa.zero_grad(); loss.backward(); self.oa.step()
        return float(loss)

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
        o, a, r, o2, d, pr = B
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2, _ = self.actor.sample(o2)
            q1t, q2t = self.target(o2, a2)
            soft = 0.0 if HP["actor"] == "awac" else alpha * lp2     # AWAC needs no entropy bonus; with near-constant logged actions it diverged
            y = (r + HP["gamma"] * (1 - d) * (torch.min(q1t, q2t) - soft)).clamp(*Q_RANGE)
        q1, q2 = self.critic(o, a)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        an, lp, mean_a = self.actor.sample(o)
        if HP["actor"] == "awac":
            # AWAC (Nair et al. 2020): regress the actor on the logged actions weighted by exp(advantage / lambda);
            # stays on the data (no exploitation of critic errors with ~300 transitions), V(s) = Q(s, pi sample)
            with torch.no_grad():
                qd = torch.min(*self.critic(o, a))
                qv = torch.min(*self.critic(o, an))
                w = torch.clamp(torch.exp((qd - qv) / HP["awac_lambda"]), max=HP["awac_wmax"])
            mu, ls = self.actor(o)
            ac = torch.clamp(a, -0.999, 0.999)
            u = 0.5 * torch.log((1 + ac) / (1 - ac))
            logp = (-0.5 * ((u - mu) / ls.exp()) ** 2 - ls).sum(-1) - torch.log(1 - ac ** 2 + 1e-6).sum(-1)
            la = -(w * logp).mean() + bc_w * ((torch.tanh(mu) - pr) ** 2).sum(-1).mean()
            self.last_w = float(w.mean())
        else:
            q1n, q2n = self.critic(o, an)
            la = (alpha * lp - torch.min(q1n, q2n)).mean() + bc_w * ((mean_a - pr) ** 2).sum(-1).mean()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        if HP["actor"] != "awac":
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
            if st.get("meta", {}).get("prior_version") == PRIOR_VERSION and st.get("meta", {}).get("learner_version") == LEARNER_VERSION:
                L.load(st); prev = st.get("meta", prev)
            else:
                print(f"learner state is from prior/learner version {st.get('meta', {}).get('prior_version')}/{st.get('meta', {}).get('learner_version')}: starting fresh on all data")
        except Exception as e:
            print(f"learner state not loaded ({e}): starting fresh")
    D, eps = load_data(d)
    n = 0 if D is None else len(D["rew"])
    print(f"data: {len(eps)} episodes, {n} transitions; device {dev}; updates so far {L.updates}")
    if check:
        if D is None:
            D = dict(obs=np.zeros((4, rc.OBS_DIM), np.float32), act=np.zeros((4, rc.ACT_DIM), np.float32),
                     rew=np.zeros(4, np.float32), next_obs=np.zeros((4, rc.OBS_DIM), np.float32), done=np.ones(4, np.float32))
            D["prior"] = rc.prior(D["obs"]).astype(np.float32)
            print("check: no real data yet, synthetic batch")
        T = {k: torch.tensor(v, device=dev) for k, v in D.items()}
        idx = torch.randint(0, len(T["rew"]), (min(HP["batch"], len(T["rew"])),), device=dev)
        info = L.update([T[k][idx] for k in ("obs", "act", "rew", "next_obs", "done", "prior")], 0.5)
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
        if prev.get("prior_version") != PRIOR_VERSION:      # prior/BC target changed: one full-budget pass
            n_up = HP["max_updates"]
        bc_w = HP["bc_weight"] * math.exp(-len(eps) / HP["bc_decay_episodes"])
        if prev.get("prior_version") is None:            # fresh learner only: never wipe what the actor learned
            stats_bc = L.bc_fit(T)
            print(f"actor fitted to prior version {PRIOR_VERSION}: loss {stats_bc:.4f}")
        acc = []
        for i in range(n_up):
            idx = torch.randint(0, n, (HP["batch"],), device=dev)
            acc.append(L.update([T[k][idx] for k in ("obs", "act", "rew", "next_obs", "done", "prior")], bc_w))
            if time.time() - t0 > HP["time_budget_s"]:
                n_up = i + 1; break
        stats = {k: round(float(np.mean([a[k] for a in acc[-100:]])), 4) for k in acc[0]}
        stats["bc_w"] = round(bc_w, 4)
        with torch.no_grad():
            _, _, m = L.actor.sample(T["obs"][-min(n, 200):])
            stats["mean_act"] = np.round(m.mean(0).cpu().numpy(), 3).tolist()
    st = L.state(); st["meta"] = dict(n_trans=n, episodes=len(eps), t=time.time(), learner_version=LEARNER_VERSION,
                                      prior_version=PRIOR_VERSION if (D is not None and n >= HP["min_transitions"]) else prev.get("prior_version"))
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
