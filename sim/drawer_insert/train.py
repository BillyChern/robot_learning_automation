"""PPO with reset-to-states (SGS-style curriculum) for the drawer insertion; exports a numpy policy (.npz).

nice -n 15 python -m sim.drawer_insert.train --name run1 [--engine cpu|newton] [--iters 300]"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import params as P
from .curriculum import Curriculum
from .policy import OBS_DIM, H_STEPS

RUNS = Path(__file__).resolve().parent / "runs"


def mlp(n_in, n_out, hidden=256):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi, self.v = mlp(OBS_DIM, 3), mlp(OBS_DIM, 1)
        self.log_std = nn.Parameter(torch.full((3,), np.log(0.5)))
        with torch.no_grad():                      # prior: push about 5.5 mm per step, no correction
            self.pi[-1].weight.mul_(0.1); self.pi[-1].bias.copy_(torch.tensor([np.arctanh(0.4), 0.0, 0.0]))

    def dist(self, x):
        return torch.distributions.Normal(torch.tanh(self.pi(x)), self.log_std.exp())

    def export(self, path):
        """Actor weights as x @ W + b layers (policy.Policy)."""
        lin = [m for m in self.pi if isinstance(m, nn.Linear)]
        np.savez(path, **{f"W{i}": m.weight.detach().cpu().numpy().T for i, m in enumerate(lin)},
                 **{f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(lin)})


def make_env(engine, n, seed, f_pen):
    if engine == "newton":
        from . import newton_env
        newton_env.F_PEN = f_pen
        return newton_env.NewtonEnv(n, seed=seed)
    from .vec import CpuPool
    return CpuPool(num_workers=12, per_worker=n // 12, seed=seed, f_pen=f_pen)


def rollout(env, agent, starts, dev, deterministic=False):
    """One synchronized batch of episodes -> dict of (T, N) tensors and final infos."""
    obs = env.reset(starts)
    N = len(starts)
    active = np.ones(N, bool)
    buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "mask", "done")}
    info_end = {}
    for _ in range(H_STEPS + 1):
        if not active.any():
            break
        x = torch.as_tensor(obs, device=dev)
        with torch.no_grad():
            d = agent.dist(x)
            a = d.mean if deterministic else d.sample()
            logp, val = d.log_prob(a).sum(-1), agent.v(x).squeeze(-1)
        obs2, rew, done, info = env.step(np.clip(a.cpu().numpy(), -1, 1))
        for k, v in (("obs", x), ("act", a), ("logp", logp), ("val", val)):
            buf[k].append(v)
        buf["rew"].append(torch.as_tensor(rew * active, dtype=torch.float32, device=dev))
        buf["mask"].append(torch.as_tensor(active.copy(), device=dev))
        fin = active & done
        buf["done"].append(torch.as_tensor(fin, device=dev))
        for k, v in info.items():
            info_end.setdefault(k, np.zeros(N))[fin] = v[fin]
        active &= ~done
        obs = obs2
    return {k: torch.stack(v) for k, v in buf.items()}, info_end


def gae(b, gamma=0.99, lam=0.95):
    T = b["rew"].shape[0]
    adv = torch.zeros_like(b["rew"])
    last = torch.zeros_like(b["rew"][0])
    for t in reversed(range(T)):
        nv = b["val"][t + 1] if t + 1 < T else torch.zeros_like(last)
        nonterm = (~b["done"][t]).float()
        delta = b["rew"][t] + gamma * nv * nonterm - b["val"][t]
        last = (delta + gamma * lam * nonterm * last) * b["mask"][t].float()
        adv[t] = last
    return adv, adv + b["val"]


def update(agent, opt, b, adv, ret, epochs=4, mb=4096, clip=0.2):
    m = b["mask"].reshape(-1)
    x, a, lp0 = b["obs"].reshape(-1, OBS_DIM)[m], b["act"].reshape(-1, 3)[m], b["logp"].reshape(-1)[m]
    adv, ret = adv.reshape(-1)[m], ret.reshape(-1)[m]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    out = []
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=x.device).split(mb):
            d = agent.dist(x[idx])
            ratio = (d.log_prob(a[idx]).sum(-1) - lp0[idx]).exp()
            pl = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            vl = ((agent.v(x[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            loss = pl + 0.5 * vl - 0.001 * d.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step()
            with torch.no_grad():
                agent.log_std.clamp_(np.log(0.05), np.log(1.0))
            out.append((pl.item(), vl.item()))
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--engine", default="cpu")
    ap.add_argument("--envs", type=int, default=384)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fresh", type=float, default=0.4)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--init", default="")
    ap.add_argument("--f_pen", type=float, default=0.05, help="reward per 10 N of peak force above 40 N")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(a.engine, a.envs, a.seed, a.f_pen)
    agent = Agent().to(dev)
    if a.init:
        agent.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr)
    cur = Curriculum(size=max(4096, 4 * a.envs), fresh=a.fresh, seed=a.seed)
    rng = np.random.default_rng(12345)
    eval_starts = [dict(p=P.sample(np.random.default_rng(200000 + i)), kind=None, k_pre=0) for i in range(a.envs)]
    log = csv.writer(open(out / "log.csv", "a", newline="", buffering=1))
    log.writerow(["iter", "samples", "fresh_success", "bank_success", "bank_p", "ret", "max_f_med", "std", "pl", "vl",
                  "eval_success", "eval_max_f_med", "sec"])
    best, samples, t0 = -1.0, 0, time.time()
    for it in range(1, a.iters + 1):
        starts = cur.sample(a.envs)
        b, info = rollout(env, agent, starts, dev)
        idx = np.array([s["idx"] for s in starts])
        cur.update(idx, info["success"])
        adv, ret = gae(b)
        pl, vl = update(agent, opt, b, adv, ret)
        samples += int(b["mask"].sum())
        fresh = idx < 0
        ev_s, ev_f = "", ""
        if it % a.eval_every == 0 or it == a.iters:
            _, ei = rollout(env, agent, eval_starts, dev, deterministic=True)
            ev_s, ev_f = float(ei["success"].mean()), float(np.median(ei["max_f"]))
            agent.export(out / f"policy_{it:04d}.npz")
            torch.save(agent.state_dict(), out / "agent.pt")
            if ev_s >= best:
                best = ev_s
                agent.export(out / "best.npz")
                torch.save(agent.state_dict(), out / "best.pt")
        row = [it, samples, info["success"][fresh].mean(), info["success"][~fresh].mean(), cur.stats()["bank_p"],
               float((b["rew"].sum(0)).mean()), np.median(info["max_f"]), float(agent.log_std.exp().mean().detach()), pl, vl,
               ev_s, ev_f, round(time.time() - t0)]
        log.writerow([round(v, 4) if isinstance(v, float) else v for v in row])
        print(" ".join(f"{h}={v:.3f}" if isinstance(v, float) else f"{h}={v}" for h, v in
                       zip(("it", "n", "fresh", "bank", "bank_p", "ret", "maxF", "std", "pl", "vl", "eval", "evalF", "s"), row)), flush=True)
    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
