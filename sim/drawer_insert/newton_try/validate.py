"""Validate NewtonEnv against env.DrawerEnv: same params, Straight push, success/depth/force; plus throughput.

python newton_try/validate.py N [cpu]   (run from sim/drawer_insert with PYTHONPATH=repo root)"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.drawer_insert import params as P
from sim.drawer_insert.baselines import N_FLUSH
from sim.drawer_insert.env import action_of
from sim.drawer_insert.policy import H_STEPS

SEED0 = 100000


def cpu_episode(seed):
    os.nice(15)
    from sim.drawer_insert.baselines import Straight
    from sim.drawer_insert.env import DrawerEnv
    env = DrawerEnv(seed)
    o = env.reset(P.sample(np.random.default_rng(seed)))
    c = Straight()
    while True:
        o, r, done, info = env.step(action_of(c.act(o)))
        if done:
            return dict(success=bool(info["success"]), depth=info["depth"], max_f=info["max_f"], n_stops=info["n_stops"],
                        front_err=info["front_err"], f_end=info["f_end"], e_t=info["e_t"], e_yaw=info["e_yaw"])


def straight(env):
    """baselines.Straight vectorized on the env's state."""
    a = np.zeros((env.N, 3))
    stop = (env.stopped > 0) | (env.f_peak > 50.0)
    a[:, 0] = np.where(stop, -0.003, np.minimum(0.008, np.maximum(0.0, N_FLUSH + 0.003 - env.cmd[:, 0])))
    return action_of(a)


def gpu_run(n, seeds):
    from sim.drawer_insert.newton_env import NewtonEnv
    t0 = time.time()
    env = NewtonEnv(n)
    t_build = time.time() - t0
    starts = [dict(p=P.sample(np.random.default_rng(s))) for s in seeds]
    if "warm" in sys.argv:                 # a different episode first: resets must fully replace the parameters
        env.reset([dict(p=P.sample(np.random.default_rng(s + 7777)), kind="drift", k_pre=5) for s in seeds])
        for _ in range(20):
            env.step(np.random.default_rng(0).uniform(-1, 1, (n, 3)))
    t0 = time.time()
    env.reset(starts)
    t_reset = time.time() - t0
    t0 = time.time()
    for k in range(H_STEPS):
        o, r, done, info = env.step(straight(env))
    t_steps = time.time() - t0
    res = [dict(success=bool(info["success"][i]), depth=float(info["depth"][i]), max_f=float(info["max_f"][i]),
                n_stops=int(info["n_stops"][i]), front_err=float(info["front_err"][i]), f_end=float(info["f_end"][i]),
                e_t=float(info["e_t"][i]), e_yaw=float(info["e_yaw"][i])) for i in range(n)]
    return res, dict(build_s=t_build, reset_s=t_reset, steps_s=t_steps, dec_per_s=n * H_STEPS / (t_steps + t_reset))


def stats(res, tag):
    a = lambda k: np.array([r[k] for r in res], float)
    d, f = a("depth"), a("max_f")
    print(f"{tag}: success {a('success').mean():.3f}, depth mean {d.mean() * 1e3:.1f} mm, "
          f"depth hist {np.histogram(d * 1e3, bins=[-60, 0, 30, 60, 90, 110, 140])[0].tolist()}, "
          f"max_f median {np.median(f):.1f} N, hist {np.histogram(f, bins=[0, 10, 20, 40, 60, 80, 1e4])[0].tolist()}, "
          f"stops mean {a('n_stops').mean():.2f}")


if __name__ == "__main__":
    n = int(sys.argv[1])
    seeds = list(range(SEED0, SEED0 + n))
    res, perf = gpu_run(n, seeds)
    stats(res, "newton")
    print("perf", {k: round(v, 2) for k, v in perf.items()})
    if len(sys.argv) > 2 and sys.argv[2] == "cpu":
        with mp.Pool(8) as pool:
            cpu = pool.map(cpu_episode, seeds)
        stats(cpu, "cpu   ")
        sg, sc = np.array([r["success"] for r in res]), np.array([r["success"] for r in cpu])
        dg, dc = np.array([r["depth"] for r in res]), np.array([r["depth"] for r in cpu])
        print(f"agreement success {np.mean(sg == sc):.3f}, |depth diff| median {np.median(np.abs(dg - dc)) * 1e3:.1f} mm, "
              f"corr {np.corrcoef(dg, dc)[0, 1]:.3f}")
        out = Path(__file__).resolve().parents[1] / "runs" / "newton_validate.json"
        json.dump(dict(newton=res, cpu=cpu, perf=perf), open(out, "w"))
        print("saved", out)
