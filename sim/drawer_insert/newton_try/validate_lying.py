"""Validate newton_lying.NewtonLying against lying_env.LyingEnv: same params, Straight descent; plus throughput.

PYTHONPATH=. python sim/drawer_insert/newton_try/validate_lying.py N [variant]"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from sim.drawer_insert import lying_params as P
from sim.drawer_insert.lying_policy import H_STEPS, action_of

SEED0 = 100000


def cpu_episode(args):
    seed, variant = args
    os.nice(15)
    from sim.drawer_insert.lying_baselines import Straight
    from sim.drawer_insert.lying_env import LyingEnv
    env = LyingEnv(seed)
    o = env.reset(P.sample(np.random.default_rng(seed), variant))
    c = Straight()
    while True:
        o, r, done, info = env.step(action_of(c.act(o)))
        if done:
            return dict(success=bool(info["success"]), depth=info["depth"], max_f=info["max_f"], n_stops=info["n_stops"],
                        top_err=info["top_err"], e_x=info["e_x"], e_y=info["e_y"])


def straight(env):
    """lying_baselines.Straight vectorized on the env's state."""
    a = np.zeros((env.N, 4))
    stop = (env.stopped > 0) | (env.f_peak > 50.0)
    seated = (env.cmd[:, 3] < 0.01) & (env.f_last[:, 2] > 10.0)
    a[:, 3] = np.where(stop, 0.002, np.where(seated, 0.0, -np.minimum(0.006, np.maximum(0.0, env.cmd[:, 3] + 0.004))))
    return action_of(a)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    variant = sys.argv[2] if len(sys.argv) > 2 else "carry"
    from sim.drawer_insert.newton_lying import NewtonLying
    seeds = [SEED0 + i for i in range(n)]
    t0 = time.time()
    env = NewtonLying(n)
    t_build = time.time() - t0
    t0 = time.time()
    env.reset([dict(p=P.sample(np.random.default_rng(s), variant)) for s in seeds])
    t_reset = time.time() - t0
    t0 = time.time()
    for k in range(H_STEPS):
        o, r, done, info = env.step(straight(env))
    t_steps = time.time() - t0
    with mp.get_context("spawn").Pool(10) as pool:
        cpu = pool.map(cpu_episode, [(s, variant) for s in seeds])
    cs, gs = np.array([c["success"] for c in cpu]), info["success"]
    cd, gd = np.array([c["depth"] for c in cpu]), info["depth"]
    cf, gf = np.array([c["max_f"] for c in cpu]), info["max_f"]
    out = dict(n=n, variant=variant, build_s=round(t_build, 1), reset_s=round(t_reset, 1), steps_s=round(t_steps, 1),
               decisions_per_s=round(n * H_STEPS / t_steps), success_cpu=float(cs.mean()), success_newton=float(gs.mean()),
               agree=float((cs == gs).mean()), depth_cpu_med=float(np.median(cd)), depth_newton_med=float(np.median(gd)),
               depth_absdiff_med_mm=float(np.median(np.abs(cd - gd)) * 1e3), maxf_cpu_med=float(np.median(cf)),
               maxf_newton_med=float(np.median(gf)), overflow=env.overflow,
               e_xy_absdiff_max_mm=float(np.max(np.abs(np.c_[[c["e_x"] for c in cpu], [c["e_y"] for c in cpu]] - np.c_[info["e_x"], info["e_y"]])) * 1e3))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
