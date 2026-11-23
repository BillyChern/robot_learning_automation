"""Offline checks of the lying-cabinet sim, its numpy policy export and the robot-side Descent bookkeeping."""
import numpy as np

from sim.drawer_insert import lying_params as P
from sim.drawer_insert.lying_baselines import Straight
from sim.drawer_insert.lying_env import LyingEnv
from sim.drawer_insert.lying_policy import (OBS_DIM, Descent, Policy, action_of, obs_vector, rot, start_pose,
                                            step_of)


def straight(p, steps=30):
    env, b = LyingEnv(0), Straight()
    o = env.reset(p)
    for _ in range(steps):
        o, r, done, info = env.step(action_of(b.act(o)))
    return env, info


def test_aligned_straight_descent_goes_in():
    env, info = straight(P.nominal())
    assert info["success"] and info["top_err"] < 0.003, info


def test_offset_drawer_lands_on_the_rim():
    p = P.nominal(); p.update(open_dx=0.008)
    env, info = straight(p, 20)
    assert not info["success"] and env.max_f > 40.0 and info["depth"] < 0.02


def test_obs_and_actions():
    o = LyingEnv(1).reset()
    assert obs_vector(o).shape == (OBS_DIM,)
    assert np.allclose(step_of(action_of([0.002, -0.001, 1.0, -0.005])), [0.002, -0.001, 1.0, -0.005])


def test_policy_export_matches_torch(tmp_path):
    import torch
    from sim.drawer_insert.train import Agent
    agent = Agent(task="lying")
    agent.export(tmp_path / "p.npz")
    o = LyingEnv(2).reset()
    with torch.no_grad():
        mu = agent.dist(torch.as_tensor(obs_vector(o))[None]).mean[0].numpy()
    assert np.allclose(Policy(tmp_path / "p.npz").act(o), step_of(mu), atol=1e-5)


def test_descent_matches_env(tmp_path):
    """Base-frame bookkeeping (robot side) reproduces the env's commands and observations."""
    from sim.drawer_insert.train import Agent
    torch_agent = Agent(task="lying")
    torch_agent.export(tmp_path / "p.npz")
    pol = Policy(tmp_path / "p.npz")
    c, axis, table, ys, f0 = np.array([0.4469, -0.011]), 21.2, -0.016, 110.0, np.array([1.0, -2.0, 3.0])
    env = LyingEnv(3)
    p = P.sample(np.random.default_rng(3))
    o = env.reset(p)
    est = dict(off=o["off"], yaw=o["yaw_est"], tilt=o["tilt"], hold=p["est_hold"])
    base = lambda cmd: (np.r_[c + rot(axis) @ cmd[:2], cmd[3] + table + P.RIM_H + p["est_hold"]], cmd[2] + ys)
    xyz, yaw = base(env.cmd)
    sxyz, syaw = start_pose(c, axis, table + P.RIM_H, ys, est, P.L_EST)
    assert np.allclose(sxyz[:2], xyz[:2], atol=1e-6) and abs(syaw - yaw) < 1e-6
    ins = Descent(pol, c, axis, table + P.RIM_H, ys, est, xyz, yaw, f0)
    for _ in range(12):
        ins.next(f0 + np.r_[rot(axis) @ o["f"][:2], o["f"][2]])
        assert np.allclose(obs_vector(ins.obs), obs_vector(o), atol=1e-5)
        o, r, done, info = env.step(action_of(pol.act(o)))
        ins.done(o["f_peak"], o["stopped"], *base(env.cmd))
        assert np.allclose(ins.cmd, env.cmd, atol=1e-6)


def test_guided_steps_along_the_lateral_force():
    from sim.drawer_insert.lying_baselines import Guided
    o = dict(x=0.0, y=0.0, yaw=0.0, z=0.09, f=np.array([-8.0, 3.0, 25.0]), f_peak=30.0, stopped=0.0)
    d = Guided().act(o)
    assert d[0] < 0 < d[1] and d[3] > 0
