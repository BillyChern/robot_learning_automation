"""Offline checks of the drawer sim, the numpy policy export and the baselines (sim venv)."""
import numpy as np

from sim.drawer_insert import params as P
from sim.drawer_insert.env import DrawerEnv, action_of
from sim.drawer_insert.policy import OBS_DIM, Policy, obs_vector, step_of


def test_aligned_straight_push_goes_in():
    env = DrawerEnv(0)
    o = env.reset(P.nominal())
    for _ in range(30):
        o, r, done, info = env.step(action_of([min(0.008, max(0.0, 0.008 - o["n"])), 0, 0]))
    assert info["success"], info


def test_offset_drawer_jams():
    env = DrawerEnv(0)
    p = P.nominal(); p.update(hand_off=0.015)
    o = env.reset(p)
    for _ in range(20):
        o, r, done, info = env.step(action_of([0.008, 0, 0]))
    assert not info["success"] and env.max_f > 40.0


def test_obs_and_actions():
    env = DrawerEnv(1)
    o = env.reset()
    assert obs_vector(o).shape == (OBS_DIM,)
    assert np.allclose(step_of(action_of([0.004, -0.002, 1.0])), [0.004, -0.002, 1.0])


def test_policy_export_matches_torch(tmp_path):
    import torch
    from sim.drawer_insert.train import Agent
    agent = Agent()
    agent.export(tmp_path / "p.npz")
    o = DrawerEnv(2).reset()
    with torch.no_grad():
        mu = agent.dist(torch.as_tensor(obs_vector(o))[None]).mean[0].numpy()
    assert np.allclose(Policy(tmp_path / "p.npz").act(o), step_of(mu), atol=1e-5)


def test_insertion_helper_matches_env(tmp_path):
    """Base-frame bookkeeping (robot side) reproduces the env's believed-frame commands."""
    from sim.drawer_insert.policy import Insertion
    from sim.drawer_insert.train import Agent
    Agent().export(tmp_path / "p.npz")
    pol = Policy(tmp_path / "p.npz")
    N, off, tc, ys = np.array([0.9282, 0.3722]), 0.5283, -0.0585, 21.85
    N = N / np.linalg.norm(N)
    T = np.array([-N[1], N[0]])
    env = DrawerEnv(3)
    o = env.reset()
    base = lambda c: ((c[0] + off) * N + (c[1] + tc) * T, c[2] + ys)
    xy, yaw = base(env.cmd)
    ins = Insertion(pol, N, off, tc, ys, o["off_est"], o["yaw_est"], xy, yaw, f0=np.zeros(3))
    for _ in range(10):
        fb = np.r_[o["f"][0] * N + o["f"][1] * T, o["f"][2]]
        txy, tyaw = ins.next(fb)
        a = pol.act(o)
        o, r, done, info = env.step(action_of(a))
        mxy, myaw = base(env.cmd)
        ins.done(o["f_peak"], o["stopped"], mxy, myaw)
        assert np.allclose(ins.cmd, env.cmd, atol=1e-6) and np.allclose(ins.prev, o["prev"], atol=1e-9)


def test_v2_insertion_helper_matches_env(tmp_path):
    """v2 (dz action, jam count): base-frame bookkeeping reproduces the env's commands and obs."""
    from sim.drawer_insert.policy import Insertion
    from sim.drawer_insert.train import Agent
    Agent(v2=True).export(tmp_path / "p2.npz")
    pol = Policy(tmp_path / "p2.npz")
    assert pol.v2
    N, off, tc, ys, z0 = np.array([0.9282, 0.3722]), 0.5283, -0.0585, 21.85, 0.09
    N = N / np.linalg.norm(N)
    T = np.array([-N[1], N[0]])
    env = DrawerEnv(4)
    o = env.reset(P.sample(np.random.default_rng(4), "carry", "wedge"))
    base = lambda c: ((c[0] + off) * N + (c[1] + tc) * T, c[2] + ys)
    xy, yaw = base(env.cmd)
    ins = Insertion(pol, N, off, tc, ys, o["off_est"], o["yaw_est"], xy, yaw, f0=np.zeros(3), tool_z=z0, gap_est=o["gap_est"])
    for _ in range(12):
        ins.next(np.r_[o["f"][0] * N + o["f"][1] * T, o["f"][2]])
        assert np.allclose(obs_vector(ins.obs), obs_vector(o), atol=1e-6)
        o, r, done, info = env.step(action_of(pol.act(o)))
        mxy, myaw = base(env.cmd)
        ins.done(o["f_peak"], o["stopped"], mxy, myaw, z0 + env.cmd[3])
        assert np.allclose(np.r_[ins.cmd, ins.z], env.cmd, atol=1e-6)
