"""Shared by execute.py (numpy only) and learner.py: observation/action layout, zero-residual prior, numpy actor."""
import json
import os
import time

import numpy as np

OBS_DIM = 13
# 0 phase (0 align, 1 descend), 1 inserted fraction, 2-4 applied corrections (a, b, yaw), 5-7 force change (a, b, z),
# 8 last peak force, 9 step fraction, 10-11 pinch point on the drawer (s, t), 12 contacts so far
ZERO_OBS = ()
ACT_DIM = 4                                    # corrections along a, b, yaw; descent step (scaled in execute.py)
PRIOR = np.zeros(ACT_DIM)                      # the scripted prior: straight down the measured opening centre
PRIOR_VERSION = 2                              # 2: residuals on the measured alignment, search points on contact (data of v1 mean something else)
MIN_ACTOR_VERSION = 2
INIT_STD = 0.30                                # pre-tanh std of the initial policy
LOG_STD_MIN, LOG_STD_MAX = -2.3, 0.0           # min std 0.1: most logged actions are exactly 0, and the fit collapsed the std to 0.018 (no exploration)
EXPLORE_MAX = 0.5


def prior(obs):
    o = np.asarray(obs, float)
    return np.zeros(ACT_DIM) if o.ndim == 1 else np.zeros((len(o), ACT_DIM))


def rl_dir():
    d = os.environ.get("RLPJ_RL_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
    os.makedirs(d, exist_ok=True)
    return d


def atanh(x):
    x = np.clip(x, -0.999, 0.999)
    return 0.5 * np.log((1 + x) / (1 - x))


class Actor:
    """numpy forward pass of the learner's actor (actor.npz: W0,b0,W1,b1,Wmu,bmu,Wls,bls); the prior when missing."""

    def __init__(self, path=None):
        self.p, self.info = None, dict(source="prior")
        path = path or os.path.join(rl_dir(), "policy", "actor.npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                p = {k: z[k] for k in z.files}
                if int(p.get("prior_version", np.array(0))) < MIN_ACTOR_VERSION or p["W0"].shape[1] != OBS_DIM:
                    self.info = dict(source="prior", note=f"{path} is from another version")
                else:
                    self.p = p
                    self.info = dict(source=path, updates=int(p.get("updates", np.array(0))), mtime=time.ctime(os.path.getmtime(path)))
            except Exception as e:
                self.info = dict(source="prior", error=str(e)[:100])

    def dist(self, obs):
        o = np.asarray(obs, float)
        if self.p is None:
            return atanh(prior(o)), np.full(ACT_DIM, np.log(INIT_STD))
        h = np.maximum(self.p["W0"] @ o + self.p["b0"], 0)
        h = np.maximum(self.p["W1"] @ h + self.p["b1"], 0)
        mu = self.p["Wmu"] @ h + self.p["bmu"]
        ls = np.clip(self.p["Wls"] @ h + self.p["bls"], LOG_STD_MIN, LOG_STD_MAX)
        return mu, ls

    def act(self, obs, explore, rng):
        mu, ls = self.dist(obs)
        std = np.minimum(np.exp(ls), EXPLORE_MAX)
        u = mu + (std * rng.standard_normal(ACT_DIM) if explore else 0.0)
        return np.tanh(u), dict(mu=np.round(np.tanh(mu), 3).tolist(), std=np.round(np.exp(ls), 3).tolist())


def save_episode(name, trans, meta):
    """transitions/<name>.npz (obs, act, rew, next_obs, done) + <name>.json (meta)."""
    d = os.path.join(rl_dir(), "transitions")
    os.makedirs(d, exist_ok=True)
    if trans:
        arr = {k: np.array([t[k] for t in trans], np.float32) for k in ("obs", "act", "rew", "next_obs", "done")}
    else:
        arr = dict(obs=np.zeros((0, OBS_DIM), np.float32), act=np.zeros((0, ACT_DIM), np.float32), rew=np.zeros(0, np.float32),
                   next_obs=np.zeros((0, OBS_DIM), np.float32), done=np.zeros(0, np.float32))
    p = os.path.join(d, name + ".npz")
    np.savez_compressed(p + ".tmp.npz", **arr)
    os.replace(p + ".tmp.npz", p)
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f, default=lambda o: np.round(o, 4).tolist() if isinstance(o, np.ndarray) else str(o))
    return p
