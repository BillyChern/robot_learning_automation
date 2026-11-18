"""Shared by execute.py (numpy only) and learner.py: observation/action layout, prior, numpy actor."""
import json
import os
import time

import numpy as np

OBS_DIM = 13
ACT_DIM = 4
PRIOR = np.array([0.6, 0.0, 0.0, 0.0])        # 12 mm straight along the nominal line
PRIOR_VERSION = 3                              # must match learner.PRIOR_VERSION
INIT_STD = 0.25                                # pre-tanh std of the initial policy
LOG_STD_MIN, LOG_STD_MAX = -4.0, 0.5


def prior(obs):
    """Scripted prior action for (batched) observations. Insert stage: 12 mm advance plus half of the remaining
    measured lateral (t) and yaw error per step (held-drawer side-depth measurement in obs[5], obs[6] minus the
    corrections already applied, obs[2], obs[4]). Push stage: 12 mm straight. The policy learns residuals on top
    (actor initialised to / regularised toward this prior)."""
    o = np.asarray(obs, float)
    single = o.ndim == 1
    o = np.atleast_2d(o)
    a = np.tile(PRIOR, (len(o), 1))
    ins = o[:, 0] < 0.5
    e_t = (o[:, 5] + o[:, 2]) * 0.01            # m: drawer t error still to correct
    e_y = (o[:, 6] + o[:, 4]) * 5.0             # deg
    # v3: no t correction in the prior: the side-depth t of the held drawer is biased by the partial view (attempt
    # 17: t err -0.024 -> moved 2.4 cm sideways -> jams); the policy may still learn to use obs[5]
    a[ins, 3] = np.clip(-0.5 * e_y[ins] / 2.0, -0.8, 0.8)
    return a[0] if single else a


def rl_dir():
    d = os.environ.get("RLPJ_RL_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
    os.makedirs(d, exist_ok=True)
    return d


def atanh(x):
    x = np.clip(x, -0.999, 0.999)
    return 0.5 * np.log((1 + x) / (1 - x))


class Actor:
    """numpy forward pass of the learner's actor (actor.npz: W0,b0,W1,b1,Wmu,bmu,Wls,bls); prior when missing."""

    def __init__(self, path=None):
        self.p, self.info = None, dict(source="prior")
        path = path or os.path.join(rl_dir(), "policy", "actor.npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                p = {k: z[k] for k in z.files}
                if int(p.get("prior_version", np.array(1))) < PRIOR_VERSION:
                    self.info = dict(source="prior", note=f"{path} predates prior version {PRIOR_VERSION}")
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
        u = mu + (np.exp(ls) * rng.standard_normal(ACT_DIM) if explore else 0.0)
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
