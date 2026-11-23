"""CPU pool of DrawerEnvs in worker processes; synchronized batches: reset(starts) -> obs, step(actions) until all done."""
import multiprocessing as mp
import os

import numpy as np

INFO_KEYS = ("success", "front_err", "top_err", "f_end", "max_f", "n_stops", "e_t", "e_x", "e_y", "e_yaw", "depth")


def worker(conn, seed, m, f_pen, task="slot"):
    os.nice(15)
    if task == "lying":
        from . import lying_env as E
        from .lying_policy import obs_vector
        Env = E.LyingEnv
    else:
        from . import env as E
        from .policy import obs_vector
        Env = E.DrawerEnv
    E.F_PEN = f_pen
    envs = [Env(seed * 1000 + i) for i in range(m)]
    done = np.ones(m, bool)
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            obs = np.stack([obs_vector(e.reset(s["p"], s.get("kind"), s.get("k_pre", 0))) for e, s in zip(envs, arg)])
            dim = obs.shape[1]
            done[:] = False
            conn.send(obs)
        elif cmd == "step":
            obs, rew = np.zeros((m, dim), np.float32), np.zeros(m)
            info = {k: np.zeros(m) for k in INFO_KEYS}
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                o, rew[i], done[i], inf = e.step(arg[i])
                obs[i] = obs_vector(o)
                for k in INFO_KEYS:
                    info[k][i] = inf.get(k, 0.0)
            conn.send((obs, rew, done.copy(), info))
        else:
            break


class CpuPool:
    """num_workers processes x per_worker envs (MuJoCo CPU)."""

    def __init__(self, num_workers=12, per_worker=32, seed=0, f_pen=0.05, task="slot"):
        ctx = mp.get_context("spawn")
        self.m, self.num_envs = per_worker, num_workers * per_worker
        self.conns, self.procs = [], []
        for w in range(num_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, seed * 100 + w, per_worker, f_pen, task), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)

    def reset(self, starts):
        for w, c in enumerate(self.conns):
            c.send(("reset", starts[w * self.m:(w + 1) * self.m]))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        for w, c in enumerate(self.conns):
            c.send(("step", actions[w * self.m:(w + 1) * self.m]))
        out = [c.recv() for c in self.conns]
        info = {k: np.concatenate([o[3][k] for o in out]) for k in INFO_KEYS}
        return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                np.concatenate([o[2] for o in out]), info)

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
