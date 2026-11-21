"""Scripted insertion baselines on the same observations as the policy: act(obs) -> (dn m, dt m, dyaw deg)."""
import numpy as np

F_JAM = 30.0          # N: a step whose peak force passes this counts as a jam
N_FLUSH = 0.005       # believed tool n at flush (front wall mid-plane)


class Straight:
    """Push straight along the believed slot normal at 8 mm/step; back off 3 mm after a force stop; stop at flush."""

    def reset(self):
        pass

    def act(self, o):
        if o["stopped"] or o["f_peak"] > 50.0:
            return np.array([-0.003, 0.0, 0.0])
        return np.array([min(0.008, max(0.0, N_FLUSH + 0.003 - o["n"])), 0.0, 0.0])


class StraightZ(Straight):
    """Straight push that first sets the height from the start-gap estimate: bottom ~6 mm above the lip (mid-opening
    for a 6.5 cm drawer in the measured 8 cm opening). Needs the v2 obs (z, gap_est)."""

    GAP = 0.006

    def act(self, o):
        dz = float(np.clip(self.GAP - (o["gap_est"] + o["z"]), -0.003, 0.003))
        a = super().act(o)
        return np.r_[a[0] if abs(dz) < 0.001 else 0.0, a[1:], dz]


class Wiggle:
    """Push at 6 mm/step; on a jam back off and move to the next point of a spiral in (t, yaw) around the start pose."""

    PATTERN = [(0, 0)] + [(s * r, s * y) for r, y in ((3, 1), (6, 2), (9, 3), (12, 4), (15, 5), (18, 6)) for s in (1, -1)]

    def __init__(self):
        self.reset()

    def reset(self):
        self.i, self.base, self.backoff = 0, None, 0

    def act(self, o):
        if self.base is None:
            self.base = np.array([o["t"], o["yaw"]])
        if o["stopped"] or o["f_peak"] > F_JAM:
            self.i = (self.i + 1) % len(self.PATTERN)
            self.backoff = 2
        dt_mm, dyaw = self.PATTERN[self.i]
        goal = self.base + [dt_mm * 1e-3, dyaw]
        d_t = float(np.clip(goal[0] - o["t"], -0.004, 0.004))
        d_y = float(np.clip(goal[1] - o["yaw"], -2.0, 2.0))
        if self.backoff > 0:
            self.backoff -= 1
            return np.array([-0.003, d_t, d_y])
        far = abs(goal[0] - o["t"]) > 0.001 or abs(goal[1] - o["yaw"]) > 0.3
        dn = 0.0 if far else min(0.006, max(0.0, N_FLUSH + 0.003 - o["n"]))
        return np.array([dn, d_t, d_y])


def preroll(kind, O, rng):
    """Vectorized scripted pre-roll for reset-to-state starts. O: dict of arrays (N,) with 'stopped', 'f_peak';
    kind 'push' (straight) or 'drift' (straight with random lateral/yaw steps). Returns (N, 3) (dn m, dt m, dyaw deg)."""
    stop = (np.asarray(O["stopped"]) > 0) | (np.asarray(O["f_peak"]) > 50.0)
    n = len(stop)
    a = np.zeros((n, 3))
    a[:, 0] = np.where(stop, -0.003, 0.008)
    if kind == "drift":
        a[:, 1] = rng.normal(0, 0.002, n)
        a[:, 2] = rng.normal(0, 1.0, n)
    return a
