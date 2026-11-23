"""Scripted vertical-insertion baselines on the policy's observations: act(obs) -> (dx m, dy m, dyaw deg, dz m)."""
import numpy as np

from .lying_params import L_EST

F_SEAT = 10.0                  # N: upward force that means "seated on the floor" near the believed flush height


class Straight:
    """Descend straight at 6 mm/step from the nominal pose; back up 2 mm after a force stop or a 50 N peak; stop once
    the upward force passes 10 N within 1 cm of the believed flush height."""

    def reset(self):
        pass

    def act(self, o):
        if o["stopped"] or o["f_peak"] > 50.0:
            return np.array([0.0, 0.0, 0.0, 0.002])
        if o["z"] < 0.01 and o["f"][2] > F_SEAT:
            return np.zeros(4)
        return np.array([0.0, 0.0, 0.0, -min(0.006, max(0.0, o["z"] + 0.004))])


class Spiral:
    """Descend at 6 mm/step; when the drawer jams on the rim (force stop or a 50 N peak while its bottom is still above
    the believed rim), back up 4 mm, move to the next point of a spiral in (x, y) around the start (2.5 mm
    pitch, alternating +-1 deg yaw) and descend again; once the bottom is past the rim it only descends."""

    PATTERN = [(0, 0, 0)] + [(r * np.cos(a), r * np.sin(a), y) for r, y in ((2.5, 1), (5, -1), (7.5, 1), (10, -1), (12.5, 0))
                             for a in np.arange(8) * np.pi / 4]
    ENTERED = L_EST - 0.010    # believed z below which the bottom is ~1 cm into the opening

    def __init__(self):
        self.reset()

    def reset(self):
        self.i, self.base, self.backoff = 0, None, 0

    def act(self, o):
        if self.base is None:
            self.base = np.array([o["x"], o["y"], o["yaw"]])
        jam = o["stopped"] or o["f_peak"] > 50.0
        if jam and o["z"] > self.ENTERED and self.backoff == 0:
            self.i = (self.i + 1) % len(self.PATTERN)
            self.backoff = 1
        if o["z"] < 0.01 and o["f"][2] > F_SEAT:
            return np.zeros(4)
        if o["stopped"] or o["f_peak"] > 50.0:
            return np.array([0.0, 0.0, 0.0, 0.002])
        gx, gy, gyaw = self.base + np.array(self.PATTERN[self.i]) * [1e-3, 1e-3, 1.0]
        d = np.array([np.clip(gx - o["x"], -0.004, 0.004), np.clip(gy - o["y"], -0.004, 0.004), np.clip(gyaw - o["yaw"], -2, 2)])
        if self.backoff > 0:
            self.backoff -= 1
            return np.r_[0.0, 0.0, 0.0, 0.004]
        far = abs(gx - o["x"]) > 0.001 or abs(gy - o["y"]) > 0.001 or abs(gyaw - o["yaw"]) > 0.3
        if o["z"] < self.ENTERED:
            d[:], far = 0.0, False
        return np.r_[d, 0.0 if far else -min(0.006, max(0.0, o["z"] + 0.004))]


def preroll(kind, O, rng):
    """Vectorized scripted pre-roll for reset-to-state starts. O: dict of arrays (N,) 'stopped', 'f_peak'; kind
    'descend' (straight) or 'drift' (descent with random lateral/yaw steps). Returns (N, 4) (dx, dy, dyaw, dz)."""
    stop = (np.asarray(O["stopped"]) > 0) | (np.asarray(O["f_peak"]) > 50.0)
    n = len(stop)
    a = np.zeros((n, 4))
    a[:, 3] = np.where(stop, 0.002, -0.006)
    if kind == "drift":
        a[:, :2] = rng.normal(0, 0.0015, (n, 2))
        a[:, 2] = rng.normal(0, 0.8, n)
    return a
