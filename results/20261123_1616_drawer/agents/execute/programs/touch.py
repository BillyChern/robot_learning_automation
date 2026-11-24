"""Touch search for the opening (operator 18:30), numpy only. Each touch lowers the drawer until contact and reports the bottom
height there; a lower contact means the drawer is closer to dropping in (attempt 28: 96.5 mm at a = -6 mm, 93.9 mm at +6 mm).
Line searches along a, then b, then yaw, each from the best point so far; then a 4 mm spiral over +-15 mm so nothing is missed."""
import json
import os

import numpy as np

A_STEPS = (0.003, 0.006, 0.009, 0.012, -0.003, -0.006, -0.009)
B_STEPS = (0.002, -0.002, 0.004, -0.004, 0.006, -0.006, 0.008, -0.008)
YAW_STEPS = (1.5, -1.5, 3.0, -3.0)
SPAN, GRID = 0.015, 0.004
MIN_GAIN = 0.0003                 # m: a contact must be this much lower to count as better (touch heights repeat to ~0.3 mm)
DEFAULT_CENTRE = np.array([0.003, -0.003, 0.0])   # lowest contacts of attempt 28, relative to the plan


class TouchSearch:
    """next() -> the next point (da, db, dyaw) relative to the search centre, or None; report(point, height) after a touch."""

    def __init__(self, max_touches=36):
        self.max = max_touches
        self.best = np.zeros(3)
        self.best_h = np.inf
        self.seen = []
        self.stage = 0
        self.queue = []
        self.report(np.zeros(3), None)

    def report(self, point, height):
        if height is not None:
            self.seen.append(np.asarray(point, float).copy())
            if height < self.best_h - MIN_GAIN:
                self.best, self.best_h = np.asarray(point, float).copy(), float(height)

    def _visited(self, p):
        return any(abs(p[0] - q[0]) < 0.001 and abs(p[1] - q[1]) < 0.001 and abs(p[2] - q[2]) < 0.5 for q in self.seen)

    def _fill(self):
        b = self.best
        if self.stage == 0:
            q = [b + [x, 0, 0] for x in A_STEPS]
        elif self.stage == 1:
            q = [b + [0, y, 0] for y in B_STEPS]
        elif self.stage == 2:
            q = [b + [0, 0, w] for w in YAW_STEPS]
        elif self.stage == 3:
            v = np.arange(-SPAN, SPAN + 1e-6, GRID)
            q = sorted(([x, y, b[2]] for x in v for y in v), key=lambda p: (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2)
            q = [np.array(p) for p in q]
        else:
            q = []
        self.stage += 1
        self.queue = q

    def next(self):
        if len(self.seen) >= self.max:
            return None
        while self.stage <= 4:
            while self.queue:
                p = np.asarray(self.queue.pop(0), float)
                if not self._visited(p):
                    return p
            if self.stage == 4:
                return None
            self._fill()
        return None


def load_centre(rl_dir):
    """Running estimate of the offset (a, b, yaw) where the drawer enters, else the first working offset. Files without v = 2 held
    the pose at the END of the descent, which includes the exploration's drift (episode 5: +1.8, +1.1 mm, 0.75 deg): ignored."""
    try:
        with open(os.path.join(rl_dir, "touch_centre.json")) as f:
            d = json.load(f)
        c = np.array(d["centre"], float)
        ok = d.get("v") == 2 and c.shape == (3,) and np.all(np.abs(c) <= [0.03, 0.03, 6.0])
        return c if ok else DEFAULT_CENTRE.copy()
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_CENTRE.copy()


def save_centre(rl_dir, entry, info, gain=0.3):
    """Move the estimate toward the offset at which the drawer just entered the opening."""
    p = os.path.join(rl_dir, "touch_centre.json")
    old = load_centre(rl_dir)
    new = (1.0 - gain) * old + gain * np.asarray(entry, float)
    with open(p + ".tmp", "w") as f:
        json.dump(dict(v=2, centre=np.round(new, 5).tolist(), entry=np.round(entry, 5).tolist(), **info), f)
    os.replace(p + ".tmp", p)
    return new
