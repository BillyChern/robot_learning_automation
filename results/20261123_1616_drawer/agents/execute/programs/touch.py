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
            elif height > self.best_h + MIN_GAIN and self.stage in (1, 2):   # higher contact: this side of the line leads away
                k = self.stage - 1
                side = np.sign(point[k] - self.best[k])
                self.queue = [q for q in self.queue if np.sign(q[k] - self.best[k]) != side]

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


SEED_ENTRIES = [[0.003, -0.003, 0.0], [0.0029, -0.0032, -0.0134], [0.0029, -0.0031, 0.0088], [0.00295, -0.00308, 0.0562]]   # first-descent entries: attempts 30, 32, 34, 40
KEEP = 9                          # entries in the median


def _entries(rl_dir):
    try:
        with open(os.path.join(rl_dir, "touch_centre.json")) as f:
            d = json.load(f)
        e = np.array(d["entries"], float) if d.get("v") == 4 else None
        if e is not None and e.ndim == 2 and e.shape[1] == 3 and len(e) and np.all(np.abs(e) <= [0.03, 0.03, 6.0]):
            return e
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return np.array(SEED_ENTRIES, float)


def densest(entries, radius=0.0015):
    """Centre of the largest group of entries within radius in a and b (latest group on a tie). The entries form separate clusters
    (a +3, +6, +9.8 mm), so a mean or an even-count median can land between them, where the drawer never entered (episode 11: +4.5)."""
    e = np.asarray(entries, float)
    near = np.all(np.abs(e[:, None, :2] - e[None, :, :2]) < radius, axis=2)
    k = len(e) - 1 - int(np.argmax(near.sum(1)[::-1]))
    return np.median(e[near[k]], axis=0)


def frozen_centre():
    """The start offset of scripted r4, fixed for all arms (operator 19:49): if it followed the entries, the scripted baseline would
    inherit what the learned arm finds and the curves would not measure learning."""
    return densest(np.array(SEED_ENTRIES, float))


def load_centre(rl_dir):
    """The offset (a, b, yaw) to start from: the densest cluster of the recent entries."""
    return densest(_entries(rl_dir)[-KEEP:])


def save_centre(rl_dir, entry, info):
    """Add an offset at which the drawer entered on its FIRST descent; returns the new centre. Entries after a rim touch must not be
    added: they sit ~3 mm past wherever the search started (start +3 -> entry +6, start +6.2 -> +9.2), so they ratchet the start."""
    p = os.path.join(rl_dir, "touch_centre.json")
    e = np.vstack([_entries(rl_dir), np.asarray(entry, float)])[-KEEP:]
    new = densest(e)
    with open(p + ".tmp", "w") as f:
        json.dump(dict(v=4, centre=np.round(new, 5).tolist(), entries=np.round(e, 5).tolist(), **info), f)
    os.replace(p + ".tmp", p)
    return new
