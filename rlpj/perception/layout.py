"""Reset targets from the task yaml's layout section, blob-to-object assignment by size priors, layout checks."""
import itertools

import numpy as np


def sample(task_cfg, rng=None, tries=2000):
    """{object: [x, y]} (base frame, m) satisfying task_cfg['layout']: per-object x/y ranges or from/distance/angle
    placement, min_gap between centres and radius from the robot axis; rejection sampling."""
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    lay = task_cfg["layout"]
    names = _order(lay, [o for o in task_cfg["objects"] if o in lay])
    for _ in range(tries):
        xy = {}
        for o in names:
            s = lay[o]
            if "from" in s:
                d, a = rng.uniform(*s["distance"]), np.radians(rng.uniform(*s["angle"]))
                xy[o] = np.asarray(xy[s["from"]]) + d * np.array([np.cos(a), np.sin(a)])
            else:
                xy[o] = np.array([rng.uniform(*s["x"]), rng.uniform(*s["y"])])
        if valid(lay, xy):
            return {o: [round(float(v[0]), 3), round(float(v[1]), 3)] for o, v in xy.items()}
    raise ValueError("layout: no sample satisfied min_gap/radius after %d tries" % tries)


def valid(lay, xy):
    """min_gap and radius constraints of a layout for the positions xy = {object: [x, y]}."""
    pts = [np.asarray(v, float) for v in xy.values()]
    gap = float(lay.get("min_gap", 0.0))
    if any(np.linalg.norm(p - q) < gap for p, q in itertools.combinations(pts, 2)):
        return False
    if "radius" in lay:
        lo, hi = lay["radius"]
        if any(not (lo <= np.hypot(*p) <= hi) for p in pts):
            return False
    return True


def matches(prior, blob):
    """True when the blob lies inside every range of a size prior ({height|extent|outer_r|inner_r|z_top: [lo, hi],
    hollow: bool}); keys with other values are ignored."""
    for key, rng in prior.items():
        if key == "hollow":
            if bool(blob.get("hollow")) != bool(rng):
                return False
        elif isinstance(rng, (list, tuple)) and len(rng) == 2:
            v = measure(blob, key)
            if v is None or not (rng[0] <= v <= rng[1]):
                return False
    return True


def measure(blob, key):
    """Scalar blob measurement for a prior key (extent = the larger side)."""
    if key == "extent":
        return None if blob.get("extent") is None else float(max(blob["extent"]))
    v = blob.get(key)
    return None if v is None else float(v)


def assign(task_cfg, blobs, targets=None):
    """{object: blob|None}: the injective blob assignment that matches the most size priors (task_cfg['perception'])
    and then best fits the layout relations (from/angle/distance) and, when given, the targets."""
    priors = task_cfg.get("perception", {}) or {}
    lay = task_cfg.get("layout", {}) or {}
    names = list(task_cfg["objects"])
    cands = {o: [i for i, b in enumerate(blobs) if matches(priors.get(o, {}) or {}, b)] + [None] for o in names}
    best, best_score = {o: None for o in names}, -np.inf
    for combo in itertools.product(*(cands[o] for o in names)):
        used = [i for i in combo if i is not None]
        if len(used) != len(set(used)):
            continue
        pick = {o: (None if i is None else blobs[i]) for o, i in zip(names, combo)}
        score = 100 * len(used) - _penalty(lay, pick, targets)
        if score > best_score:
            best, best_score = pick, score
    return best


def check(task_cfg, targets, blobs):
    """(ok, report): every target object is found within layout.tolerance of its target."""
    tol = float(task_cfg["layout"].get("tolerance", 0.02))
    pick = assign(task_cfg, blobs, targets)
    rep = dict(tolerance=tol, objects={}, extra_blobs=len(blobs) - sum(b is not None for b in pick.values()))
    for o, t in targets.items():
        b = pick.get(o)
        d = None if b is None else float(np.linalg.norm(np.subtract(b["xy"], t)))
        rep["objects"][o] = dict(target=[round(float(v), 3) for v in t], found=None if b is None else list(b["xy"]),
                                 dist=None if d is None else round(d, 4), ok=bool(d is not None and d <= tol))
    bad = [o for o, v in rep["objects"].items() if not v["ok"]]
    rep["reason"] = "layout ok" if not bad else "; ".join(
        f"{o}: " + ("not found" if rep["objects"][o]["dist"] is None else f"{rep['objects'][o]['dist'] * 100:.1f} cm off")
        for o in bad)
    return not bad, rep


def _order(lay, names):
    """Objects placed independently first, then the ones placed relative to another."""
    return [o for o in names if "from" not in lay[o]] + [o for o in names if "from" in lay[o]]


def _penalty(lay, pick, targets):
    tol = float(lay.get("tolerance", 0.02))
    pen = 0.0
    for o, b in pick.items():
        if b is None:
            continue
        if targets and o in targets:
            pen += float(np.linalg.norm(np.subtract(b["xy"], targets[o]))) / tol
        s = lay.get(o) or {}
        ref = pick.get(s.get("from")) if "from" in s else None
        if ref is not None:
            v = np.subtract(b["xy"], ref["xy"])
            pen += _range_dev(np.degrees(np.arctan2(v[1], v[0])), s["angle"], wrap=360) / 45.0
            pen += _range_dev(float(np.hypot(*v)), s["distance"]) / 0.05
    return pen


def _range_dev(v, rng, wrap=None):
    """Distance from v to the interval rng (circular when wrap is given)."""
    lo, hi = rng
    if wrap:
        v = lo + (v - lo) % wrap
        return 0.0 if v <= hi else min(v - hi, lo + wrap - v)
    return max(lo - v, v - hi, 0.0)
