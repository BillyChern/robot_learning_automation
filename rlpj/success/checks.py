"""Geometric success checks on before/after tabletop blobs, independent of the program's own verification."""
import numpy as np

from rlpj.perception import layout, tabletop

MOVE_TOL = 0.03      # m: the tape roll may not move more than this
SPOT_TOL = 0.04      # m: a blob this close to a cup's start is "at its spot"
EMPTY_TOL = 0.05     # m: a spot counts as empty when no blob lies within this
BOWL_MOVE_MAX = 0.06 # m: the bowl may be nudged this far, not more
RISE = 0.015         # m: inside the bowl, points this far above the empty bowl's surface are new (the bee)
RISEN_CELLS = 20     # 5 mm cells of risen points that count as the bee inside (real bee: ~80, empty bowl noise: <10)
MIN_BEE_POINTS = 300 # a stray bee outside the bowl shows >1000 side-depth points; smaller blobs are depth noise


def run(task_cfg, before_blobs, after_blobs):
    """(ok: True|False|None, report) for task_cfg['name']; None (unknown) when the scene cannot be parsed."""
    fn = CHECKS.get(task_cfg.get("name"))
    if fn is None:
        return None, dict(reason=f"no geometric check for task {task_cfg.get('name')!r}")
    return fn(before_blobs or [], after_blobs or [], task_cfg)


def bee_in_tape_roll(before, after, task_cfg):
    """Success: a tape-roll blob lies within MOVE_TOL of its start, something is inside its hole (a separate blob
    within inner_r of the hole centre, new above-table cells inside, or the visible hole floor mostly covered)
    and no bee-sized blob lies elsewhere."""
    pri = task_cfg.get("perception", {}) or {}
    ring_prior = {k: v for k, v in (pri.get("tape_roll") or {}).items() if k not in ("hollow", "inner_r")}
    bee_prior = pri.get("foam_bee") or {}
    ring_b = layout.assign(task_cfg, before).get("tape_roll")
    rep = dict(before_ring=_brief(ring_b))
    if not after:
        return None, _why(rep, "no blobs after")
    cands = [b for b in after if layout.matches(ring_prior, b)]
    if cands:
        ring_a = min(cands, key=lambda b: _dist(b["xy"], ring_b["xy"])) if ring_b else cands[0]
    else:
        near = [b for b in after if ring_b and _dist(b["xy"], ring_b["xy"]) < EMPTY_TOL]
        if near:
            rep["at_ring_spot"] = _brief(near[0])
            return False, _why(rep, "tape roll disturbed: the blob at its spot is not a flat ring")
        return None, _why(rep, "tape roll not found after")
    rep["after_ring"] = _brief(ring_a)
    if ring_b:
        moved = _dist(ring_a["xy"], ring_b["xy"])
        rep["ring_moved_m"] = round(moved, 4)
        if moved > MOVE_TOL:
            return False, _why(rep, f"tape roll moved {moved * 100:.1f} cm")
    hole = ring_a["hole_xy"] if ring_a.get("hollow") else ring_a["xy"]
    r_in = ring_b.get("inner_r") if ring_b and ring_b.get("hollow") else None
    r_in = r_in or ring_a.get("inner_r") or 0.6 * (ring_a.get("outer_r") or 0.05)
    others = [b for b in after if b is not ring_a]
    inside = [b for b in others if _dist(b["xy"], hole) < r_in]
    outside = [b for b in others if layout.matches(bee_prior, b) and _dist(b["xy"], hole) >= r_in]
    fill = _hole_fill(ring_a, ring_b, hole, r_in)
    rep.update(inner_r=round(float(r_in), 4), inside_blobs=[_brief(b) for b in inside],
               outside_bees=[_brief(b) for b in outside], **fill)
    if outside:
        return False, _why(rep, "bee-sized blob outside the tape roll")
    if inside or fill["new_cells_inside"] >= 6 or fill["floor_covered"]:
        return True, _why(rep, "bee inside the tape roll's hole")
    return None, _why(rep, "nothing seen inside the hole and no bee elsewhere (hidden?)")


def cup_down_stacking(before, after, task_cfg):
    """Success: one blob at the base cup's spot, 6..35 mm taller than the base cup (nested upside-down cups),
    nothing at the moved cup's spot and no other cup-sized blob."""
    pri = task_cfg.get("perception", {}) or {}
    pick = layout.assign(task_cfg, before)
    base_b, move_b = pick.get("base_cup"), pick.get("move_cup")
    rep = dict(before=dict(base_cup=_brief(base_b), move_cup=_brief(move_b)))
    if base_b is None or move_b is None:
        return None, _why(rep, "both cups not found before")
    if not after:
        return None, _why(rep, "no blobs after")
    at_base = [b for b in after if _dist(b["xy"], base_b["xy"]) < SPOT_TOL]
    at_move = [b for b in after if _dist(b["xy"], move_b["xy"]) < EMPTY_TOL]
    rep.update(at_base=[_brief(b) for b in at_base], at_move=[_brief(b) for b in at_move])
    if not at_base:
        return False, _why(rep, "nothing at the base cup's spot")
    stack = max(at_base, key=lambda b: b["height"])
    dh = float(stack["height"]) - float(base_b["height"])
    rep["stack_minus_base_mm"] = round(dh * 1000, 1)
    if at_move:
        return False, _why(rep, "a blob is still at the moved cup's spot")
    if not 0.006 <= dh <= 0.035:
        return False, _why(rep, f"stack is {dh * 1000:.0f} mm taller than the base cup, expected 6..35 mm")
    others = [b for b in after if b is not stack and layout.matches(pri.get("move_cup") or {}, b)]
    if others:
        rep["others"] = [_brief(b) for b in others]
        return False, _why(rep, "another cup-sized blob on the table")
    return True, _why(rep, "nested cups at the base cup's spot")


def bee_in_bowl(before, after, task_cfg):
    """Success: the bowl blob lies within BOWL_MOVE_MAX of its start, still upright (footprint and height kept),
    something new rises above its empty inside near the centre (the bee, merged with the bowl blob or separate) and no
    bee-sized blob lies outside the rim. A bee resting on the rim is left to the VLM judge: the bowl's measured
    footprint varies by several cm between scans (depth noise), too much to see an overhang from it."""
    bee_prior = (task_cfg.get("perception", {}) or {}).get("bee") or {}
    bowl_b = layout.assign(task_cfg, before).get("bowl")
    rep = dict(before_bowl=_brief(bowl_b))
    if bowl_b is None:
        return None, _why(rep, "bowl not found before")
    if not after:
        return None, _why(rep, "no blobs after")
    bowl_a = min(after, key=lambda b: _dist(b["xy"], bowl_b["xy"]))
    moved = _dist(bowl_a["xy"], bowl_b["xy"])
    rep.update(after_bowl=_brief(bowl_a), bowl_moved_m=round(moved, 4))
    if moved > BOWL_MOVE_MAX:
        return False, _why(rep, f"bowl moved {moved * 100:.1f} cm or is gone")
    ext_b, ext_a = sorted(bowl_b["extent"]), sorted(bowl_a["extent"])
    if ext_a[0] < 0.75 * ext_b[0] or bowl_a["height"] < 0.75 * bowl_b["height"]:
        return False, _why(rep, "the bowl is tipped over or not at its spot")
    c, r_rim = np.asarray(bowl_a["xy"], float), 0.5 * ext_b[1]
    others = [b for b in after if b is not bowl_a]
    outside = [b for b in others if layout.matches(bee_prior, b) and _dist(b["xy"], c) > r_rim and b["n"] >= MIN_BEE_POINTS]
    rep.update(rim_r=round(r_rim, 4), outside_bees=[_brief(b) for b in outside])
    if outside:
        return False, _why(rep, "bee-sized blob outside the bowl")
    r_in = 0.55 * r_rim
    pts_a = [p for p in (_rel(b, c, r_in) for b in [bowl_a] + [b for b in others if _dist(b["xy"], c) <= r_rim]) if len(p)]
    pts_a = np.vstack(pts_a) if pts_a else np.zeros((0, 3))
    base = _profile(_rel(bowl_b, np.asarray(bowl_b["xy"], float), r_in), bowl_b["xy"], r_in)
    rise = pts_a[:, 2] - base(np.linalg.norm(pts_a[:, :2] - c, axis=1))
    risen = tabletop.cells(pts_a[rise > RISE, :2])
    rep.update(inside_floor_m=round(float(base(np.zeros(1))[0]), 4), risen_cells=risen)
    if risen >= RISEN_CELLS:
        return True, _why(rep, "bee inside the bowl")
    return None, _why(rep, "nothing seen inside the bowl and no bee elsewhere (hidden?)")


SLOT_MIN_POINTS = 60    # side-depth points in the bottom-slot window needed to judge it (measured: 1300 with the drawer in)
FLUSH_SHARE = 0.6       # share of slot points on the front plane for "flush" (measured: 0.72-0.89 with the drawer in)
BEHIND_SHARE = 0.3      # share more than 2 cm behind the plane: the slot is empty (measured: 0.41 for an empty slot)
FRONT_SHARE = 0.3       # share more than flush_tol in front of the plane: the drawer sticks out
CABINET_MOVE_MAX = 0.02 # m


def slot_profile(points, table_z, fx, tol):
    """Shares of the bottom-slot window's points on, behind and in front of the cabinet's front plane."""
    n = np.asarray(fx["front_normal"], float)
    d = points[:, :2] @ n - fx["front_offset"]
    t = points[:, :2] @ np.array([-n[1], n[0]])
    h = points[:, 2] - table_z
    m = (h > fx["slot_h"][0]) & (h < fx["slot_h"][1]) & (t > fx["slot_t"][0]) & (t < fx["slot_t"][1]) & (d > -0.15) & (d < 0.25)
    dd = d[m]
    if len(dd) < SLOT_MIN_POINTS:
        return dict(n=int(len(dd)))
    return dict(n=int(len(dd)), flush=round(float(np.mean(np.abs(dd) < tol)), 3), behind=round(float(np.mean(dd > 0.02)), 3),
                front=round(float(np.mean(dd < -tol)), 3), median_d=round(float(np.median(dd)), 4))


def drawer_rl(before, after, task_cfg):
    """Success: the bottom slot's window is mostly on the cabinet's front plane (the drawer front, flush), not behind it
    (empty slot) or in front of it (sticking out); no drawer-sized blob lies on the table in front of the cabinet; the
    cabinet blob has not moved."""
    fixtures = task_cfg["fixtures"]
    fx, tol = fixtures["cabinet"], fixtures.get("flush_tol", 0.012)
    pri = task_cfg.get("perception", {}) or {}
    if not after:
        return None, dict(reason="no blobs after")
    pts = np.vstack([np.asarray(b["points"], float) for b in after if len(b.get("points", []))])
    table_z = float(np.median([b["z_top"] - b["height"] for b in after]))
    prof = slot_profile(pts, table_z, fx, tol)
    rep = dict(slot=prof)
    n = np.asarray(fx["front_normal"], float)
    hc = (pri.get("cabinet") or {}).get("height", [0.08, 0.12])
    big = lambda b: hc[0] < b.get("height", 0) < hc[1] and 0.2 < max(b.get("extent", [0, 0])) < 0.36  # noqa: E731
    # by size: the standing drawer (~0.12 m) also fits the cabinet prior's extent range (episode 46: "moved 27 cm")
    cab_a = [b for b in after if big(b)]
    cab_b = [b for b in (before or []) if big(b)]
    if cab_a and cab_b:
        moved = _dist(max(cab_a, key=lambda b: b["n"])["xy"], max(cab_b, key=lambda b: b["n"])["xy"])
        rep["cabinet_moved_m"] = round(moved, 4)
        if moved > CABINET_MOVE_MAX:
            return False, _why(rep, f"the cabinet moved {moved * 100:.1f} cm")
    loose = [b for b in after if layout.matches(pri.get("drawer") or {}, b) and np.asarray(b["xy"][:2]) @ n - fx["front_offset"] < -0.08]
    rep["drawer_on_table"] = [_brief(b) for b in loose]
    if loose:
        return False, _why(rep, "a drawer-sized blob is still on the table")
    if "flush" not in prof:
        return None, _why(rep, f"only {prof['n']} points in the slot window (arm or object in the way?)")
    if prof["front"] > FRONT_SHARE:
        return False, _why(rep, "the drawer sticks out of the slot")
    if prof["flush"] >= FLUSH_SHARE:
        return True, _why(rep, "drawer flush in the bottom slot")
    if prof["behind"] > BEHIND_SHARE:
        return False, _why(rep, "the bottom slot is empty")
    return None, _why(rep, "slot window ambiguous")


def drawer_lying(before, after, task_cfg):
    """Success (cabinet lying on its back): the target compartment's opening (shrunk 1 cm) is filled at rim height (the
    drawer's top flush, within flush_tol), no drawer-sized blob stands on the table, the cabinet has not moved."""
    fx = task_cfg["fixtures"]["cabinet"]
    tol = task_cfg["fixtures"].get("flush_tol", 0.012)
    pri = task_cfg.get("perception", {}) or {}
    if not after:
        return None, dict(reason="no blobs after")
    pts = np.vstack([np.asarray(b["points"], float) for b in after if len(b.get("points", []))])
    table_z = float(np.median([b["z_top"] - b["height"] for b in after]))
    y = np.radians(fx["yaw_deg"]); a = np.array([np.cos(y), np.sin(y)]); bb = np.array([-a[1], a[0]])
    d = pts[:, :2] - np.asarray(fx["target_centre"], float)
    s, t, h = d @ a, d @ bb, pts[:, 2] - table_z
    hl, hw = fx["compartment"][0] / 2 - 0.01, fx["compartment"][1] / 2 - 0.01
    m = (np.abs(s) < hl) & (np.abs(t) < hw) & (h > 0.003)
    rep = dict(n=int(m.sum()))
    hc = (pri.get("cabinet") or {}).get("height", [0.08, 0.12])
    big = lambda b: hc[0] < b.get("height", 0) < hc[1] and 0.2 < max(b.get("extent", [0, 0])) < 0.36  # noqa: E731
    # by size: the standing drawer (~0.12 m) also fits the cabinet prior's extent range (episode 46: "moved 27 cm")
    cab_a = [b for b in after if big(b)]
    cab_b = [b for b in (before or []) if big(b)]
    if cab_a and cab_b:
        # outline centre, not the blob centroid: an inserted drawer fills a compartment and moves the centroid ~3 cm
        oa, ob = (outline_xy(max(c, key=lambda b: b["n"]), fx["yaw_deg"]) for c in (cab_a, cab_b))
        if oa is not None and ob is not None:
            moved = _dist(oa, ob)
            rep["cabinet_moved_m"] = round(moved, 4)
            if moved > CABINET_MOVE_MAX:
                return False, _why(rep, f"the cabinet moved {moved * 100:.1f} cm")
    c0 = np.asarray(fx["centre"], float)
    loose = [b for b in after if layout.matches(pri.get("drawer") or {}, b)
             and np.linalg.norm(np.asarray(b["xy"][:2]) - c0) > max(fx["size"]) / 2 + 0.05]
    rep["drawer_on_table"] = [_brief(b) for b in loose]
    if loose:
        return False, _why(rep, "a drawer-sized blob is still on the table")
    if m.sum() < SLOT_MIN_POINTS:
        return None, _why(rep, f"only {int(m.sum())} points in the target compartment (arm in the way?)")
    hh = h[m]
    rep.update(median_h=round(float(np.median(hh)), 4), flush=round(float(np.mean(np.abs(hh - fx["rim_h"]) < tol)), 3),
               above=round(float(np.mean(hh > fx["rim_h"] + tol)), 3), deep=round(float(np.mean(hh < fx["rim_h"] - 0.03)), 3))
    if rep["above"] > FRONT_SHARE:
        return False, _why(rep, "the drawer sticks out of the compartment")
    if rep["flush"] >= FLUSH_SHARE:
        return True, _why(rep, "drawer flush in the target compartment")
    if rep["deep"] > BEHIND_SHARE:
        return False, _why(rep, "the target compartment is empty")
    return None, _why(rep, "target compartment ambiguous")


CHECKS = {"bee_in_tape_roll": bee_in_tape_roll, "cup_down_stacking": cup_down_stacking, "bee_in_bowl": bee_in_bowl,
          "drawer_rl": drawer_rl, "drawer_lying": drawer_lying}


def outline_xy(b, yaw_deg, lo=0.02):
    """Centre (xy, m) of a blob's outline along/across yaw_deg (2nd-98th percentiles of its points above lo m)."""
    P = np.asarray(b.get("points") if b.get("points") is not None else [], float)
    if len(P) < 200:
        return None
    P = P[P[:, 2] - (b["z_top"] - b["height"]) > lo]
    if len(P) < 200:
        return None
    y = np.radians(yaw_deg); R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    q = P[:, :2] @ R
    mid = np.array([np.percentile(q[:, k], [2, 98]).mean() for k in (0, 1)])
    return [float(v) for v in R @ mid]


def _rel(b, c, r):
    """Points of blob b within r of c (xy), with z as height above the table."""
    P = b.get("points")
    if P is None or len(P) == 0:
        return np.zeros((0, 3))
    P = np.asarray(P, float)
    P = P[np.linalg.norm(P[:, :2] - c, axis=1) < r]
    return np.c_[P[:, :2], P[:, 2] - (b["z_top"] - b["height"])]


def _profile(pts, c, r, step=0.005):
    """Empty bowl's inside height as a function of distance from its centre: 75th percentile per 5 mm ring of pts
    (x, y, height), empty rings interpolated; 0 everywhere when nothing is seen (only table inside)."""
    edges = np.arange(0, r + step, step)
    rr = np.linalg.norm(pts[:, :2] - np.asarray(c, float), axis=1) if len(pts) else np.zeros(0)
    mid, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rr >= lo) & (rr < hi)
        if m.sum() >= 5:
            mid.append(0.5 * (lo + hi))
            val.append(float(np.percentile(pts[m, 2], 75)))
    if not mid:
        return lambda d: np.zeros(len(np.atleast_1d(d)))
    return lambda d: np.interp(np.atleast_1d(d), mid, val)


def _hole_fill(ring_a, ring_b, hole, r_in):
    """Above-table cells inside the hole (after vs before) and whether the visible hole floor shrank."""
    r = 0.8 * r_in
    cells_a = tabletop.cells(_within(ring_a.get("points"), hole, r))
    floor_a = tabletop.cells(_within(ring_a.get("table_xy"), hole, r))
    cells_b = floor_b = 0
    if ring_b:
        hole_b = ring_b["hole_xy"] if ring_b.get("hollow") else ring_b["xy"]
        cells_b = tabletop.cells(_within(ring_b.get("points"), hole_b, r))
        floor_b = tabletop.cells(_within(ring_b.get("table_xy"), hole_b, r))
    return dict(cells_inside=[cells_b, cells_a], new_cells_inside=max(cells_a - cells_b, 0),
                floor_cells=[floor_b, floor_a], floor_covered=bool(floor_b >= 6 and floor_a <= 0.4 * floor_b))


def _within(pts, c, r):
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2))
    xy = np.asarray(pts, float)[:, :2]
    return xy[np.linalg.norm(xy - np.asarray(c, float), axis=1) < r]


def _dist(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))


def _brief(b):
    return None if b is None else {k: b.get(k) for k in ("xy", "height", "extent", "hollow", "inner_r", "outer_r")}


def _why(rep, reason):
    rep["reason"] = reason
    return rep
