"""Geometric success checks on before/after tabletop blobs, independent of the program's own verification."""
import numpy as np

from rlpj.perception import layout, tabletop

MOVE_TOL = 0.03      # m: the tape roll may not move more than this
SPOT_TOL = 0.04      # m: a blob this close to a cup's start is "at its spot"
EMPTY_TOL = 0.05     # m: a spot counts as empty when no blob lies within this


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


CHECKS = {"bee_in_tape_roll": bee_in_tape_roll, "cup_down_stacking": cup_down_stacking}


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
