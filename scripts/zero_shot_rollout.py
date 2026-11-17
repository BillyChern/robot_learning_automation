"""One blind zero-shot rollout: agent 3's rollout_look (scripted look move above the bowl, then rollout_floor's policy
loop) with object detection made generic: the bowl as before, and the object is the largest other blob in the task
area of any plausible size (nothing about the new object is given). Run by scripts/zero_shot_eval.py through
run_attempt (cwd = the learn agent's workspace). Same arguments as rollout_floor."""
import sys
from pathlib import Path

import numpy as np

PROGRAMS = Path(__file__).resolve().parents[1] / "runs" / "20261104_1510_bee" / "agents" / "learn" / "programs"
sys.path.insert(0, str(PROGRAMS))

import rollout_look  # noqa: E402  (installs the look move before the policy phase)
import rollout_floor  # noqa: E402

E = rollout_look.E
OBJ_H = (0.010, 0.15)        # any object from a flat toy to a tall one
OBJ_EXT = (0.015, 0.20)
MIN_PTS = 150                # side-depth points; below this a blob is depth noise


def classify_any(blobs):
    """-> (bowl, object, others): bowl as the bee programs find it; object = most points among the other blobs."""
    cand = [b for b in blobs if E.in_range(b["xy"][0], E.TASK_X) and E.in_range(b["xy"][1], E.TASK_Y)]
    bowls = [b for b in cand if E.in_range(b["height"], E.BOWL_H) and E.in_range(max(b["extent"]), E.BOWL_EXT) and b["n"] >= 2500]
    bowl = max(bowls, key=lambda b: b["n"]) if bowls else None
    rest = [b for b in cand if b is not bowl and E.in_range(b["height"], OBJ_H) and E.in_range(max(b["extent"]), OBJ_EXT)
            and b["n"] >= MIN_PTS]
    obj = max(rest, key=lambda b: b["n"]) if rest else None
    return bowl, obj, [b for b in cand if b is not bowl and b is not obj]


E.classify = classify_any


def perceive_bowl_only(s):
    """Look-move inputs from the bowl alone: bowl centre (as rollout_look), and the jaw yaw from the object's long axis
    when side depth sees it, else the READY yaw (dark or small objects can be nearly invisible to depth)."""
    blobs = s.tabletop()
    tz = next((float(b["z_top"] - b["height"]) for b in blobs), None)
    s.require(tz is not None, "look: no objects on the table")
    bowl, obj, _ = classify_any(blobs)
    s.require(bowl is not None, f"look: no bowl in the task area x {E.TASK_X} y {E.TASK_Y} ({[E.blob_info(b) for b in blobs]})")
    s.require(abs(tz - s.table_z) < 0.012, f"look: fitted table z {tz:.4f} far from calibration {s.table_z:.4f}")
    bowl_xy = np.array(bowl["xy"][:2], float)
    cxy, cr, _ = E.bowl_circle(bowl, tz)
    if cr is not None and float(np.linalg.norm(np.asarray(cxy, float) - bowl_xy)) <= 0.03:
        bowl_xy = np.asarray(cxy, float)
    if obj is not None:
        obj_xy, long_deg, _, _ = E.bee_pose(obj, tz)
        yaw, how = E.pick_yaw(long_deg, E.tool_yaw(s)), f"object at {np.round(obj_xy, 3).tolist()} axis {long_deg:.0f} deg"
    else:
        yaw, how = float(E.READY_YAW), "object not seen in depth: READY yaw"
    s.decide(f"look: bowl {bowl_xy.round(3).tolist()}; {how} -> yaw {yaw:.0f}; table z {tz:.4f}")
    s.summary(look=dict(bowl=bowl_xy.round(3).tolist(), yaw=round(float(yaw), 1), object_seen=obj is not None))
    return tz, bowl_xy, yaw


rollout_look.perceive = perceive_bowl_only

if __name__ == "__main__":
    sys.exit(rollout_floor.main())
