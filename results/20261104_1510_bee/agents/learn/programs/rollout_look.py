"""Policy from the look pose (operator 2026-11-08): before the policy loop, do the execute program's scripted camera
move: perceive bowl and bee with the side camera, choose the grasp yaw (pick_yaw), to_ready, then travel(look_bowl)
to tool-down above the bowl at table z + LOOK_Z with that yaw (no bowl refine). The functions are imported from the
execute agent's execute.py (same checked RobotSession moves, inside the keep-in box). Then the wrapped rollout runs
its policy loop from there; grasp, lift and place are the policy's.

Policy loop: rollout_floor.main (stock loop + 8 mm tip-floor prefix handling; for full_v1). programs/rollout_look_v4.py wraps rollout_v4 (for full_v4 / v4 recipes).
Extra flag: --look-only (perceive + look move, then stop; for dry runs). Never claims success."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from rlpj.robot import session as S
from rlpj.robot.plan import tool_down_T

EXEC_PY = Path(__file__).resolve().parents[2] / "execute" / "programs" / "execute.py"
_spec = importlib.util.spec_from_file_location("bee_execute", str(EXEC_PY))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

import rlpj.vla.rollout as R

LOOK_ONLY = "--look-only" in sys.argv
if LOOK_ONLY:
    sys.argv.remove("--look-only")

    class _NoServer:                       # look-only dry runs need no policy server
        metadata, last_ms = {}, float("nan")

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    R.PolicyClient = _NoServer


def perceive(s):
    """execute.py run(): side-camera blobs -> bowl centre (rim fit / ring test), bee pose, grasp yaw."""
    def require(c, m):
        s.require(c, m)

    blobs = s.tabletop()
    tz = None
    for b in blobs:
        tz = float(b["z_top"] - b["height"])
        break
    require(tz is not None, "look: no objects on the table")
    bowl, bee, others = E.classify(blobs)
    require(bowl is not None, f"look: no bowl-sized blob ({[E.blob_info(b) for b in blobs]})")
    if bee is None:
        cxy0, cr0, _ = E.bowl_circle(bowl, tz)
        bee = E.split_bee_from_bowl(bowl, tz, cxy0, float(np.clip(cr0 or 0.072, 0.06, 0.082)))
    require(bee is not None, f"look: no bee-sized blob ({[E.blob_info(b) for b in blobs]})")
    require(abs(tz - s.table_z) < 0.012, f"look: fitted table z {tz:.4f} far from calibration {s.table_z:.4f}")
    bowl_xy = np.array(bowl["xy"][:2], float)
    cxy, cr, _ = E.bowl_circle(bowl, tz)
    if cr is not None:
        ring_xy = bowl_xy.copy()
        bowl_xy = np.asarray(cxy, float)
        if float(np.linalg.norm(bowl_xy - ring_xy)) > 0.03 and not bee.get("split"):
            bowl_xy = (bowl_xy + ring_xy) / 2.0
    bee_xy, long_deg, bee_len, bee_wid = E.bee_pose(bee, tz)
    if bee_len > E.BEE_LEN_SMEAR or bee.get("split") or max(bee["extent"]) > 0.12:
        P = np.asarray(bee["points"], float)
        upper = P[P[:, 2] - tz > 0.65 * float(bee["height"])]
        if len(upper) >= 60:
            _, long_deg, bee_len, bee_wid = E.bee_pose(dict(bee, points=upper), tz)
    yaw = E.pick_yaw(long_deg, E.tool_yaw(s))
    s.decide(f"look: bowl {bowl_xy.round(3).tolist()}, bee {np.round(bee_xy, 3).tolist()} long axis {long_deg:.0f} deg "
             f"-> yaw {yaw:.0f}; table z {tz:.4f}")
    s.summary(look=dict(bowl=bowl_xy.round(3).tolist(), bee=np.round(bee_xy, 3).tolist(), yaw=round(float(yaw), 1)))
    return tz, bowl_xy, yaw


def look(s):
    s.phase("perceive")
    try:
        tz, bowl_xy, yaw = perceive(s)
    except S.Abort as e:
        if os.environ.get("RLPJ_MOTION") == "1" or LOOK_ONLY:
            raise
        s.decide(f"dry run: look skipped ({str(e)[:80]}); policy loop from the current pose")
        return
    s.phase("look")
    E.to_ready(s, tz)
    E.travel(s, "look_bowl", tool_down_T([bowl_xy[0], bowl_xy[1], tz + E.LOOK_Z], yaw), "normal")
    s.decide(f"at the look pose above the bowl (z {tz + E.LOOK_Z:.3f}): policy takes over")


_orig_phase = S.RobotSession.phase


def phase(self, name, *a, **k):
    if name == "policy" and not getattr(self, "_look_done", False):
        self._look_done = True
        look(self)
        if LOOK_ONLY:
            self.summary(stop_reason="look only")
            raise S.Abort("look-only run: stop before the policy")
    return _orig_phase(self, name, *a, **k)


S.RobotSession.phase = phase

if __name__ == "__main__":
    import rollout_floor
    sys.exit(rollout_floor.main())
