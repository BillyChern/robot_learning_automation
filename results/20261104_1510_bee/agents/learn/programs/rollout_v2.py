"""rlpj.vla.rollout + eased chunks (rollout_smooth.py) + release handling.

Release handling: the session refuses to open the gripper on a held bee unless the tool is inside the task release
zone (x/y) and <= max_height above the table; the stock rollout then aborts and the arm goes home still holding the bee
(eval 03 ep 117, eval 04 ep 120 stalled the evaluation). Here, when the policy commands "open" while holding:
  - inside the x/y zone but too high: a slow straight descent (move_line, model-checked) to RELEASE_H above the table,
    then open;
  - outside the x/y zone: the open is ignored (logged) and the policy keeps running.
Everything moves through RobotSession; the program still claims "fail". Same arguments as rlpj.vla.rollout."""
import sys

import numpy as np

import rollout_smooth  # noqa: F401  (installs the eased segments in rlpj.vla.rollout)
import rlpj.vla.rollout as R
from rlpj.robot import calib
from rlpj.robot import session as S

RELEASE_H = 0.19     # m above the table (session limit 0.20)
_orig_gripper = S.RobotSession.gripper


def _pose_T(pose):
    return np.array(calib.pose_to_T(pose), float)   # Kortex tool_pose -> 4x4 (repo convention)


def gripper(self, value, what=""):
    z = getattr(self, "release_zone", None)
    value = float(np.clip(value, 0, 1))
    reading = getattr(self, "grip_reading", None)
    if z and reading is not None and value < getattr(self, "grip_frac", 1.0) - 0.02:
        lo, hi = z.get("reading", (60.0, 92.0))
        if lo <= reading <= hi:
            pose = self.feedback()["pose"]
            x, y, h = float(pose[0]), float(pose[1]), float(pose[2]) - self.table_z
            in_xy = z["x"][0] <= x <= z["x"][1] and z["y"][0] <= y <= z["y"][1]
            if not in_xy:
                self.decide(f"policy open ignored: holding (reading {reading:.1f}) at ({x:.3f}, {y:.3f}) outside the release zone")
                return reading
            if h > min(RELEASE_H, z.get("max_height", 0.20)):
                T = _pose_T(pose)
                T[2, 3] = self.table_z + RELEASE_H
                self.decide(f"policy open at {h:.3f} m above the table: descend to {RELEASE_H} m before releasing")
                self.move_line("release_descend", T, speed="slow")
    return _orig_gripper(self, value, what)


S.RobotSession.gripper = gripper

if __name__ == "__main__":
    sys.exit(R.main())
