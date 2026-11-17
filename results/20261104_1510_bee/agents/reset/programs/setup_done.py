"""Test helper: put the bee into the bowl (the task's done state) so reset.py can be tested from it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reset as R                                    # noqa: E402
from rlpj.robot.session import RobotSession          # noqa: E402


def main():
    with RobotSession(program="setup_done") as s:
        R.S = s
        tz = float(s.table_z)
        s.phase("perceive")
        bowl, bee = R.perceive_retry("start")
        s.require(bowl is not None and bee is not None, "need a bowl and a bee on the table")
        s.phase("pick")
        h = float(np.clip(bee["height"], 0.03, 0.055))
        gz = tz + h + R.BEE_TOOL_ABOVE_TOP
        near = R.dist(bee["xy"], bowl["xy"]) < 0.14
        ok, g, y = R.pick_bee(bee["xy"], R.long_axis(bee) + 90.0, gz, R.BEE_BOWL_OPEN if near else 0.0, "bee")
        s.require(ok, "bee not held")
        s.phase("drop_in_bowl")
        release = float(bowl["z_top"]) + 0.03 + (gz - tz)
        R.place_bee(bowl["xy"], y, release, "into_bowl")
        s.phase("verify")
        s.home("home_verify")
        b2, e2 = R.perceive_retry("after", need_bee=False, tries=2)
        done = b2 is not None and e2 is None
        s.summary(verify=dict(bowl=None if b2 is None else b2["xy"], bee_on_table=e2 is not None), measured=None)
        s.claim("success" if done or s.dry else "fail")


if __name__ == "__main__":
    main()
