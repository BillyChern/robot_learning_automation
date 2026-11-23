"""Robot program: measure the lying cabinet (run through run_attempt, see scripts/run_measure_lying.py).

Pulls the arm back and up out of the side camera's view of the cabinet (straight tool line) and saves a side image and
point cloud, then looks straight down at the cabinet with the wrist camera from ~30 cm above the rim and saves the
wrist image with the tool pose, then goes home. Every move is model-checked against the task's keep-out (the whole
cabinet with a margin).
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from rlpj import config
from rlpj.perception import tabletop
from rlpj.robot import calib
from rlpj.robot.plan import tool_down_T
from rlpj.robot.session import RobotSession

OUT = Path(os.environ.get("RLPJ_ATTEMPT_DIR", "."))


def main():
    cab = config.load()["task"]["fixtures"]["cabinet"]
    cx, cy = cab["centre"]
    tz = calib.table_z()
    with RobotSession(program="measure_lying") as s:
        s.phase("clear_view")
        s.move_line("pull_back_up", tool_down_T([0.25, 0.10, tz + 0.55], 0.0), "slow")
        time.sleep(2.5)                                            # side camera auto-exposure
        cv2.imwrite(str(OUT / "side_clear.jpg"), s.side.latest()[0])
        np.save(OUT / "cloud_clear.npy", tabletop.cloud(s.side, n=10))
        s.decide("side image and cloud saved with the arm pulled back")
        s.phase("look_down")
        yaw = float(cab["yaw_deg"])
        s.move_line("above_cabinet", tool_down_T([cx, cy, tz + cab["rim_h"] + 0.32], yaw), "slow")
        time.sleep(1.5)
        cv2.imwrite(str(OUT / "wrist_down.jpg"), s.wrist.latest()[0])
        cv2.imwrite(str(OUT / "side_with_arm.jpg"), s.side.latest()[0])
        (OUT / "wrist_down_pose.json").write_text(json.dumps(dict(pose=[float(v) for v in s.feedback()["pose"]],
                                                                  q=[float(v) for v in s.feedback()["q"]])))
        s.decide("wrist look-down saved")
        s.phase("home")
        s.move_line("back_up", tool_down_T([0.25, 0.10, tz + 0.55], 0.0), "slow")
        s.go_home()
        s.claim("success")


if __name__ == "__main__":
    main()
