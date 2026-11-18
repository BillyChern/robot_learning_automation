"""Diagnostic (no pulling): go to the pin pinch pose of reset.py, close the gripper in steps, wrist image + reading
at each step; then open and retreat home."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj.robot.session import RobotSession
import reset as R


def main():
    with RobotSession(program="reset") as s:
        r = R.Reset(s, {})
        s.phase("diag")
        if np.abs(R.wrap180(r.q_now() - R.HOME_Q)).max() > 3:
            r.escape()
        r.to_ready()
        R.PIN_OUT, R.PIN_H, r.pin_dy, r.pin_dz = R.PIN_VARIANTS[0]
        r.pin_t = R.PIN_T
        Tg = r.pin_T(R.PIN_OUT)
        z = Tg[:3, 2]
        T_pre = Tg.copy(); T_pre[:3, 3] -= 0.05 * z
        T_hi = T_pre.copy(); T_hi[2, 3] = r.tz + R.PIN_HI; T_hi[:2, 3] -= 0.04 * R.N
        s.gripper(R.PRECLOSE, "pre-close")
        q = r.ik(T_hi, [np.array([0, 40, -120, 0, j5, j6], float) for j5 in (20.0, 40.0, 60.0) for j6 in (-90.0, 90.0)])
        s.move_joint("above_pin", q, "normal")
        r.line("down_to_pin_pre", T_pre, "normal")
        r.line("approach_pin", Tg, "slow", tip_margin=R.PIN_MARGIN)
        steps = []
        for g in (0.706, 0.78, 0.84, 0.88, 0.92, 0.96, 1.0):
            rd = s.gripper(g, f"step {g}")
            time.sleep(0.5)
            img = s.wrist.latest()[0]
            if img is not None:
                cv2.imwrite(os.path.join(s.A, f"close_{int(g * 100)}.jpg"), img[360:720, 320:960])
            steps.append([g, round(rd, 1)])
            s.log(f"step {g}: reading {rd:.1f}")
        s.summary(close_steps=steps)
        s.gripper(R.PRECLOSE, "open")
        r.line("back_from_pin", T_pre, "slow", tip_margin=R.PIN_MARGIN)
        r.line("up_from_pin", T_hi, "normal")
        r.go_home("home_end")
        s.claim("success")


if __name__ == "__main__":
    main()
