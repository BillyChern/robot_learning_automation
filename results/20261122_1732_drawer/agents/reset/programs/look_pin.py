"""No-contact measurement run: wrist images of the drawer pin from three well-separated poses (for triangulation).
Saves per view: full-res 640x360 crop around the predicted pin, a downscaled full image, the measured tool pose."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj.robot import calib
from rlpj.robot.session import RobotSession
from rlpj.perception.wrist import project_base_to_wrist
import scene
from scene import N, TT, OFF, xy_of
import reset as R

PIN_EST = (0.006, -0.096, 0.050)    # out, t, h (rough)
VIEWS = [(-0.10, 0.10, 0.06), (0.0, 0.10, 0.15), (-0.03, 0.14, 0.05), (0.03, 0.12, 0.12), (-0.08, 0.06, 0.16)]


def look_T(p_tool, target, tz):
    z = R.unit(target - p_tool)
    x = np.r_[TT, 0.0]; x = R.unit(x - (x @ z) * z)
    T = np.eye(4); T[:3, :3] = np.stack([x, np.cross(z, x), z], 1); T[:3, 3] = p_tool
    return T


def main():
    with RobotSession(program="reset") as s:
        r = R.Reset(s, {})
        tz = r.tz
        P = np.r_[xy_of(OFF - PIN_EST[0], PIN_EST[1]), tz + PIN_EST[2]]
        s.phase("look")
        if np.abs(R.wrap180(r.q_now() - R.HOME_Q)).max() > 3:
            r.escape()
        r.to_ready()
        out = []
        for i, (dt, back, up) in enumerate(VIEWS):
            p = P + np.r_[-back * N + dt * TT, up]
            T = look_T(p, P, tz)
            try:
                q = r.ik(T, [np.array([0, 40, -120, 0, 20, j6], float) for j6 in (-90.0, 90.0)] + [R.SEED_DOWN])
                s.move_joint(f"view{i}", q, "normal")
            except Exception as e:
                s.log(f"view {i} skipped: {str(e)[:150]}")
                continue
            import time; time.sleep(0.8)
            img, _ = s.wrist.latest()
            fb = s.feedback()
            Tm = calib.pose_to_T(fb["pose"])
            uv = project_base_to_wrist(P[None], Tm)[0]
            cx, cy = (int(np.clip(uv[0] - 320, 0, 640)), int(np.clip(uv[1] - 180, 0, 360))) if np.all(np.isfinite(uv)) else (320, 180)
            if img is not None:
                cv2.imwrite(os.path.join(s.A, f"view{i}_crop.jpg"), img[cy:cy + 360, cx:cx + 640])
                cv2.imwrite(os.path.join(s.A, f"view{i}_full.jpg"), cv2.resize(img, (640, 360)))
            out.append(dict(view=i, pose=np.round(fb["pose"], 5).tolist(), q=np.round(fb["q"], 3).tolist(), crop_xy=[cx, cy],
                            pin_est_px=np.round(uv, 1).tolist()))
            s.log(f"view {i}: {out[-1]}")
            r.to_ready()
        s.summary(views=out)
        r.go_home("home_end")
        s.claim("success")


if __name__ == "__main__":
    main()
