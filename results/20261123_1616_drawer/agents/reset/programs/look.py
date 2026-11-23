"""Wrist look-down at the target compartment from two safe heights (no contact), then back to the start pose.
Saves a downscaled view with the compartment outline, a native-resolution centre crop and the tool pose of each view."""
import json
import os
import time

import cv2
import numpy as np

from rlpj import config
from rlpj.perception.wrist import project_base_to_wrist
from rlpj.robot import calib
from rlpj.robot.plan import tool_down_T
from rlpj.robot.session import RobotSession

FX = config.load("drawer_lying")["task"]["fixtures"]["cabinet"]
YAW_DEG = float(FX["yaw_deg"])
YAW = np.radians(YAW_DEG)
A_AX = np.array([np.cos(YAW), np.sin(YAW)])
B_AX = np.array([-A_AX[1], A_AX[0]])
T0 = np.array(FX["target_centre"], float)
RIM = float(FX["rim_h"])
SEED_DOWN = np.array([0, 25, -110, 0, -45, 90.0])
HEIGHTS = (0.26, 0.17)                               # tool height above the table (open tips stay > 4 cm above the rim)


def look_T(h, tz):
    """Tool-down pose at height h whose camera optical axis passes through the compartment centre at rim height."""
    T = tool_down_T(np.r_[T0, tz + h], YAW_DEG)
    Tc = T @ calib.T_tool_cam()
    X = np.r_[T0, tz + RIM]
    lam = (X[2] - Tc[2, 3]) / Tc[2, 2]
    T[:2, 3] += X[:2] - (Tc[:2, 3] + lam * Tc[:2, 2])
    return T


with RobotSession(program="reset") as s:
    s.phase("look")
    tz = calib.table_z()
    views = []
    q = SEED_DOWN
    for h in HEIGHTS:
        T = look_T(h, tz)
        q, _, _ = s.planner.ik_exact(T, q)
        s.move_joint(f"look_{int(h * 100)}", q, "normal")
        time.sleep(0.8)
        img = s.wrist.latest()[0]
        Tn = calib.pose_to_T(s.feedback()["pose"])
        hl, hw = FX["compartment"][0] / 2, FX["compartment"][1] / 2
        corners = np.array([np.r_[T0 + i * hl * A_AX + j * hw * B_AX, tz + RIM] for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        uv = project_base_to_wrist(corners, Tn)
        c = project_base_to_wrist(np.r_[T0, tz + RIM][None], Tn)[0]
        draw = img.copy()
        if np.all(np.isfinite(uv)):
            cv2.polylines(draw, [uv.astype(np.int32)], True, (0, 255, 0), 2)
        name = f"look_{int(h * 100)}"
        cv2.imwrite(os.path.join(s.A, f"{name}.jpg"), cv2.resize(draw, (640, 360)))
        u, v = int(np.clip(c[0], 320, img.shape[1] - 320)), int(np.clip(c[1], 180, img.shape[0] - 180))
        cv2.imwrite(os.path.join(s.A, f"{name}_crop.png"), img[v - 180:v + 180, u - 320:u + 320])
        views.append(dict(h=h, tool=np.round(Tn[:3, 3], 4).tolist(), centre_uv=np.round(c, 1).tolist(),
                          crop_origin=[u - 320, v - 180], corners_uv=np.round(uv, 1).tolist()))
        s.log(f"view {views[-1]}")
    with open(os.path.join(s.A, "views.json"), "w") as f:
        json.dump(views, f)
    s.summary(views=[dict(h=v["h"], tool=v["tool"], centre_uv=v["centre_uv"]) for v in views])
    s.home()
    s.claim("success")
