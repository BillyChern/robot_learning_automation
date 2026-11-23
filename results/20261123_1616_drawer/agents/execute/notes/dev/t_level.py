import json, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot.plan import tool_down_T
# 1. sign of the levelling: a plane with slope g along a must become flat after rotating by atan(g) about +b
y = 18.3; a, b = V.axes(y); A, B = np.r_[a, 0], np.r_[b, 0]
for deg in (-5.0, -2.0, 3.0):
    g = np.tan(np.radians(deg)); s, t = np.meshgrid(np.linspace(-.03, .03, 9), np.linspace(-.05, .05, 9))
    P = (s.ravel()[:, None] * A + t.ravel()[:, None] * B); P[:, 2] = g * s.ravel()
    R = cv2.Rodrigues(B * np.arctan(g))[0]; Q = P @ R.T
    print("tilt", deg, "-> after levelling", round(float(np.degrees(np.arctan(np.polyfit(Q @ A, Q[:, 2], 1)[0]))), 3))
# 2. opening fit on the saved 640 px look-downs (commanded look pose), with the drawer beside the cabinet
Ttc = V.wrist_cam()[2]; tz = -0.0154; C = np.array([.4519, .0029]); z = tz + .095
T = tool_down_T(np.r_[C, tz + .30], 0.0); Tc = T @ Ttc; o, zc = Tc[:3, 3], Tc[:3, 2]
T[:2, 3] += C - (o + (z - o[2]) / zc[2] * zc)[:2]
for n in (10, 12, 14):
    im = cv2.resize(cv2.imread(f"{W}/attempts/attempt_{n}/wrist_opening.jpg"), (1280, 720), interpolation=cv2.INTER_CUBIC)
    r = V.find_opening(im, T, C, 18.3, [.0757, .115], z)
    print("attempt", n, {k: r.get(k) for k in ("ok", "c", "yaw", "wall_a", "width_b", "why", "n")})
