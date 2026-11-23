"""Offline: where the fitted and the operator's camera-side table edges fall in the side image (is the fit just the image border?)."""
import json, os
import cv2
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = json.load(open(R + "/inputs/calibration/side_pose_935322070708.json"))
Tbs = np.array(sp["T_base_side"] if "T_base_side" in sp else sp["T_base_cam"], float)
I = json.load(open(R + "/inputs/calibration/intrinsics_935322070708_1280x720.json"))
K = np.array(I["K"], float); D = np.array(I["dist"], float)
Tsb = np.linalg.inv(Tbs)
def px(x, y, z=-0.0156):
    p = Tsb[:3, :3] @ np.array([x, y, z]) + Tsb[:3, 3]
    return cv2.projectPoints(p[None], np.zeros(3), np.zeros(3), K, D)[0].ravel()
print("x      fit-edge px          operator-edge px     (image 1280x720)")
for x in (0.20, 0.25, 0.29, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    a = px(x, -0.0948 - 0.2568 * x); b = px(x, -0.13 - x / 6)
    print(f"{x:.2f}  {a[0]:7.0f} {a[1]:6.0f}      {b[0]:7.0f} {b[1]:6.0f}")
