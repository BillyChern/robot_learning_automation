"""Project the opening, the cabinet and the expected held drawer into the recorded wrist frame at the hover of attempt 18."""
import json, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot import calib
from rlpj.robot.kin import Kin
A = W + "/attempts/attempt_18/"
# the commanded hover pose (the offline Kin has no per-arm fit, so its FK is several mm off): c = 0, values from attempt 18
from rlpj.robot.plan import tool_down_T
S = json.load(open(A + "summary.json")); O = S["objects"]; off = np.array(O["tab"]["xy"]) - np.array(O["drawer"]["c"]); dy = np.radians(-34.7)
tool_xy = np.array([0.4519, 0.0029]) + np.array([np.cos(dy) * off[0] - np.sin(dy) * off[1], np.sin(dy) * off[0] + np.cos(dy) * off[1]])
T = tool_down_T(np.r_[tool_xy, -0.0154 + S["grasp"]["h_g"] + 0.103], -71.7)
K, D, Ttc = V.wrist_cam(); tz = -0.0154
print("off", np.round(off, 4).tolist(), "tool xyz", np.round(T[:3, 3], 4).tolist(), "tool x", np.round(T[:3, 0], 3).tolist(), "tool y", np.round(T[:3, 1], 3).tolist())
print("cam axes in base: right", np.round((T @ Ttc)[:3, 0], 3).tolist(), "down", np.round((T @ Ttc)[:3, 1], 3).tolist(), "cam pos", np.round((T @ Ttc)[:3, 3], 4).tolist())
def proj(P): return calib.project(calib.apply(np.linalg.inv(T @ Ttc), np.asarray(P, float)), K, D) / 2.0     # 640 px frame
def rect(c, y, sz, z):
    a, b = V.axes(y); return np.array([np.r_[c + sa * sz[0] / 2 * a + sb * sz[1] / 2 * b, z] for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
c = np.array([0.4519, 0.0029]); a, b = V.axes(18.3)
im = cv2.imread(A + "../../dev_out/ep3_hover_dash.jpg")
cv2.polylines(im, [np.int32(proj(rect(c, 18.3, (.0757, .115), tz + .095)))], True, (0, 255, 0), 1, cv2.LINE_AA)                # opening
cab = c + (0.5286 - 0.4469) * a * 0 + 0.0877 * a
cv2.polylines(im, [np.int32(proj(rect(cab, 18.3, (.275, .139), tz + .095)))], True, (255, 0, 0), 1, cv2.LINE_AA)               # cabinet outline
for z, col in ((tz + .103 + .0945, (0, 255, 255)), (tz + .103, (255, 0, 255))):
    cv2.polylines(im, [np.int32(proj(rect(c, 18.3, (.0723, .1131), z)))], True, col, 1, cv2.LINE_AA)                           # planned drawer: top, bottom
pin = proj((T[:3, 3] + T[:3, 2] * 0.044)[None])[0]; print("pinch point px", np.round(pin, 1).tolist()); cv2.drawMarker(im, tuple(np.int32(pin)), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
for n, P in (("+a", c + .06 * a), ("-a", c - .06 * a), ("+b", c + .08 * b), ("-b", c - .08 * b)):
    u = proj(np.r_[P, tz + .095][None])[0]; print(n, np.round(u, 0).tolist())
    if np.all(np.isfinite(u)): cv2.putText(im, n, (int(np.clip(u[0], 0, 610)), int(np.clip(u[1], 12, 350))), 0, .45, (0, 0, 255), 1)
cv2.imwrite(W + "/dev_out/ep3_overlay.jpg", im)
