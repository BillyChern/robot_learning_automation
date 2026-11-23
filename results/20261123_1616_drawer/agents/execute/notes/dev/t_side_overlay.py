"""Attempt 26 at the last contact: planned drawer box and the measured opening drawn on the side image, 3x crop."""
import json, cv2, numpy as np, sys
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs"); import vision as V
d = W + "/inputs/calibration/"; Tb = np.array(json.load(open(d + "side_pose_935322070708.json"))["T_base_side"]); I = json.load(open(d + "intrinsics_935322070708_1280x720.json")); K, D = np.array(I["K"]), np.array(I["dist"])
def proj(P):
    Pc = (np.linalg.inv(Tb) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; return cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2) / 2.0
S = json.load(open(W + "/attempts/attempt_26/summary.json")); tz = -0.0154; c_o = np.array(S["objects"]["opening"]["c"]); y = S["objects"]["opening"]["yaw"]; a, b = V.axes(y); A, B = np.r_[a, 0], np.r_[b, 0]; Z = np.array([0, 0, 1.0])
ins = S["insert"]; c = np.array(ins["c"]); bot = ins["bottom"]; face = S["objects"]["drawer"]["face_h"]; phi = ins["tilt"]
tab_a = -0.0205; R = cv2.Rodrigues(B * np.radians(phi))[0]; cc = np.r_[c_o + c[0] * a + c[1] * b, tz]; piv = cc + tab_a * A + (bot + face + 0.004) * Z
def box(z, ha=0.0362, hb=0.0566):
    P = np.array([cc + sa * ha * A + sb * hb * B + z * Z for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))]); return (R @ (P - piv).T).T + piv
im = cv2.imread(W + "/attempts/attempt_26/side_inserted.jpg"); x0, y0, x1, y1 = 205, 120, 345, 275; s = 4
out = cv2.resize(im[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
def draw(P, col):
    cv2.polylines(out, [np.int32((proj(P) - [x0, y0]) * s)], True, col, 1, cv2.LINE_AA)
rim = np.array([np.r_[c_o + sa * 0.0378 * a + sb * 0.0575 * b, tz + 0.095] for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))]); draw(rim, (0, 255, 0))
draw(box(bot), (255, 0, 255)); draw(box(bot + face), (0, 255, 255))
cv2.imwrite(W + "/dev_out/ep26_side_overlay.jpg", out); print(out.shape, "bottom", bot, "tilt", phi, "c", c.tolist())
