import json, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot.kin import Kin
kin = Kin.load(); tz = -0.0154
for n in (36, 38):
    A = f"{W}/attempts/attempt_{n}/"; S = json.load(open(A + "summary.json")); dr = S["objects"]["drawer"]; face = tz + dr["face_h"]
    T = kin.fk(np.load(A + "plans/02_look_drawer.npz")["q"][-1]); im = cv2.resize(cv2.imread(A + "wrist_drawer.jpg"), (1280, 720), interpolation=cv2.INTER_CUBIC)
    d = V.find_drawer(im, T, np.array(dr["c"]), face); print(n, "drawer", d)
    half = (d["size"][0] / 2 - 0.003, d["size"][1] / 2 - 0.003); R, valid = V.rectify(im, T, d["c"], d["yaw"] - 90.0, half, face)
    hsv = cv2.cvtColor(R, cv2.COLOR_BGR2HSV); grey = ((hsv[..., 1] < 95) | ((hsv[..., 1] < 135) & (hsv[..., 2] < 120))) & (hsv[..., 2] > 25) & valid
    print("  rect", R.shape, "valid", round(float(valid.mean()), 3), "grey share", round(float(grey.mean()), 3), "median S/V on face", np.median(hsv[..., 1][valid]), np.median(hsv[..., 2][valid]))
    m = cv2.morphologyEx(cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8); o = np.argsort(-st[1:, 4])[:3] + 1
    print("  largest grey comps (x,y,w,h,area):", st[o].tolist(), "col max of largest", int((lab == o[0]).sum(0).max()) if len(o) else None, "needed", int(0.03 / V.RES))
    print("  tab", V.find_tab(im, T, d, face))
    cv2.imwrite(W + f"/dev_out/rect_{n}.jpg", np.hstack([R, cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR)]))
