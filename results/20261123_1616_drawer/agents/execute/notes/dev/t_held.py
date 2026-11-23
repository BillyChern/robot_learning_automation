"""held_offset on the recorded wrist frames at the hover (dashboard.mp4, right half, 640x360 of the 1280x720 image)."""
import json, re, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot.plan import tool_down_T
ts = lambda l: sum(int(x) * m for x, m in zip(re.match(r"\[(\d+):(\d+):(\d+)\]", l).groups(), (3600, 60, 1)))
wrap = lambda a: (a + 180.0) % 360.0 - 180.0
for n in (10, 12, 14, 16, 18):
    A = f"{W}/attempts/attempt_{n}/"; S = json.load(open(A + "summary.json")); O = S["objects"]; L = open(A + "log.txt").read().splitlines()
    hov = [l for l in L if "exec" in l and "to_hover" in l][0]
    c_o = np.array(O["opening"]["c"]); yo = O["opening"]["yaw"]; d = O["drawer"]; tab = O["tab"]; off = np.array(tab["xy"]) - np.array(d["c"])
    dy = wrap(yo + 180.0 - (d["yaw"] - 90.0 + (0.0 if tab["side"] > 0 else 180.0)))
    r = np.radians(dy); xy = c_o + np.array([np.cos(r) * off[0] - np.sin(r) * off[1], np.sin(r) * off[0] + np.cos(r) * off[1]])
    tz, h = -0.0154, S["grasp"]["h_g"]; T = tool_down_T(np.r_[xy, tz + h + 0.103], d["yaw"] + dy)
    v = cv2.VideoCapture(A + "dashboard.mp4"); fps = v.get(5); out = []
    for dt in (0.4, 0.8, 1.2):
        v.set(cv2.CAP_PROP_POS_FRAMES, int((ts(hov) - ts(L[0]) + dt) * fps)); ok, f = v.read()
        if not ok: continue
        im = cv2.resize(f[:360, 640:], (1280, 720), interpolation=cv2.INTER_CUBIC)
        m = V.held_offset(im, T, c_o, yo, tz + 0.103 + d["face_h"]); out.append(m)
    print(n, "tool yaw", round(d["yaw"] + dy, 1), [{k: x.get(k) for k in ("ok", "da", "db", "dyaw", "width_b", "resid_mm", "why", "n") if x.get(k) is not None} for x in out])
