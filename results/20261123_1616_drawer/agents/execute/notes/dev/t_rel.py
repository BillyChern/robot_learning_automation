"""drawer_vs_opening on recorded wrist frames at the 2 cm level height (dashboard.mp4), commanded tool pose."""
import json, re, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot.plan import tool_down_T
ts = lambda l: sum(int(x) * m for x, m in zip(re.match(r"\[(\d+):(\d+):(\d+)\]", l).groups(), (3600, 60, 1)))
wrap = lambda a: (a + 180.0) % 360.0 - 180.0
for n in (16, 18, 20):
    A = f"{W}/attempts/attempt_{n}/"; S = json.load(open(A + "summary.json")); O = S["objects"]; L = open(A + "log.txt").read().splitlines()
    lv = [l for l in L if "exec" in l and "to_level" in l][0]
    c_o = np.array(O["opening"]["c"]); yo = O["opening"]["yaw"]; d = O["drawer"]; tab = O["tab"]; off = np.array(tab["xy"]) - np.array(d["c"])
    dy = wrap(yo + 180.0 - (d["yaw"] - 90.0 + (0.0 if tab["side"] > 0 else 180.0))); r = np.radians(dy)
    xy = c_o + np.array([np.cos(r) * off[0] - np.sin(r) * off[1], np.sin(r) * off[0] + np.cos(r) * off[1]])
    tz, h = -0.0154, S["grasp"]["h_g"]; T = tool_down_T(np.r_[xy, tz + h + 0.095 + 0.020], d["yaw"] + dy)
    v = cv2.VideoCapture(A + "dashboard.mp4"); fps = v.get(5)
    for dt in (0.4, 0.9):
        v.set(cv2.CAP_PROP_POS_FRAMES, int((ts(lv) - ts(L[0]) + dt) * fps)); ok, f = v.read()
        im = cv2.resize(f[:360, 640:], (1280, 720), interpolation=cv2.INTER_CUBIC)
        m = V.drawer_vs_opening(im, T, c_o, yo, tz + 0.095, tz + 0.095 + 0.020 + d["face_h"])
        k = m["cabinet"]; q = m["drawer"]
        print(n, dt, "ok", m["ok"], "ea/eb/eyaw", m.get("ea"), m.get("eb"), m.get("eyaw"), "| cab", {x: k.get(x) for x in ("ok", "a_out", "cb", "width_b", "why")}, "| drawer", {x: q.get(x) for x in ("ok", "da", "db", "dyaw")})
