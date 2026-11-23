"""Tilt of the top face about b from the convergence of the long edges: repeatability on saved and recorded frames, and the
zero reference on a drawer standing on the table (same function, look-down at the drawer)."""
import json, re, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot.plan import tool_down_T
ts = lambda l: sum(int(x) * m for x, m in zip(re.match(r"\[(\d+):(\d+):(\d+)\]", l).groups(), (3600, 60, 1)))
wrap = lambda a: (a + 180.0) % 360.0 - 180.0
def rot(v, deg):
    r = np.radians(deg); return np.array([np.cos(r) * v[0] - np.sin(r) * v[1], np.sin(r) * v[0] + np.cos(r) * v[1]])
tz = -0.0154
def geom(S):
    O = S["objects"]; d, tab = O["drawer"], O["tab"]; off = np.array(tab["xy"]) - np.array(d["c"])
    c_o, yo = np.array(O["opening"]["c"]), O["opening"]["yaw"]
    return c_o, yo, d, off, wrap(yo + 180.0 - (d["yaw"] - 90.0 + (0.0 if tab["side"] > 0 else 180.0)))
print("held, attempt 24 (saved 640 px, 3 looks at the level height):")
A = W + "/attempts/attempt_24/"; S = json.load(open(A + "summary.json")); c_o, yo, d, off, dy = geom(S); a, b = V.axes(yo)
for k, h in enumerate(S["align"]):
    c = np.array(h["c0"]); T = tool_down_T(np.r_[c_o + c[0] * a + c[1] * b + rot(off, dy + c[2]), tz + S["grasp"]["h_g"] + 0.095 + 0.025], yo - 90.0 + c[2])
    im = cv2.resize(cv2.imread(A + f"wrist_align{k}.jpg"), (1280, 720), interpolation=cv2.INTER_CUBIC)
    m = V.held_offset(im, T, c_o, yo, tz + 0.095 + 0.025 + d["face_h"]); print("  look", k, {x: m.get(x) for x in ("ok", "tilt_s", "da", "db", "dyaw", "width_b")})
print("held, recorded frames at the level height:")
for n in (16, 18, 20):
    A = f"{W}/attempts/attempt_{n}/"; S = json.load(open(A + "summary.json")); c_o, yo, d, off, dy = geom(S); L = open(A + "log.txt").read().splitlines()
    lv = [l for l in L if "exec" in l and "to_level" in l][0]; T = tool_down_T(np.r_[c_o + rot(off, dy), tz + S["grasp"]["h_g"] + 0.095 + 0.020], d["yaw"] + dy)
    v = cv2.VideoCapture(A + "dashboard.mp4"); out = []
    for dt in (0.4, 0.7, 1.0):
        v.set(cv2.CAP_PROP_POS_FRAMES, int((ts(lv) - ts(L[0]) + dt) * v.get(5))); ok, f = v.read()
        out.append(V.held_offset(cv2.resize(f[:360, 640:], (1280, 720), interpolation=cv2.INTER_CUBIC), T, c_o, yo, tz + 0.095 + 0.020 + d["face_h"]).get("tilt_s"))
    print("  attempt", n, "tilt_s", out)
print("standing on the table (zero reference), attempt 4 look-downs at the drawer, jaws-free view:")
A = W + "/attempts/attempt_4/"; vw = json.load(open(A + "views.json")); face = tz + 0.0915
for nm in ("drawer_high", "drawer_low"):
    im = cv2.imread(A + nm + ".jpg"); T = np.array(vw[nm]["T"]); o = vw[nm]["crop_xy"]; q = V.find_drawer(im, T, np.array([0.2855, -0.0055]), face, origin=o)
    R, val = V.rectify(im, T, q["c"], q["yaw"] - 90.0, (0.060, 0.075), face, o); w = V.wood_mask(R); Hh, Ww = w.shape; cs, ct = (Ww - 1) / 2, (Hh - 1) / 2
    pm, pp = [], []
    for x in np.arange(-0.030, 0.0301, 0.001):
        c = int(round(cs + x / V.RES)); k1 = V._outer(w[:, c], val[:, c]); k2 = V._outer(w[:, c], val[:, c], reverse=True)
        if k1 is not None and k1 < ct: pm.append((x, (k1 - ct) * V.RES))
        if k2 is not None and k2 > ct: pp.append((x, (k2 - ct) * V.RES))
    Lm, Lp = V._line(pm), V._line(pp)
    dep = (T @ V.wrist_cam()[2])[2, 3] - face
    print("  ", nm, "tilt_s", None if Lm is None or Lp is None else round(float(np.degrees(np.arctan((Lp["slope"] - Lm["slope"]) * dep / (Lp["off"] - Lm["off"])))), 2))
