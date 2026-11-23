import os, sys, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
import touch, vision as V, cv2
from rlpj.robot.plan import tool_down_T
from rlpj.robot import calib, sim
from rlpj.robot.kin import Kin
from rlpj.robot.plan import Planner, SLOW
# 1. search on a contact model: height falls toward the true offset, the drawer drops within the clearance window
def run(true, seed, noise=0.0003):
    r = np.random.default_rng(seed); T = touch.TouchSearch(36); p = np.zeros(3); n = 0
    while p is not None:
        d = p - true; n += 1
        if abs(d[0]) < 0.0018 and abs(d[1]) < 0.0012 and abs(d[2]) < 1.0: return n
        T.report(p, 0.095 + 0.22 * abs(d[0]) + 0.15 * abs(d[1]) + 0.0002 * abs(d[2]) + r.normal(0, noise)); p = T.next()
    return None
for tr in ((0, 0, 0), (0.006, 0, 0), (0.009, -0.002, 0), (0.012, 0.004, 1.0), (-0.006, 0.006, -1.5), (0.014, -0.012, 0)):
    R = [run(np.array(tr), s) for s in range(200)]; ok = [x for x in R if x]
    print("offset", tr, "found", round(len(ok) / 200, 2), "median touches", int(np.median(ok)) if ok else None)
# 2. opening look: all four outer corners and the rim inside the image with the new aim
K, D, Ttc = V.wrist_cam(); tz = -0.0154; z = tz + 0.095; C = np.array([.4519, .0029]); a, b = V.axes(18.3)
for aim in (np.zeros(2), np.array([0, -0.028])):
    X = C + aim; T = tool_down_T(np.r_[X, tz + .30], 0.0); Tc = T @ Ttc; o, zc = Tc[:3, 3], Tc[:3, 2]; T[:2, 3] += X - (o + (z - o[2]) / zc[2] * zc)[:2]
    P = np.array([np.r_[C + sa * 0.0462 * a + sb * 0.0695 * b, z] for sa in (-1, 1) for sb in (-1, 1)])
    uv = calib.project(calib.apply(np.linalg.inv(T @ Ttc), P), K, D); print("aim", aim.tolist(), "outer corners u", np.round(uv[:, 0]).tolist(), "v", np.round(uv[:, 1]).tolist())
# 3. model check of descents over the whole search range
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__)
kin = Kin.load(); P_ = Planner(kin); e = E.Episode.__new__(E.Episode); e.tz = tz; e.R_fix = np.eye(3); q = np.array([0, 25, -110, 0, -45, 90.0])
w = dict(c_o=E.OPEN_C.copy(), yaw_o=18.3, face=0.0915); g = dict(h_g=0.1396); pl = dict(off=np.array([-0.0193, -0.0064]), dyaw=26.0, yaw_f=-71.7); bad = []
for ca in (-0.018, -0.009, 0, 0.009, 0.018):
    for cb in (-0.018, 0, 0.018):
        for cy in (-3.0, 0, 3.0):
            c = np.array([ca, cb, cy]); q0 = P_.ik_exact(e.pose(w, g, pl, c, E.RIM + E.HOVER), q)[0]
            seg = P_.cart_line("t", q0, e.pose(w, g, pl, c, E.B_MIN), SLOW); ok, rep = sim.check(seg.q, kin, SLOW, gripper_frac=1.0, tip_margin=0.008)
            if not ok: bad.append((ca, cb, cy, rep.get("keep_out_violations")))
print("touch descents refused by the model:", len(bad), "of 45", bad[:4])
