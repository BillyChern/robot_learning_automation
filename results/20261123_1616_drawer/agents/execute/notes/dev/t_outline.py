import os, sys, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__); V = E.V
y = 21.2; a, b = V.axes(y); c0 = np.asarray(E.CAB["centre"]); r = np.random.default_rng(0)
def cab(shift=0.0, filled=False):
    s, t = r.uniform(-.1375, .1375, 9000), r.uniform(-.0695, .0695, 9000)
    if filled: s = np.r_[s, r.uniform(-.1375, -.06, 9000)]; t = np.r_[t, r.uniform(-.0695, .0695, 9000)]
    xy = c0 + (s[:, None] + shift) * a + t[:, None] * b; P = np.c_[xy, np.full(len(s), .08)]
    return dict(points=P, n=len(s), height=.096, extent=[.28, .18], xy=P[:, :2].mean(0).tolist(), z_top=.08)
class S:
    def log(s, m): print("log", m)
e = E.Episode.__new__(E.Episode); e.s = S(); e.before = [cab()]
print("filled compartment: centroid moves %.1f cm, outline moves %.2f cm" % (np.linalg.norm(np.subtract(cab(filled=True)["xy"], e.before[0]["xy"])) * 100, e.cabinet_outline_move([cab(filled=True)]) * 100))
print("cabinet really moved 3 cm: outline reads %.2f cm" % (e.cabinet_outline_move([cab(shift=.03)]) * 100))
m = cab(); m["extent"] = [.44, .18]; e.before = [m]; print("merged blob before ->", e.cabinet_outline_move([cab()]))
