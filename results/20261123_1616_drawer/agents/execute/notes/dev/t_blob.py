import os, sys, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__)
from rlpj.success import checks
fx = E.CAB; y = np.radians(fx["yaw_deg"]); a, b = np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)]); r = np.random.default_rng(0)
def cab():
    s, t = r.uniform(-.03, .03, 12000), r.uniform(-.05, .05, 12000); xy = np.asarray(fx["target_centre"]) + s[:, None] * a + t[:, None] * b
    return dict(points=np.c_[xy, -0.0154 + 0.097 + r.normal(0, .001, 12000)], z_top=-0.0154 + 0.097, height=0.097, xy=[0.5, 0.01], extent=[0.28, 0.18], n=12000)
def blob(x, y_, h, ex, n): return dict(points=np.c_[r.normal(x, .01, n), r.normal(y_, .01, n), np.full(n, -0.0154 + h)], z_top=-0.0154 + h, height=h, xy=[x, y_], extent=[ex, ex], n=n)
f = lambda L: checks.drawer_lying([], [q for q in L if q["n"] >= E.MIN_DRAWER_POINTS], E.TASK)
for name, L in (("seated", [cab()]), ("seated + episode 21's noise patch", [cab(), blob(0.693, 0.151, 0.059, 0.094, 344)]), ("drawer really on the table", [cab(), blob(0.29, -0.01, 0.097, 0.114, 4200)])):
    o, rep = f(L); raw = checks.drawer_lying([], L, E.TASK)[0]; print(f"{name:36s} filtered {o} ({rep.get('reason')}) | unfiltered {raw}")
