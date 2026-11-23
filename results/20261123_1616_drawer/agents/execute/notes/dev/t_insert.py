"""Offline flow test of insert(): fake contact model (a hole at a hidden offset), zero policy, plus model checks at the bounds."""
import os, sys, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "execute_head", "exec"), E.__dict__)
from rlpj.robot import sim
from rlpj.robot.kin import Kin
from rlpj.robot.plan import Planner, SLOW
class Side:
    def latest(s): return None, None, 0
class S:
    side = Side()
    def log(s, m): pass
    def phase(s, m): pass
    def summary(s, **k): s.__dict__.setdefault("sum", {}).update(k)
    dry = True
    def feedback(s): return dict(force=np.zeros(3), gripper=85.0)
w = dict(c_o=E.OPEN_C.copy(), yaw_o=18.3, face=0.0915, tab=dict(s=0.02, t=0.0)); g = dict(h_g=0.1396); pl = dict(off=np.array([-0.0193, -0.0064]), dyaw=-34.0, yaw_f=-71.7)
def episode(hole, tol, seed):
    e = E.Episode.__new__(E.Episode); e.s = S(); e.tz = -0.0154; e.mode = "train"; e.rng = np.random.default_rng(seed); e.actor = E.rc.Actor("/nonexistent")
    e.trans, e.events, e.ret, e.phi, e.R_fix, e.tilt = [], [], 0.0, 0.0, np.eye(3), {}
    st = dict(b=E.RIM + E.HOVER, c=np.zeros(3))
    e.force = lambda: np.zeros(3)
    e.T_now = lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, e.tz + g["h_g"] + st["b"]], [0, 0, 0, 1.0]])
    captured = {}
    orig = e.pose
    def pose(w_, g_, pl_, c, b): captured.update(c=np.array(c), b=b); return orig(w_, g_, pl_, c, b)
    e.pose = pose
    def try_step(name, T):
        c, b = captured["c"], captured["b"]
        fits = np.all(np.abs(c - hole) < tol)
        if b < E.RIM + 0.001 and st["b"] >= E.RIM + 0.001 and not fits:
            st["b"] = E.RIM + 0.001; return "force", dict(reason="contact force (sim)")
        st["b"], st["c"] = b, c; return "ok", dict(max_force_n=2.0)
    e.try_step = try_step
    return e.insert(w, g, pl), len(e.trans)
tol = np.array([0.0012, 0.0010, 0.8])
for hole in ((0, 0, 0), (0.002, -0.0015, 0.7), (0.004, 0.002, -1.5)):
    R = [episode(np.array(hole), tol, i) for i in range(300)]
    print("hole", hole, "seated", round(np.mean([r[0]["end"] == "seated" for r in R]), 3), "ends", {k: sum(r[0]["end"] == k for r in R) for k in ("seated", "jam", "steps")}, "max steps", max(r[1] for r in R))
kin = Kin.load(); P = Planner(kin); e = E.Episode.__new__(E.Episode); e.tz = -0.0154; e.R_fix = np.eye(3)
bad = 0; q = np.array([0, 25, -110, 0, -45, 90.0])
for sa in (-1, 1):
    for sb in (-1, 1):
        for sy in (-1, 1):
            c = E.MAX_C * [sa, sb, sy]; q0 = P.ik_exact(e.pose(w, g, pl, c, E.RIM + E.HOVER), q)[0]
            seg = P.cart_line("t", q0, e.pose(w, g, pl, c, E.B_MIN), SLOW); bad += not sim.check(seg.q, kin, SLOW, gripper_frac=1.0, tip_margin=0.008)[0]
print("descents at the correction bounds refused by the model:", bad, "of 8")
