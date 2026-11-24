"""Offline flow test of insert(): fake contact model (a hole at a hidden offset), zero policy, plus model checks at the bounds."""
import os, sys, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "execute_head", "exec"), E.__dict__)
from rlpj.robot import sim
V = E.V
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
    e = E.Episode.__new__(E.Episode); e.s = S(); e.tz = -0.0154; e.mode = MODE; e.arm = "learned" if MODE == "train" else "learned_eval"; e.rng = np.random.default_rng(seed); e.actor = E.rc.Actor("/nonexistent")
    e.trans, e.events, e.ret, e.phi, e.R_fix, e.tilt, e.contact_map = [], [], 0.0, 0.0, np.eye(3), {}, []
    e.c0 = np.zeros(3)
    st = dict(b=E.RIM + E.HOVER, c=np.zeros(3))
    e.force = lambda: np.zeros(3)
    e.T_now = lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, e.tz + g["h_g"] + st["b"]], [0, 0, 0, 1.0]])
    captured = {}
    orig = e.pose
    def pose(w_, g_, pl_, c, b): captured.update(c=np.array(c), b=b); return orig(w_, g_, pl_, c, b)
    e.pose = pose
    def try_step(name, T):
        c, b = captured["c"], captured["b"]
        fits = np.all(np.abs(c - hole) < tol) and abs(e.phi - TRUE_TILT) < 1.2
        if b < E.RIM + 0.001 and st["b"] >= E.RIM + 0.001 and not fits:
            st["b"] = E.RIM + 0.001 + 0.22 * abs(c[0] - hole[0]) + 0.15 * abs(c[1] - hole[1]); return "force", dict(reason="contact force (sim)")
        st["b"], st["c"] = b, c; return "ok", dict(max_force_n=2.0)
    e.try_step = try_step
    out = e.insert(w, g, pl); LAST.update(events=e.events); return out, len(e.trans)
tol = np.array([0.0018, 0.0010, 0.8]); TRUE_TILT = 0.0; MODE = "eval"; LAST = {}
for hole, TRUE_TILT in (((0, 0, 0), 0.0), ((0.003, 0.002, 0), 0.0), ((-0.0055, 0.0035, 0), 0.0), ((0.006, -0.004, 0), 0.0), ((0.0095, 0, 0), 0.0)):
    R = [episode(np.array(hole), tol, i) for i in range(300)]
    print("hole", hole, "true tilt", TRUE_TILT, "seated", round(np.mean([r[0]["end"] == "seated" for r in R]), 3), "ends", {k: sum(r[0]["end"] == k for r in R) for k in ("seated", "jam", "steps")}, "max steps", max(r[1] for r in R))
MODE = "train"; episode(np.zeros(3), np.array([0.01, 0.01, 3.0]), 3); inside = [ev[5] for ev in LAST["events"] if ev[4] <= E.RIM - E.ENTERED]
print("train mode: lateral drift after entering the opening (mm):", round(float(np.abs(np.array(inside)[:, :2] - np.array(inside)[0, :2]).max()) * 1000, 3), "over", len(inside), "steps"); MODE = "eval"
kin = Kin.load(); P = Planner(kin); e = E.Episode.__new__(E.Episode); e.tz = -0.0154; e.R_fix = np.eye(3)
bad = 0; q = np.array([0, 25, -110, 0, -45, 90.0])
for sa in (-1, 1):
    for sb in (-1, 1):
        for sy in (-1, 1):
            c = E.MAX_C * [sa, sb, sy]; q0 = P.ik_exact(e.pose(w, g, pl, c, E.RIM + E.HOVER), q)[0]
            seg = P.cart_line("t", q0, e.pose(w, g, pl, c, E.B_MIN), SLOW); bad += not sim.check(seg.q, kin, SLOW, gripper_frac=1.0, tip_margin=0.008)[0]
print("descents at the correction bounds refused by the model:", bad, "of 8")
bad = 0
for phi in (-5.0, -3.0, -1.5, 2.5):
    e.phi = phi; e.R_fix = E.cv2.Rodrigues(np.r_[V.axes(18.3)[1], 0.0] * np.radians(phi))[0]
    for ca in (-0.004, 0.0, 0.004):
        c = np.array([ca, -0.005, 0.5]); q0 = P.ik_exact(e.pose(w, g, pl, c, E.RIM + E.HOVER), q)[0]
        seg = P.cart_line("t", q0, e.pose(w, g, pl, c, E.B_MIN), SLOW); bad += not sim.check(seg.q, kin, SLOW, gripper_frac=1.0, tip_margin=0.008)[0]
e.R_fix = np.eye(3)
print("tilted descents refused by the model:", bad, "of 12")

# align_fix: converges on a measured error, stops on a rejected measurement; grasp yaw keeps the camera on the -a side
class Wrist:
    def latest(s): return np.zeros((720, 1280, 3), np.uint8), 0
for e0 in (np.array([0.002, 0.004, -0.7]), np.array([0.014, -0.012, 6.0]), None):
    s = S(); s.dry = False; s.wrist = Wrist(); got = []
    e = E.Episode.__new__(E.Episode); e.s = s; e.tz = -0.0154; e.R_fix = np.eye(3); e.c0 = np.zeros(3); e.centre = np.zeros(3); e.save = lambda *a, **k: None
    e.T_now = lambda: np.eye(4); s.move_line = lambda name, T, speed="slow", tip_margin=0.008: got.append(name)
    def meas(*a, **k):
        if e0 is None: return dict(ok=False, drawer=dict(why="edge not found"), cabinet={})
        x = e0 + e.c0; return dict(ok=True, ea=float(x[0]), eb=float(x[1]), eyaw=float(x[2]), drawer={}, cabinet={})
    keep = E.V.drawer_vs_opening; E.V.drawer_vs_opening = meas; e.align_fix(w, g, pl); E.V.drawer_vs_opening = keep
    print("error", None if e0 is None else e0.tolist(), "-> c0", np.round(e.c0, 4).tolist(), "moves", got, "left", s.sum["align"][-1]["ea"], s.sum["align"][-1]["eb"], s.sum["align"][-1]["eyaw"])
from rlpj.robot.plan import tool_down_T
for dyw, side in ((82.3, -1), (-37.6, 1), (60.0, 1), (-60.0, -1)):
    W2 = dict(yaw_o=18.3, d=dict(yaw=dyw), tab=dict(side=side)); turn = E.Episode.drawer_turn(W2); y = E.wrapd(18.3 - 90 - turn)
    a_ = V.axes(dyw - 90)[0] * side * 0.0195; xy = np.array([0.29, -0.01]) + a_
    q1 = P.ik_exact(tool_down_T(np.r_[xy, -0.0154 + 0.22], y), q)[0]; q2 = P.ik_exact(tool_down_T(np.r_[xy, -0.0154 + 0.14], y), q1)[0]
    home = np.array([0, 15, -130, 0, 55, 90.0]); ok1 = sim.check(P.joint_move("j", home, q1, E.CARRY).q, kin, E.CARRY)[0]; ok2 = sim.check(P.cart_line("l", q1, tool_down_T(np.r_[xy, -0.0154 + 0.14], y), SLOW).q, kin, SLOW)[0]
    print(f"drawer yaw {dyw} tab {side}: turn {turn:.1f}, grasp yaw {y:.1f}, final {E.wrapd(y + turn):.1f}, jaw along long axis {abs(E.wrapd(2 * (y - dyw))) < 1e-6}, moves ok {bool(ok1)}, {bool(ok2)}")
