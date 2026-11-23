"""Offline flow test of the levelling: Episode with a fake session; keep-out and limit checks on the real model."""
import os, sys, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"
sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")   # no Kortex on this machine
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "execute_head", "exec"), E.__dict__)
from rlpj.robot import sim
from rlpj.robot.kin import Kin
from rlpj.robot.plan import Planner, SLOW
V, np_ = E.V, np
kin = Kin.load(); pl_ = Planner(kin)
class S:
    def __init__(s): s.moves, s.sum, s.q = [], {}, np.array([0, 25, -110, 0, -45, 90.0])
    def log(s, m): pass
    def summary(s, **k): s.sum.update(k)
    def feedback(s): return dict(q=s.q, pose=E.calib.T_to_pose(kin.fk(s.q)), force=np.zeros(3), gripper=85.0)
    def move_line(s, name, T, speed="slow", tip_margin=0.008):
        seg = pl_.cart_line(name, s.q, T, SLOW); ok, rep = sim.check(seg.q, kin, SLOW, gripper_frac=1.0, tip_margin=tip_margin)
        s.moves.append((name, bool(ok), rep.get("keep_out_violations"))); s.q = seg.q[-1].copy()
for sag, bias in ((-4.0, 1.4), (3.0, -1.0), (-0.3, 0.5), (-9.0, 0.0)):
    s = S(); e = E.Episode.__new__(E.Episode); e.s = s; e.tz = -0.0154; e.tilt = {}; e.phi = 0.0; e.R_fix = np.eye(3)
    w = dict(c_o=E.OPEN_C.copy(), yaw_o=18.3, face=0.0915); g = dict(h_g=0.1392); pl = dict(off=np.array([-0.0193, -0.0064]), dyaw=26.0, yaw_f=108.3)
    ax = V.axes(18.3)[0]; e.g_ref = np.tan(np.radians(bias)) * ax
    e.measure_held = lambda w_, b_: dict(ok=True, n=800, tilt_s_deg=float(np.degrees(np.arctan(np.tan(np.radians(sag - e.phi)) + np.tan(np.radians(bias))))), tilt_t_deg=0.0, z0=0.17)
    q0 = pl_.ik_exact(e.pose(w, g, pl, np.zeros(3), E.RIM + E.LEVEL_H), s.q)[0]; s.q = q0
    p0 = e.pose(w, g, pl, np.zeros(3), E.RIM + E.LEVEL_H); pin0 = p0[:3, 3] + p0[:3, :3] @ [0, 0, E.TIP_CLOSED]
    e.level(w, g, pl)
    p1 = e.pose(w, g, pl, np.zeros(3), E.RIM + E.LEVEL_H); pin1 = p1[:3, 3] + p1[:3, :3] @ [0, 0, E.TIP_CLOSED]
    for b in (E.RIM + E.HOVER, 0.05, E.B_MIN):
        s.move_line(f"down_{b:.3f}", e.pose(w, g, pl, np.zeros(3), b))
    print(f"sag {sag:+.1f} bias {bias:+.1f}: phi {e.phi:+.2f}, final rel {s.sum['tilt']['held'][-1]['rel_s']}, pinch moved {np.linalg.norm(pin1 - pin0) * 1000:.2f} mm,",
          "moves ok" if all(m[1] for m in s.moves) else [m for m in s.moves if not m[1]])
