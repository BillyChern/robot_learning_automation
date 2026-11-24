import json, os, sys, tempfile, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = tempfile.mkdtemp(dir=W + "/dev_out"); os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__); T = E.touch
f = lambda c: np.round(np.asarray(c) * [1000, 1000, 1], 3).tolist()
print("frozen", f(T.frozen_centre()), "= what r4 would load with no new entries", f(T.load_centre(D)), "| rev", E.SCRIPTED_REV)
class Side:
    def latest(s): return None, None, 0
class S:
    side = Side(); dry = False
    def __init__(s): s.sum = {}; s.dec = []
    def log(s, m): pass
    def phase(s, m): pass
    def decide(s, m): s.dec.append(m)
    def summary(s, **k): s.sum.update(k)
    def feedback(s): return dict(force=np.zeros(3), gripper=85.0)
w = dict(c_o=E.OPEN_C.copy(), yaw_o=18.3, face=0.0912, tab=dict(s=0.02, t=0.0)); g = dict(h_g=0.1392); pl = dict(off=np.array([-0.0193, -0.0064]), dyaw=-34.0, yaw_f=-71.7)
def run(ep, hole):
    os.environ["RLPJ_RL_MODE"], os.environ["RLPJ_EPISODE"] = "train", str(ep); s = S(); e = E.Episode(s); e.tz = -0.0154; e.c0 = e.centre.copy(); e.save = lambda *a, **k: None; st = dict(b=E.RIM + E.HOVER); cap = {}
    start = e.centre.copy(); e.force = lambda: np.zeros(3); e.T_now = lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, e.tz + g["h_g"] + st["b"]], [0, 0, 0, 1.0]])
    orig = e.pose; e.pose = lambda w_, g_, pl_, c, b: (cap.update(c=np.array(c), b=b), orig(w_, g_, pl_, c, b))[1]
    def step(name, Tm):
        c, b = cap["c"], cap["b"]
        if b < E.RIM + 0.001 and st["b"] >= E.RIM + 0.001 and abs(c[0] - hole) > 0.0018: st["b"] = E.RIM + 0.001 + 0.2 * abs(c[0] - hole); return "force", dict(reason="contact force (sim)")
        st["b"] = b; return "ok", dict(max_force_n=2.0)
    e.try_step = step; out = e.insert(w, g, pl); print(f"ep {ep} {e.arm:9s} start {f(start)} {out['end']} entry {s.sum['entry']['c']} touches {s.sum['entry']['rim_touches']} | wrote file: {os.path.exists(D + '/touch_centre.json')}")
for ep, h in ((15, 0.0030), (16, 0.0060), (17, 0.0030), (18, 0.0092)): run(ep, h)
