import os, sys, tempfile, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = tempfile.mkdtemp(dir=W + "/dev_out"); os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__)
class Side:
    def latest(s): return None, None, 0
class S:
    side = Side(); dry = True
    def __init__(s): s.sum = {}
    def log(s, m): pass
    def phase(s, m): pass
    def decide(s, m): pass
    def summary(s, **k): s.sum.update(k)
    def feedback(s): return dict(force=np.zeros(3), gripper=85.0)
class Zero:
    info = dict(source="zero")
    def act(s, o, explore, rng): return np.zeros(4), dict(mu=[0.0] * 4, std=[0.0] * 4)
w = dict(c_o=E.OPEN_C.copy(), yaw_o=18.3, face=0.0912, tab=dict(s=0.02, t=0.0)); g = dict(h_g=0.1392); pl = dict(off=np.array([-0.0193, -0.0064]), dyaw=-34.0, yaw_f=-71.7)
def first_contact(ep):
    os.environ["RLPJ_RL_MODE"], os.environ["RLPJ_EPISODE"] = "train", str(ep); e = E.Episode(S()); e.rng = np.random.default_rng(ep); e.actor = Zero(); e.tz = -0.0154; e.c0 = e.centre.copy(); e.save = lambda *a, **k: None
    st = dict(b=E.RIM + E.HOVER); cap = {}; got = {}; e.force = lambda: np.zeros(3); e.T_now = lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, e.tz + g["h_g"] + st["b"]], [0, 0, 0, 1.0]])
    orig = e.pose; e.pose = lambda w_, g_, pl_, c, b: (cap.update(c=np.array(c), b=b), orig(w_, g_, pl_, c, b))[1]
    def step(name, T):
        if cap["b"] < E.RIM + 0.001 and "c" not in got: got["c"] = cap["c"].copy(); st["b"] = E.RIM + 0.001; return "abort", dict(reason="end test")
        st["b"] = cap["b"]; return "ok", dict(max_force_n=1.0)
    e.try_step = step; e.insert(w, g, pl); return got["c"] - e.c0, e.arm
L = [first_contact(3 * k) for k in range(1, 301)]; X = np.array([x[0] for x in L]) * [1000, 1000, 1]
print("learned arm, residual at the first rim touch: sd a %.2f mm, b %.2f mm, yaw %.2f deg; share with b at least +1.0 mm: %.2f; max |a|,|b| %.1f, %.1f mm" % (X[:, 0].std(), X[:, 1].std(), X[:, 2].std(), (X[:, 1] >= 1.0).mean(), np.abs(X[:, 0]).max(), np.abs(X[:, 1]).max()), "| arm", L[0][1])
Y = np.array([first_contact(3 * k + 2)[0] for k in range(1, 31)]); print("scripted arm: max |residual|", float(np.abs(Y).max()))
