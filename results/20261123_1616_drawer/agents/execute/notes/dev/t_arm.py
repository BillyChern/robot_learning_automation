import os, sys, tempfile, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = tempfile.mkdtemp(dir=W + "/dev_out"); os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__); V = E.V
class Side:
    def latest(s): return None, None, 0
class Wr:
    def __init__(s): s.n = 0
    def latest(s): s.n += 1; return np.zeros((720, 1280, 3), np.uint8), 0
class S:
    side = Side(); dry = True
    def __init__(s): s.sum = {}; s.wrist = Wr()
    def log(s, m): pass
    def phase(s, m): pass
    def summary(s, **k): s.sum.update(k)
    def feedback(s): return dict(force=np.zeros(3), gripper=85.0)
class FakeActor:
    info = dict(source="fake")
    def act(s, o, explore, rng): return np.array([0.7, -0.7, 0.7, 0.2]), dict(mu=[.7, -.7, .7, .2], std=[.3] * 4)
w = dict(c_o=E.OPEN_C.copy(), yaw_o=18.3, face=0.0915, tab=dict(s=0.02, t=0.0)); g = dict(h_g=0.1396); pl = dict(off=np.array([-0.0193, -0.0064]), dyaw=-34.0, yaw_f=-71.7)
def run(mode, ep):
    os.environ["RLPJ_RL_MODE"], os.environ["RLPJ_EPISODE"] = mode, str(ep)
    s = S(); e = E.Episode(s); e.actor = FakeActor(); e.tz = -0.0154; e.c0 = np.zeros(3); e.save = lambda *a, **k: None; st = dict(b=E.RIM + E.HOVER); cap = {}
    e.force = lambda: np.zeros(3); e.T_now = lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, e.tz + g["h_g"] + st["b"]], [0, 0, 0, 1.0]])
    orig = e.pose; e.pose = lambda w_, g_, pl_, c, b: (cap.update(c=np.array(c), b=b), orig(w_, g_, pl_, c, b))[1]
    def step(name, T): st["b"] = cap["b"]; return "ok", dict(max_force_n=2.0)
    e.try_step = step; out = e.insert(w, g, pl); A = np.array([x["act"] for x in e.trans])
    print(f"{mode:5s} ep {ep:3d} arm {e.arm:12s} end {out['end']} transitions {len(e.trans)} max|action| {np.abs(A).max():.2f} final residual mm {np.round(out['c'][:2] * 1000, 2).tolist()}")
for m, ep in (("train", 7), ("train", 8), ("train", 11), ("train", 12), ("train", -1), ("eval", 8)): run(m, ep)
os.environ["RLPJ_RL_MODE"], os.environ["RLPJ_EPISODE"] = "train", "9"; s = S(); s.dry = False; e = E.Episode(s); e.save = lambda *a, **k: None; calls = []
kd, kt = V.find_drawer, V.find_tab; V.find_drawer = lambda *a, **k: dict(ok=True); V.find_tab = lambda *a, **k: (calls.append(1), dict(ok=len(calls) >= 2, why="metal too small"))[1]
d, tab = e.pull_detect(np.zeros((720, 1280, 3), np.uint8), np.eye(4), [0.3, 0], 0.08); print("blurred frame, smoothed copy works: ok", tab["ok"], "tries", s.sum["pull_detect_tries"]); calls.clear()
V.find_tab = lambda *a, **k: (calls.append(1), dict(ok=False, why="x"))[1]; d, tab = e.pull_detect(np.zeros((720, 1280, 3), np.uint8), np.eye(4), [0.3, 0], 0.08); print("never found: ok", tab["ok"], "tries", s.sum["pull_detect_tries"]); V.find_drawer, V.find_tab = kd, kt
