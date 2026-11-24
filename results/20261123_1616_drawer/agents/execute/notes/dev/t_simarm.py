"""Sim arm against a contact model with the real frames: rim stop, up-force, sideways force toward the true opening. Checks signs end to end."""
import json, os, sys, tempfile, types, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; D = tempfile.mkdtemp(dir=W + "/dev_out"); os.environ["RLPJ_RL_DIR"] = D; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__); V = E.V
from rlpj.robot.plan import tool_down_T
class Side:
    def latest(s): return np.zeros((360, 640, 3), np.uint8), None, 0
class S:
    side = Side(); dry = False
    A = D
    def __init__(s): s.sum = {}; s.logs = []
    def log(s, m): s.logs.append(m)
    def phase(s, m): pass
    def decide(s, m): pass
    def summary(s, **k): s.sum.update(k)
    def feedback(s): return dict(gripper=84.5)
tz = -0.0154; yo = 18.3; ax, bx = V.axes(yo)
def run(err_a, err_b, sign=1.0, seed=0, clear=0.0018):
    os.environ["RLPJ_RL_MODE"], os.environ["RLPJ_EPISODE"] = "train", "31"; s = S(); e = E.Episode(s); assert e.arm == "sim"
    e.tz, e.c0 = tz, e.centre.copy(); e.save = lambda *a, **k: None; time_sleep = E.time.sleep; E.time.sleep = lambda x: None; r = np.random.default_rng(seed)
    w = dict(c_o=E.OPEN_C.copy(), yaw_o=yo, face=0.0912); g = dict(h_g=0.1392); dy = 21.4; off = E.rot2(np.array([0.0195, 0.002]), 180 + 0) * 1.0; pl = dict(off=off, dyaw=dy, yaw_f=yo - 90.0)
    true_c = w["c_o"] + e.c0[0] * ax + e.c0[1] * bx + err_a * ax + err_b * bx; T0 = e.pose(w, g, pl, e.c0, E.RIM + E.HOVER); st = dict(T=T0.copy(), f=np.array([1.0, -0.5, 2.0]))
    f_base = st["f"].copy(); d_tool = -E.rot2(pl["off"], pl["dyaw"] + e.c0[2])
    e.T_now = lambda: st["T"].copy(); e.force = lambda: st["f"] + r.normal(0, 0.5, 3)
    def step(name, T, limit=None):
        drawer = T[:2, 3] + d_tool; err = true_c - drawer; ea, eb = err @ ax, err @ bx; bot = T[2, 3] - tz - g["h_g"]; rim_b = E.RIM + 0.0015
        inside = abs(ea) < clear and abs(eb) < clear
        floor = 0.012; stop = None
        was_in = st["T"][2, 3] - tz - g["h_g"] < rim_b - 0.001
        if was_in and not inside:                       # below the rim the walls hold the drawer: the tool cannot leave the window
            ka, kb = np.clip(ea, -clear, clear) - ea, np.clip(eb, -clear, clear) - eb; T = T.copy(); T[:2, 3] -= ka * ax + kb * bx; inside = True; bot = T[2, 3] - tz - g["h_g"]
        if not inside and bot < rim_b: stop = rim_b
        if inside and bot < floor: stop = floor
        if stop is not None:
            Tn = T.copy(); Tn[2, 3] = tz + g["h_g"] + stop; st["T"] = Tn
            lat = np.array([ea, eb]); lat = lat / (np.linalg.norm(lat) + 1e-9); fxy = 10.0 * sign * (lat[0] * ax + lat[1] * bx) if not inside else np.zeros(2); st["f"] = f_base + np.r_[fxy, 32.0]
            return "force", dict(reason=f"contact force {np.linalg.norm(st['f'] - f_base) + 1:.1f} N total > 35 N")
        st["T"] = T.copy(); st["f"] = f_base.copy(); return "ok", dict(max_force_n=1.5)
    e.try_step = step; out = e.insert_sim(w, g, pl); E.time.sleep = time_sleep; h = e.sim_head; first = e.sim_log[0]["obs"]
    return out, h, first, e, s
print("start obs (real frames): ", {k: run(0, 0)[2][k] for k in ("x", "y", "yaw", "z", "off", "yaw_est", "tilt")})
for ea, eb in ((0, 0), (0.003, 0), (0.004, -0.002), (-0.004, 0.003), (0.007, 0.002), (-0.008, -0.004), (0.012, 0.0)):
    R = [run(ea, eb, seed=k) for k in range(8)]; ends = [x[1]["end"] for x in R]; print(f"true error a {ea*1000:+.0f} b {eb*1000:+.0f} mm: seated {ends.count('seated')}/8, ends {sorted(set(ends))}, decisions {int(np.median([x[1]['decisions'] for x in R]))}, stops {int(np.median([x[1]['stops'] for x in R]))}, away {max(x[1]['steps_against_force'] for x in R)}")
o, h, _, e, s = run(0.006, 0.0, sign=-1.0); print("force sign flipped (misbehaviour guard): end", h["end"], "decisions", h["decisions"], "stops", h["stops"], "away", h["steps_against_force"])
e.terminal = dict(bonus=0, R=0); e._saved = False; e.save_transitions(False, dict(reason="t")); print("files:", sorted(os.listdir(D)), sorted(os.listdir(D + "/sim_arm")), "| transitions dir exists:", os.path.exists(D + "/transitions"), "| log keys", sorted(e.sim_log[0]))
for ep, m, want in ((30, "train", "learned"), (31, "train", "sim"), (32, "train", "scripted"), (34, "train", "sim"), (35, "train", "scripted"), (43, "train", "sim"), (44, "train", "scripted"), (46, "train", "learned"), (31, "eval", "learned_eval")):
    os.environ["RLPJ_RL_MODE"], os.environ["RLPJ_EPISODE"] = m, str(ep); os.environ["RLPJ_ARM"] = "sim"; a = E.Episode(S()).arm; print(ep, m, a, "ok" if a == want else "WRONG", end=" | ")
os.environ["RLPJ_EPISODE"] = "-1"; os.environ["RLPJ_RL_MODE"] = "eval"; print("\ntest run with RLPJ_ARM=sim ->", E.Episode(S()).arm); os.environ.pop("RLPJ_ARM"); print("test run without it ->", E.Episode(S()).arm)
