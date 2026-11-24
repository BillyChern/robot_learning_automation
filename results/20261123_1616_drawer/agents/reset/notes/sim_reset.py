"""Offline end-to-end test of programs/reset.py: the real program logic against a fake session (model kinematics, the
session's own sim.check with the task's keep-out boxes) and a simulated drawer. No robot, no cameras.
usage: sim_reset.py [n per axis]   (targets over the task's current layout range)"""
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "programs"))
from rlpj import config
from rlpj.robot import calib, sim
from rlpj.robot.kin import Kin
from rlpj.robot.plan import NORMAL, SLOW, Planner

he = json.load(open(ROOT + "/inputs/calibration/handeye_kinova.json"))
fi = json.load(open(ROOT + "/inputs/calibration/intrinsics_kinova_1280x720.json"))["factory"]
TTC = np.array(he["T_tool_cam"])
KK = np.array([[fi["fx"], 0, fi["cx"]], [0, fi["fy"], fi["cy"]], [0, 0, 1.0]])
calib.T_tool_cam = lambda: TTC
calib.wrist_intrinsics = lambda: (KK, np.array(fi["coeffs"], float))
TASK = config.load("drawer_lying")["task"]
BOXES = tuple((b["name"], np.array(b["center"], float), np.array(b["half"], float), np.radians(b["yaw_deg"])) for b in TASK["keep_out"])
sim.cell_keep_out = lambda: BOXES

import types  # noqa: E402


class Abort(Exception):
    pass


_stub = types.ModuleType("rlpj.robot.session")          # the real session needs Kortex, absent on this machine
_stub.Abort, _stub.RobotSession = Abort, object
sys.modules["rlpj.robot.session"] = _stub
import reset as R  # noqa: E402

KIN = Kin.load()
TABLE_FACE, COMP_C = R.DRAWER_L, R.COMP_0


class World:
    def __init__(self, state, c, ax):
        self.state, self.c, self.ax, self.held, self.rel, self.ax_rel = state, np.array(c, float), np.array(ax, float), False, None, None
        self.dropped, self.tz2, self.tilt_err = None, None, 0.0

    @property
    def knob(self):
        return self.c + R.KNOB_OFF * self.ax


class Sess:
    dry, wd, executor, A = False, None, None, tempfile.gettempdir()

    def __init__(self, w):
        self.w, self.kin, self.planner = w, KIN, Planner(KIN, None, log=lambda m: None)
        self.q = R.HOME_Q.copy()
        self.q_home, self.g, self.gf, self.out, self.S, self.nseg = R.HOME_Q.copy(), 0.9, 0.0, "fail", {}, 0

    def log(self, m): pass
    def phase(self, m): pass
    def decide(self, m): self.S.setdefault("decisions", []).append(m)
    def summary(self, **k): self.S.update(k)
    def claim(self, o): self.out = o

    def require(self, c, m):
        if not c:
            raise Abort(m)

    def feedback(self):
        return dict(q=self.q.copy(), pose=calib.T_to_pose(KIN.fk(self.q)), gripper=self.g)

    def _run(self, seg, tip=0.008):
        self.nseg += 1
        ok, rep = sim.check(seg.q, KIN, seg.speed, gripper_frac=self.gf, tip_margin=tip, table_z=calib.table_z())
        if not ok:
            bad = {k: rep[k] for k in ("joint_limit_violations", "keep_out_violations", "keep_in_violations", "near_base", "min_tip_above_table") if k in rep}
            raise Abort(f"model check failed for {seg.name}: {bad}")
        self.q = seg.q[-1].copy()

    def move_joint(self, name, q, speed="normal"):
        self._run(self.planner.joint_move(name, self.q, np.asarray(q, float), NORMAL if speed == "normal" else SLOW))

    def move_line(self, name, T, speed="slow", tip_margin=0.008):
        self._run(self.planner.cart_line(name, self.q, np.asarray(T, float), NORMAL if speed == "normal" else SLOW), tip_margin)

    def gripper(self, v, what=""):
        w, T = self.w, KIN.fk(self.q)
        self.gf = float(v)
        if v >= 0.9:
            if not w.held and w.state != "held" and np.linalg.norm((T[:3, 3] + R.TIP_CLOSED * T[:3, 2])[:2] - w.knob) < 0.006 and T[2, 3] - calib.table_z() < 0.16:
                w.held, w.state = True, "held"
                w.rel = T[:2, :2].T @ (w.c - T[:2, 3])
                w.ax_rel = T[:2, :2].T @ w.ax
                w.tz2 = float(T[2, 2])                                # tool tilt at the pinch must be kept to the release
            self.g = 85.0 if w.held else 98.5
        else:
            if w.held:
                h = T[2, 3] - calib.table_z()
                w.c = T[:2, 3] + T[:2, :2] @ w.rel + R.PLACE_BIAS
                w.tilt_err = float(abs(T[2, 2] - w.tz2)) if w.tz2 is not None else 0.0
                w.ax = T[:2, :2] @ w.ax_rel
                w.held, w.state = False, "table"
                if h > 0.16:
                    w.dropped = round(h, 3)
            self.g = 100.0 * v
        return self.g

    def tabletop(self):
        return Sim.blobs(self.w)


class Sim(R.Reset):
    @staticmethod
    def blobs(w):
        if w.state != "table":
            return []
        return [dict(xy=w.c.tolist(), height=0.096, z_top=0.08, extent=[0.09, 0.115], n=4200, points=np.zeros((5, 3)))]

    def scan(self, tag):
        w = self.s.w
        return dict(blobs=self.blobs(w), comps=[0.0965 if w.state == "comp" else 0.018, 0.035, 0.048])

    def look(self, xy, h_top, h_tool, tag):
        T = R.tool_down_T(np.r_[xy, self.tz + h_tool], R.YAW_DEG)
        Tc = T @ self.Ttc
        X = np.r_[xy, self.tz + h_top]
        T[:2, 3] += X[:2] - (Tc[:2, 3] + (X[2] - Tc[2, 3]) / Tc[2, 2] * Tc[:2, 2])
        self.s.move_joint(f"look_{tag}", self.ik(T), "normal")
        w = self.s.w
        if w.state == "held" or np.linalg.norm(w.knob - np.asarray(xy)) > 0.07:
            return None
        return dict(knob=w.knob.copy(), axis=w.ax.copy(), r_m=0.007, frame=[0.06, 0.032])


def scenario(flow, target, seed):
    nom = -R.A
    a = np.radians((seed % 5) * 7.0 - 14.0)
    ax0 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ nom
    w = {"compartment": lambda: World("comp", COMP_C, nom),
         "table": lambda: World("table", [0.335, -0.075], ax0),
         "in_place": lambda: World("table", np.array(R.aim_for(target, ax0)) + [0.002, 0.002], ax0),
         "near_base": lambda: World("table", [0.206, 0.019], [-0.83, 0.558])}[flow]()   # after attempt 99
    s = Sess(w)
    r = Sim(s, target)
    r.rng = np.random.default_rng(seed)
    t0 = time.time()
    msg = ""
    try:
        r.run()
    except (Abort, RuntimeError, ValueError) as e:
        msg = str(e)[:150]
    err = float(np.linalg.norm(w.c - target))
    again = None
    if flow == "compartment" and s.out == "success":     # a repeat with the same target must leave the drawer alone
        c0 = w.c.copy()
        s2 = Sess(w)
        r2 = Sim(s2, target)
        r2.rng = np.random.default_rng(seed + 1)
        try:
            r2.run()
        except (Abort, RuntimeError, ValueError) as e:
            msg = "repeat: " + str(e)[:130]
        again = bool(s2.S.get("already_in_place")) and s2.out == "success" and float(np.linalg.norm(w.c - c0)) < 1e-6
    return dict(flow=flow, target=list(target), claim=s.out, err=err, margin=R.edge_margin(w.c, w.ax)[0], gap=R.cab_gap(w.c, w.ax),
                shift=s.S.get("aim_shift_m"), yaw=s.S.get("yaw_goal_deg"), inplace=bool(s.S.get("already_in_place")), held=w.held,
                dropped=w.dropped or (w.tilt_err > 0.01 and "tilt changed"), again=again, msg=msg, segs=s.nseg, t=time.time() - t0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    L = TASK["layout"]["drawer"]
    print("layout range", L)
    res = []
    for i, x in enumerate(np.linspace(*L["x"], n)):
        for j, y in enumerate(np.linspace(*L["y"], n)):
            for flow in ("compartment", "table", "in_place", "near_base"):
                res.append(scenario(flow, np.array([x, y]), i + 2 * j))
    bad = [r for r in res if r["claim"] != "success" or r["err"] > 0.02 or r["margin"] < 0.10 or r["held"] or r["dropped"]]
    pl = [r for r in res if r["shift"] is not None]
    print(f"{len(res)} scenarios, {len(res) - len(bad)} ok, {sum(r['inplace'] for r in res)} left in place; "
          f"max err {max(r['err'] for r in res) * 100:.2f} cm, min edge margin {min(r['margin'] for r in res) * 100:.2f} cm, "
          f"min gap to cabinet {min(r['gap'] for r in res) * 100:.1f} cm, max aim shift {max(r['shift'] for r in pl) * 100:.2f} cm, "
          f"yaws used {sorted({r['yaw'] for r in pl})}, {np.mean([r['t'] for r in res]):.1f} s each")
    low = sorted(res, key=lambda r: r["gap"])[:4]
    print("smallest cabinet gaps:", [(r["flow"], np.round(r["target"], 3).tolist(), r["yaw"], round(r["gap"] * 100, 1), r["inplace"]) for r in low])
    rep = [r for r in res if r["again"] is not None]
    print("repeat after a placement left the drawer untouched:", sum(r["again"] for r in rep), "of", len(rep))
    ip = [r for r in res if r["flow"] == "in_place"]
    print("in-place starts left alone:", sum(r["inplace"] for r in ip), "of", len(ip), "(starts turned beyond +/-30 deg must be re-placed)")
    for r in bad[:12]:
        print("BAD", r["flow"], np.round(r["target"], 3).tolist(), r["claim"], f"err {r['err'] * 100:.2f} margin {r['margin'] * 100:.2f}",
              "held" if r["held"] else "", r["dropped"] or "", r["msg"])
