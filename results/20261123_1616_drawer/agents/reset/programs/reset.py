"""Reset for drawer_lying: take the drawer out of the lying cabinet by its knob and stand it, pull up, at the target.

Flow: get clear of the scene -> side-camera scan from home -> wrist look-down finds the knob -> pinch it across its width
-> lift straight up -> carry -> lower until the back wall is on the table -> release, rise -> home -> measure with the
same side-camera blobs the orchestrator uses -> correct by re-picking (at most twice) -> confirm the pull faces up.
Handled so far: drawer in a compartment, standing pull-up on the table, held by the gripper, arm stopped low.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj import config
from rlpj.perception import layout
from rlpj.perception import tabletop as tt
from rlpj.perception.wrist import project_base_to_wrist
from rlpj.robot import calib, sim
from rlpj.robot.kin import wrap180
from rlpj.robot.plan import NORMAL, SLOW, tool_down_T
from rlpj.robot.session import Abort, RobotSession
import knob

TASK = config.load("drawer_lying")["task"]
FX = TASK["fixtures"]["cabinet"]
YAW_DEG = float(FX["yaw_deg"])
A = np.array([np.cos(np.radians(YAW_DEG)), np.sin(np.radians(YAW_DEG))])   # along the row, away from the robot
B = np.array([-A[1], A[0]])
PITCH = 0.0878                  # compartment spacing along A (divider centres in the task yaml)
COMP_0 = np.array(FX["target_centre"], float) + 0.0102 * A + 0.0114 * B    # wrist-measured centre (attempt 3)
PRIOR = TASK["perception"]["drawer"]

HOME_Q = np.array([0, 15, -130, 0, 55, 90.0])
SEED_DOWN = np.array([0, 25, -110, 0, -45, 90.0])
READY_XYZ = (0.40, -0.10, 0.25)
DRAWER_L = 0.0885               # back wall to front face (attempt 12: pinched at 0.142, touched the table at 0.141); the blob top ~0.095 is the knob
HELD_H = 0.124                  # tool height for lowering a drawer held by an unknown grasp: the 20 N stop ends the move
KNOB_OFF = 0.0168               # drawer centre to knob centre, along the knob axis (attempt 3, two wrist views)
TIP_CLOSED = 0.047              # closed fingertips below the tool frame
TIP_CLEAR = 0.006               # closed fingertips above the drawer's top face at the pinch
PRE = 0.55                      # pre-close (~38 mm) around the 15 mm knob
GRIP = 1.0                      # command full closure: a reading well below it means the pads are stopped by the knob
HOLD = (65.0, 93.0)             # knob held (attempt 6: 86.6). Attempt 9: 94.8 at a 0.95 command was an empty pinch
COMP_FACE = 0.0925              # real height of a flush drawer face: attempt 6 pinched at 0.148 and touched the table 3 mm lower,
                                # so the face was ~0.092; the side depth reads 0.095-0.097 there. Pinching 1-2 mm too high slips (33, 34)
LIFT_START = 0.025              # first part of the lift, then check that the knob is still held
LIFT_FORCE = 35.0               # N, lift out of a compartment (attempt 6: 24 N): a jammed drawer must not raise the cabinet
LIFT_FORCE_TABLE = 50.0         # N, lift from the table: the force estimate read 19-45 N on these lifts (attempts 9, 12, 17)
PLACE_BIAS = np.array([0.0008, -0.0056])   # side-camera blob centre - commanded drawer centre near the target (attempts 12, 17)
LIFT = 0.13                     # operator 16:33: lift ~13 cm
TOL = 0.02                      # orchestrator layout tolerance
YAW_RANGE = 30.0                # deg, allowed turn from the cabinet axis
IN_PLACE = 0.018                # already this close to the target, pull up, inside the edge rule: leave it (attempt 15: the
                                # orchestrator finds a motionless drawer 14 times of 20, so a repeat must not re-handle it)
AIM_TOL = 0.008                 # re-place when the drawer is further than this from the aim
HALF = (0.040, 0.058)           # half sizes of the standing drawer: along the knob axis, across it
EDGE_MARGIN = 0.10              # user 16:41: the drawer's outline stays >= 10 cm inside every table edge
SHIFT_MAX = 0.019               # the aim may leave the target by this much (layout tolerance 2 cm)


def _plane(nx, ny, d):
    k = float(np.hypot(nx, ny))
    return np.array([nx / k, ny / k]), d / k


# table edges as half-planes n.p >= d. Camera side: the operator's line (16:41). The side camera's view ends before that
# edge (a fit to its cloud is the image border, notes/fov.py); wrist views put the real edge further out for x < 0.6
# (notes/edge_wrist.py), so the operator's line is the safe one. Far side fitted to the side cloud, robot end from the
# operator's note. The end beyond the cabinet was not measured (table points reach at least x = 0.75).
EDGES = [_plane(1 / 6, 1.0, -0.13), _plane(0.40, -1.0, -0.15), _plane(1.0, 0.0, 0.05), _plane(-1.0, 0.0, -0.80)]


def edge_margin(c, axis=None):
    """(margin m, inward normal of the nearest edge) of the drawer outline at centre c; axis None: a bounding circle."""
    c = np.asarray(c, float)
    best = (np.inf, None)
    for n, d in EDGES:
        if axis is None:
            sup = float(np.hypot(*HALF))
        else:
            a = np.asarray(axis, float) / np.linalg.norm(axis)
            sup = HALF[0] * abs(n @ a) + HALF[1] * abs(n @ np.array([-a[1], a[0]]))
        m = float(n @ c - d - sup)
        if m < best[0]:
            best = (m, n)
    return best


def safe_aim(xy, axis=None):
    """The point nearest xy whose outline keeps EDGE_MARGIN (plus 4 mm for the placing error) from every edge."""
    c = np.array(xy, float)
    for _ in range(5):
        m, n = edge_margin(c, axis)
        if m >= EDGE_MARGIN + 0.004:
            break
        c = c + (EDGE_MARGIN + 0.004 - m + 1e-4) * n
    return c


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class ForceLimit:
    """Lower the executor's contact-force stop for one move (never above the task cap)."""

    def __init__(self, s, limit):
        self.ex, self.limit, self.prev = s.executor, limit, None

    def __enter__(self):
        if self.ex is not None and getattr(self.ex, "force_limit", None) is not None:
            self.prev = self.ex.force_limit
            self.ex.force_limit = min(float(self.limit), float(self.prev))

    def __exit__(self, *a):
        if self.prev is not None:
            self.ex.force_limit = self.prev
        return False


class Reset:
    def __init__(self, s, target):
        self.s, self.target = s, np.array(target, float)
        self.tz = calib.table_z()
        self.rng = np.random.default_rng(int(time.time()))
        self.K, self.dist = calib.wrist_intrinsics()
        self.Ttc = calib.T_tool_cam()

    # ------------------------------------------------------------ motion helpers
    def q_now(self):
        return np.asarray(self.s.feedback()["q"], float)

    def T_now(self):
        return calib.pose_to_T(self.s.feedback()["pose"])

    def ik(self, T):
        down = self.T_now()[2, 2] < -0.9
        last = None
        for sd in ([self.q_now(), SEED_DOWN] if down else [SEED_DOWN, self.q_now()]):
            try:
                return self.s.planner.ik_exact(T, sd)[0]
            except Exception as e:
                last = e
        self.s.require(False, f"no IK for tool at {np.round(T[:3, 3], 3).tolist()}: {last}")

    def line(self, name, T, speed="slow", tip_margin=0.008):
        self.s.move_line(name, T, speed, tip_margin=tip_margin)

    def contact_line(self, name, T, limit=20.0):
        """Slow approach with a low force stop: True when reached, False when it stopped on contact."""
        try:
            with ForceLimit(self.s, limit):
                self.line(name, T, "slow")
            return True
        except Abort as e:
            if "contact force" not in str(e):
                raise
            self.s.decide(f"{name}: contact at the {limit:.0f} N stop")
            return False

    def rise(self, name, h, speed="slow", limit=None):
        T = self.T_now()
        T[2, 3] = self.tz + h
        if limit is None:
            self.line(name, T, speed)
        else:
            with ForceLimit(self.s, limit):
                self.line(name, T, speed)

    def set_home(self):
        """HOME_Q becomes the return pose of the session and the watchdog (a run may start low next to the cabinet)."""
        s = self.s
        if s.q_home is not None and np.abs(wrap180(np.asarray(s.q_home) - HOME_Q)).max() < 1:
            return
        s.q_home = HOME_Q.copy()
        if getattr(s, "wd", None) is not None:
            s.wd.stdin.write(("HOME " + " ".join(f"{v:.4f}" for v in HOME_Q) + "\n").encode())
            s.wd.stdin.flush()
        s.log("session home set to HOME_Q")

    def go_home(self):
        if np.abs(wrap180(self.q_now() - HOME_Q)).max() < 1.0:
            self.set_home()
            return
        T = tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0)
        self.s.move_joint("ready", self.s.planner.ik_exact(T, SEED_DOWN)[0], "normal")
        self.s.move_joint("home_pose", HOME_Q, "normal")
        self.set_home()

    def escape(self):
        """Start low or holding something: get up and clear. Returns True when the drawer is held by the knob."""
        s = self.s
        T = self.T_now()
        h = T[2, 3] - self.tz
        g = float(s.feedback()["gripper"])
        if h >= 0.20:                                   # never open blindly up here: the drawer may be in the gripper
            if g <= 3.0:
                return False
            g2 = float(s.gripper(1.0, "close fully at a safe height: is the drawer held?"))
            if not s.dry and g2 <= 97.0:
                s.decide(f"start: gripper stops at {g2:.1f} when closing: the drawer is held")
                return True
            s.decide(f"start: gripper closes to {g2:.1f}: nothing is held")
            s.gripper(0.0, "open: nothing is held")
            return False
        held = HOLD[0] <= g <= HOLD[1]
        s.decide(f"start: tool {h:.3f} m above the table, gripper {g:.1f}, {'holding' if held else 'not holding'}")
        if not held and g > 3.0:
            s.gripper(0.0, "open before rising")
        if T[2, 2] > -0.9:                              # tilted: back off along the tool axis first
            Tb = T.copy()
            Tb[:3, 3] -= 0.04 * T[:3, 2]
            with ForceLimit(s, 30.0):
                self.line("escape_tool_back", Tb, "slow", tip_margin=0.001)
        Tu = self.T_now()
        Tu[2, 3] = max(Tu[2, 3], self.tz + 0.27 if held else self.tz + 0.20)
        with ForceLimit(s, 40.0 if held else 30.0):
            self.line("escape_up", Tu, "slow", tip_margin=0.001)
        return held

    # ------------------------------------------------------------ perception
    def scan(self, tag):
        """From home: drawer blobs on the table (cabinet footprint removed) and the fill height of each compartment."""
        s = self.s
        P = tt.cloud(s.side, n=5)
        tz = tt.table_fit(P)
        h = P[:, 2] - tz
        d = P[:, :2] - np.asarray(FX["centre"], float)
        in_cab = (np.abs(d @ A) < FX["size"][0] / 2 + 0.02) & (np.abs(d @ B) < FX["size"][1] / 2 + 0.02)
        blobs = [b for b in tt.detect(P[~in_cab], table_z=tz, max_height=0.16)
                 if 0.05 <= b["height"] <= 0.14 and 0.05 <= max(b["extent"]) <= 0.19 and b["n"] > 1000]   # drawer ~4300 points; depth artifacts beside it ~200
        blobs.sort(key=lambda b: -b["n"])
        comps = []
        for k in range(3):
            e = P[:, :2] - (COMP_0 + k * PITCH * A)
            m = (np.abs(e @ A) < FX["compartment"][0] / 2 - 0.012) & (np.abs(e @ B) < FX["compartment"][1] / 2 - 0.012) & (h < 0.16)
            comps.append(round(float(np.median(h[m])), 4) if m.sum() > 30 else None)
        info = dict(blobs=[dict(xy=np.round(b["xy"], 3).tolist(), h=b["height"], ext=b["extent"], n=b["n"]) for b in blobs],
                    comps=comps)
        s.log(f"scan {tag}: {info}")
        s.summary(**{f"scan_{tag}": info})
        return dict(blobs=blobs, comps=comps)

    def face_height(self, b):
        """Height of the drawer's top face: the densest 2 mm layer of the blob (its 98th percentile is the knob's top)."""
        hs = np.asarray(b["points"], float)[:, 2] - (float(b["z_top"]) - float(b["height"]))
        edges = np.arange(0.05, float(b["height"]) + 0.006, 0.002)
        cnt, _ = np.histogram(hs, bins=edges)
        k = int(np.argmax(cnt))
        c = 0.5 * (edges[k] + edges[k + 1])
        face = float(np.median(hs[np.abs(hs - c) < 0.003]))
        self.s.log(f"top face {face:.4f} m (blob top {b['height']:.4f}, layer of {int(cnt[k])} of {len(hs)} points)")
        return float(np.clip(face, 0.07, 0.125))

    @staticmethod
    def comp_face(med):
        """Face height of a drawer in a compartment from the side-depth median (which reads ~3.5 mm high): flush or a
        protruding drawer."""
        return COMP_FACE if med <= 0.102 else float(np.clip(med - 0.0035, COMP_FACE, 0.125))

    def locate(self, p):
        if p["blobs"]:
            b = p["blobs"][0]
            standing = 0.085 <= float(b["height"]) <= 0.108   # the side-depth face height varies 5 mm with the spot (attempt 13)
            return dict(where="table", xy=np.array(b["xy"], float), h_top=DRAWER_L if standing else self.face_height(b))
        filled = [k for k, v in enumerate(p["comps"]) if v is not None and v > 0.06]
        if filled:
            k = max(filled, key=lambda i: p["comps"][i])
            return dict(where=f"compartment_{k}", xy=COMP_0 + k * PITCH * A, h_top=self.comp_face(p["comps"][k]))
        return None

    def ray_hit(self, Tn, uv, z):
        n = cv2.undistortPoints(np.array([[uv]], np.float64), self.K, self.dist)[0, 0]
        Tc = Tn @ self.Ttc
        d = Tc[:3, :3] @ np.r_[n, 1.0]
        return (Tc[:3, 3] + (z - Tc[2, 3]) / d[2] * d)[:2]

    def look(self, xy, h_top, h_tool, tag):
        """Tool-down view with the optical axis through (xy, h_top); the knob in the base frame, or None."""
        s = self.s
        T = tool_down_T(np.r_[xy, self.tz + h_tool], YAW_DEG)
        Tc = T @ self.Ttc
        X = np.r_[xy, self.tz + h_top]
        lam = (X[2] - Tc[2, 3]) / Tc[2, 2]
        T[:2, 3] += X[:2] - (Tc[:2, 3] + lam * Tc[:2, 2])
        s.move_joint(f"look_{tag}", self.ik(T), "normal")
        time.sleep(0.7)
        img = s.wrist.latest()[0]
        Tn = self.T_now()
        px = float(self.K[0, 0]) / float((Tn @ self.Ttc)[2, 3] - X[2])
        uv0 = project_base_to_wrist(X[None], Tn)[0]
        r = knob.detect(img, px, uv0, 0.07)
        draw = img.copy()
        res = None
        if r is not None:
            z = X[2] + 0.005
            k = self.ray_hit(Tn, r["knob_uv"], z)
            ax = self.ray_hit(Tn, r["knob_uv"] + 40.0 * r["axis"], z) - k
            res = dict(knob=k, axis=ax / np.linalg.norm(ax), r_m=r["r_m"], frame=r["frame_size_m"])
            cv2.circle(draw, tuple(r["knob_uv"].astype(int)), 10, (0, 255, 0), 2)
            cv2.circle(draw, tuple(r["frame_uv"].astype(int)), 10, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(s.A, f"look_{tag}.jpg"), cv2.resize(draw, (640, 360)))
        s.log(f"look {tag}: " + ("no knob" if res is None else f"knob {np.round(res['knob'], 4).tolist()} axis "
                                 f"{np.round(res['axis'], 3).tolist()} r {res['r_m']} frame {res['frame']}"))
        return res

    def find_knob(self, loc, tag):
        v = self.look(loc["xy"], loc["h_top"], loc["h_top"] + 0.165, f"{tag}_hi")
        if v is None and self.s.dry:                    # the dry-run wrist image does not show the simulated pose
            self.s.decide("dry run: knob position from the prior")
            ax = -A
            prior = dict(knob=loc["xy"] + KNOB_OFF * ax, axis=ax, r_m=0.0, frame=[])
            self.look(prior["knob"], loc["h_top"], loc["h_top"] + 0.085, f"{tag}_lo")   # checks the low view's motion
            return prior
        if v is None:
            return None
        low = self.look(v["knob"], loc["h_top"], loc["h_top"] + 0.085, f"{tag}_lo")
        return low if low is not None and np.linalg.norm(low["knob"] - v["knob"]) < 0.012 else v

    # ------------------------------------------------------------ pick and place
    def grasp_poses(self, k, ax, h_g):
        """Jaws square to the knob axis: the feasible yaw (of two) with the smallest joint move."""
        s = self.s
        a = np.degrees(np.arctan2(ax[1], ax[0]))
        q0 = self.q_now()
        best = None
        for yaw in (wrap(a + 90.0), wrap(a - 90.0)):
            Ta = tool_down_T(np.r_[k, self.tz + h_g + 0.09], yaw)
            Tg = tool_down_T(np.r_[k, self.tz + h_g], yaw)
            try:
                qa = s.planner.ik_exact(Ta, q0)[0]
                seg = s.planner.joint_move("chk", q0, qa, NORMAL)
                seg2 = s.planner.cart_line("chk2", qa, Tg, SLOW)
            except Exception as e:
                s.log(f"grasp yaw {yaw:.0f}: not plannable ({str(e)[:70]})")
                continue
            ok1 = sim.check(seg.q, s.kin, NORMAL, gripper_frac=PRE, table_z=self.tz)[0]
            ok2 = sim.check(seg2.q, s.kin, SLOW, gripper_frac=PRE, table_z=self.tz)[0]
            cost = float(np.abs(qa - q0).max())
            s.log(f"grasp yaw {yaw:.0f}: move {ok1} descent {ok2} cost {cost:.0f}")
            if ok1 and ok2 and (best is None or cost < best[0]):
                best = (cost, Ta, Tg, yaw)
        s.require(best is not None, "no feasible grasp yaw for the knob")
        return best[1], best[2], best[3]

    def pick(self, loc, tag):
        """Pinch the knob and lift the drawer LIFT m. Two tries; returns the grip description."""
        s = self.s
        for trial in range(2):
            s.phase(f"pick_{tag}_{trial}")
            v = self.find_knob(loc, f"{tag}_{trial}")
            s.require(v is not None, f"the pull is not visible from above at {np.round(loc['xy'], 3).tolist()} "
                                     f"({loc['where']}): the drawer is not pull-up")
            h_g = loc["h_top"] + TIP_CLOSED + TIP_CLEAR - 0.0015 * trial    # second try a little lower on the knob
            Ta, Tg, yaw = self.grasp_poses(v["knob"], v["axis"], h_g)
            s.decide(f"pinch the knob at {np.round(v['knob'], 4).tolist()}, jaw yaw {yaw:.1f}, tool {h_g:.3f} m above the table")
            s.gripper(PRE, "pre-close around the knob")
            s.move_joint("above_knob", self.ik(Ta), "normal")
            if not self.contact_line("descend_knob", Tg):
                s.gripper(0.0, "open after contact")
                self.rise("up_after_contact", h_g + 0.09)
                continue
            g = float(s.gripper(GRIP, "pinch the knob"))
            s.summary(**{f"pinch_{tag}_{trial}": round(g, 1)})
            if not s.dry and not HOLD[0] <= g <= HOLD[1]:
                s.decide(f"gripper reads {g:.1f}: the knob is not between the pads")
                s.gripper(PRE, "release")
                self.rise("up_after_miss", h_g + 0.09)
                continue
            limit = LIFT_FORCE if loc["where"].startswith("compartment") else LIFT_FORCE_TABLE
            z0 = self.T_now()[2, 3]
            held = True
            try:
                for name, dz in (("lift_start", LIFT_START), ("lift_drawer", LIFT)):
                    T = self.T_now()
                    T[2, 3] = z0 + dz
                    with ForceLimit(s, limit):
                        self.line(name, T, "slow")
                    held = self.still_held(f"{name}_{tag}_{trial}")
                    if not held:
                        break
            except Abort as e:
                if "contact force" not in str(e):
                    raise
                s.decide("the lift stopped at its force limit: lower the drawer back to where it was before letting go")
                self.contact_line("lower_back", Tg, 25.0)
                s.gripper(PRE, "release: the drawer is back at its place")
                self.rise("up_after_stuck", h_g + 0.09)
                continue
            if not held:
                s.decide("the knob slipped out of the pads during the lift: the drawer stayed behind")
                s.gripper(PRE, "open after the slip")
                self.rise("up_after_slip", max(self.T_now()[2, 3] - self.tz, h_g + 0.09))
                continue
            return dict(knob=v["knob"], axis=v["axis"], h_g=h_g, bottom=loc["h_top"] - DRAWER_L, yaw=yaw)
        s.require(False, f"could not grasp the knob after two tries ({loc['where']})")

    def still_held(self, when):
        """Command the closure again: after a slip the reading stays where it was until the gripper is told to move
        (attempts 33, 34 carried nothing at a reading of 88), and then it runs on to ~98."""
        g = float(self.s.gripper(GRIP, f"re-close: is the knob still held ({when})?"))
        self.s.summary(**{f"held_{when}": round(g, 1)})
        return self.s.dry or HOLD[0] <= g <= HOLD[1]

    def plan_yaw(self, ax, target):
        """(turn deg, aim): a random yaw within +/-YAW_RANGE of the cabinet axis among those needing the least shift of
        the aim to keep the outline EDGE_MARGIN inside the table."""
        nominal = np.degrees(np.arctan2(-A[1], -A[0]))
        now = wrap(np.degrees(np.arctan2(ax[1], ax[0])) - nominal)
        opts = []
        for goal in np.arange(-YAW_RANGE, YAW_RANGE + 0.1, 5.0):
            d = wrap(goal - now)
            d = d if abs(d) <= 90 else wrap(d + 180.0)
            c = safe_aim(target, rz(d)[:2, :2] @ ax)
            opts.append((float(np.linalg.norm(c - target)), float(goal), d, c))
        least = min(o[0] for o in opts)
        good = [o for o in opts if o[0] <= max(least + 0.003, 0.010)]
        s_, goal, d, c = good[int(self.rng.integers(len(good)))]
        self.s.decide(f"yaw {goal:+.0f} deg from the cabinet axis, aim {np.round(c, 4).tolist()} ({s_ * 100:.1f} cm from the target "
                      f"to keep {EDGE_MARGIN * 100:.0f} cm from the table edges)")
        self.s.summary(aim_shift_m=round(s_, 4), yaw_goal_deg=goal)
        if s_ > SHIFT_MAX:
            self.s.decide(f"target {np.round(target, 3).tolist()} cannot meet both the edge rule and the 2 cm tolerance: the edge rule wins")
        return d, c

    def place(self, grip, aim, delta):
        """Carry the held drawer so its centre is over aim, turned by delta deg, lower it onto the table, release."""
        s = self.s
        s.phase("place")
        T0 = self.T_now()
        R = rz(delta)
        ax = None if grip.get("axis") is None else R[:2, :2] @ grip["axis"]
        off = np.zeros(2) if ax is None else KNOB_OFF * ax
        m0 = edge_margin(aim, ax)[0]
        s.require(m0 >= EDGE_MARGIN, f"placement refused: outline {m0 * 100:.1f} cm from a table edge at {np.round(aim, 3).tolist()}")
        self.last_axis = ax
        Tc = T0.copy()
        Tc[:3, :3] = R @ T0[:3, :3]
        bias = PLACE_BIAS if aim[0] < 0.40 else np.zeros(2)
        Tc[:2, 3] = aim - bias + off
        s.decide(f"carry: drawer centre to {np.round(aim, 3).tolist()}, turn {delta:+.1f} deg")
        try:
            self.line("carry", Tc, "normal")
        except (RuntimeError, ValueError) as e:
            s.log(f"straight carry not plannable ({str(e)[:80]}): joint move")
            s.move_joint("carry_joint", self.ik(Tc), "normal")
        if not grip.get("unknown"):
            s.require(self.still_held("after_carry"), "the drawer was lost during the carry")
        h_place = grip["h_g"] - grip["bottom"] - 0.004     # 4 mm past the expected touch: the 20 N stop ends it
        T1 = Tc.copy()
        T1[2, 3] = self.tz + h_place + 0.05
        self.line("lower_fast", T1, "normal")
        T2 = Tc.copy()
        T2[2, 3] = self.tz + h_place
        touched = not self.contact_line("lower_to_table", T2, 20.0)
        s.require(touched or s.dry or not grip.get("unknown"), "held drawer: no table contact while lowering, not releasing")
        c_now = self.T_now()[:2, 3] - off + bias
        m1 = edge_margin(c_now, ax)[0]
        s.summary(edge_margin_at_release_m=round(m1, 4))
        s.require(m1 >= EDGE_MARGIN, f"release refused: outline {m1 * 100:.1f} cm from a table edge")
        s.gripper(PRE, "release the knob")
        s.decide(f"released {'on contact' if touched else 'at the planned height'}, tool "
                 f"{self.T_now()[2, 3] - self.tz:.3f} m above the table")
        self.rise("rise_after_release", h_place + 0.07)
        self.rise("rise_clear", 0.25, "normal")
        s.gripper(0.0, "open")

    def measure(self):
        """The drawer's position from the side camera with the cabinet footprint removed (attempt 12: the plain blob search
        merged the drawer and the cabinet in one frame). Also records whether the plain search sees it as its own blob,
        which is what the orchestrator's layout check needs."""
        plain = [b for b in self.s.tabletop() if layout.matches(PRIOR, b)
                 and float(np.linalg.norm(np.subtract(b["xy"], self.target))) < 0.06]
        blobs = self.scan("measure")["blobs"]
        self.s.summary(plain_blob_found=bool(plain))
        if not plain:
            self.s.decide("the plain side-camera blob search does not separate the drawer from the cabinet in this frame")
        if not blobs:
            return None, None
        b = min(blobs, key=lambda c: float(np.linalg.norm(np.subtract(c["xy"], self.target))))
        return np.array(b["xy"], float), float(b["height"])

    def in_place(self):
        """The knob axis when the drawer already stands correctly at the target, else None. The arm is at home."""
        s = self.s
        p = self.scan("start")
        loc = self.locate(p)
        if loc is None or loc["where"] != "table" or float(np.linalg.norm(loc["xy"] - self.target)) > IN_PLACE:
            return None
        v = self.look(loc["xy"], loc["h_top"], loc["h_top"] + 0.165, "in_place")
        if v is None:
            s.decide("near the target but the pull is not visible from above: handle it")
            self.go_home()
            return None
        nominal = np.degrees(np.arctan2(-A[1], -A[0]))
        turn = wrap(np.degrees(np.arctan2(v["axis"][1], v["axis"][0])) - nominal)
        turn = turn if abs(turn) <= 90 else wrap(turn + 180.0)
        margin = edge_margin(loc["xy"], v["axis"])[0]
        ok = abs(turn) <= YAW_RANGE + 3.0 and margin >= EDGE_MARGIN
        s.decide(f"drawer near the target: turn {turn:+.0f} deg, outline {margin * 100:.1f} cm from the nearest edge: "
                 + ("already in place, not moving it" if ok else "re-place it"))
        self.go_home()
        return v["axis"] if ok else None

    # ------------------------------------------------------------ main
    def run(self):
        s = self.s
        s.summary(measured=None, target=self.target.round(4).tolist())
        s.phase("clear")
        held = self.escape()
        grip = None
        self.set_home()                                   # clear of the scene now: aborts return to HOME_Q
        if held:
            c = safe_aim(self.target)
            s.decide(f"the drawer is already in the gripper: carry it tool-down above {np.round(c, 3).tolist()}")
            s.move_joint("carry_held", self.ik(tool_down_T(np.r_[c, self.tz + 0.27], YAW_DEG - 90.0)), "normal")
            grip = dict(knob=None, axis=None, h_g=HELD_H, bottom=0.0, yaw=None, unknown=True)
        else:
            self.go_home()
        s.require(edge_margin(self.target)[0] > -0.05, f"target {self.target.tolist()} is at a table edge")
        aim = self.target.copy()
        self.last_axis = None
        err, m, hb, seen = None, None, None, None
        placed = None if grip is not None else self.in_place()
        if placed is not None:
            self.last_axis, seen = placed, True
            m, hb = self.measure()
            s.require(m is not None, "the drawer blob was lost while checking it in place")
            err = float(np.linalg.norm(m - self.target))
            s.summary(already_in_place=True)
        for rnd in range(0 if placed is not None else 3):
            if grip is None:
                s.phase(f"scan_{rnd}")
                p = self.scan(str(rnd))
                loc = self.locate(p)
                s.require(loc is not None, f"drawer not found: no table blob, compartments {p['comps']}")
                s.decide(f"round {rnd}: drawer at {loc['where']} {np.round(loc['xy'], 3).tolist()}, top {loc['h_top']:.3f} m")
                grip = self.pick(loc, str(rnd))
            if rnd == 0 and grip.get("axis") is not None:
                delta, aim = self.plan_yaw(grip["axis"], self.target)
            else:
                delta, aim = 0.0, safe_aim(aim, grip.get("axis"))
            self.place(grip, aim, delta)
            grip = None
            self.go_home()
            if s.dry:
                s.decide("dry run: the scene did not change, stopping after the first place")
                s.summary(dry_flow="ok")
                s.claim("fail")
                return
            m, hb = self.measure()
            s.require(m is not None, "no drawer-sized blob on the table after placing")
            err = float(np.linalg.norm(m - self.target))
            s.summary(measured={"drawer": np.round(m, 4).tolist()}, error_m=round(err, 4), blob_height=hb, rounds=rnd + 1)
            s.decide(f"measured {np.round(m, 4).tolist()}, error {err * 100:.1f} cm (height {hb:.3f})")
            if float(np.linalg.norm(m - aim)) <= AIM_TOL:
                break
            aim = aim - np.clip(m - aim, -0.03, 0.03)
        s.phase("verify")
        if seen is None:
            seen = self.look(m, DRAWER_L, DRAWER_L + 0.165, "verify") is not None
            self.go_home()
        m2, hb2 = self.measure()
        if m2 is not None:
            m, err = m2, float(np.linalg.norm(m2 - self.target))
        margin = edge_margin(m, self.last_axis)[0]
        ok = bool(err is not None and err <= TOL and seen and m2 is not None and margin >= EDGE_MARGIN)
        s.summary(measured={"drawer": np.round(m, 4).tolist()}, errors={"drawer": round(err, 4)}, pull_seen_from_above=seen,
                  verify=dict(tolerance=TOL, height=hb2, edge_margin_m=round(margin, 4)))
        s.decide(f"final error {err * 100:.1f} cm, outline {margin * 100:.1f} cm from the nearest table edge, pull seen: {seen}")
        s.claim("success" if ok else "fail")


def parse_targets(v):
    return json.load(open(v)) if os.path.exists(v) else json.loads(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    args = ap.parse_args()
    target = parse_targets(args.targets)["drawer"]
    with RobotSession(program="reset") as sess:
        Reset(sess, target).run()
