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
TIP_CLEAR_TABLE = 0.003         # table picks: pads 3 mm above the face, across the knob's widest part (at 6 mm the drawer hung
                                # tilted in 92-109; compartment picks, which never tilted, sat ~3 mm lower on the knob)
USE_BODY = False                # body grasp: the open side offers only the 1 cm front-wall edge (108 slid 2 cm, 109 missed)
TIP_CLEAR = 0.006               # closed fingertips above the drawer's top face at the pinch
PRE = 0.55                      # pre-close (~38 mm) around the 15 mm knob
GRIP = 1.0                      # command full closure: a reading well below it means the pads are stopped by the knob
HOLD = (65.0, 89.0)             # knob held (good grips 84.9-86.6; 87.5-88.4 slipped in 33/34; 91.4 slid off and dropped it in 101)
BODY_DEPTH = 0.035              # body grasp: open fingertips this far below the top face (attempt 108: at 2 cm the drawer
                                # slid ~2 cm down in the pads during the carry; the table was touched 23 mm late)
BODY_HELD = (3.0, 25.0)         # reading when the pads hold the 73-78 mm body
SLIP = 3.0                      # a re-close reading this much above the pinch reading = the pads are sliding off
LYING_H = (0.060, 0.084)        # blob height of a drawer lying on its bottom, open side up (attempt 101: 0.070)
RIM_H = 0.078                   # its rim height
WALL_OPEN = 0.47                # ~45 mm around the 10 mm front wall
WALL_DEPTH = 0.018              # pads this far below the rim (attempt 104: 10 mm slid out on the lift)
WALL_SIDE = 0.035               # pinch this far from the wall centre, clear of the knob in the middle of the front face
COMP_FACE = 0.0925              # real height of a flush drawer face: attempt 6 pinched at 0.148 and touched the table 3 mm lower,
                                # so the face was ~0.092; the side depth reads 0.095-0.097 there. Pinching 1-2 mm too high slips (33, 34)
LIFT_START = 0.025              # first part of the lift, then check that the knob is still held
LIFT_FORCE = 35.0               # N, lift out of a compartment (attempt 6: 24 N): a jammed drawer must not raise the cabinet
LIFT_FORCE_TABLE = 50.0         # N, lift from the table: the force estimate read 19-45 N on these lifts (attempts 9, 12, 17)
PLACE_BIAS = np.array([0.0008, -0.0056])   # side-camera blob centre - commanded drawer centre near the target (attempts 12, 17)
LIFT = 0.13                     # operator 16:33: lift ~13 cm
TOL = 0.02                      # orchestrator layout tolerance
YAW_RANGE = 30.0                # deg, allowed turn from the cabinet axis
IN_PLACE = 0.019                # already this close to the target, pull up, inside the edge rule: leave it (attempt 15: the
                                # orchestrator finds a motionless drawer 14 times of 20, so a repeat must not re-handle it)
AIM_TOL = 0.008                 # re-place when the drawer is further than this from the aim
HALF = (0.040, 0.058)           # half sizes of the standing drawer: along the knob axis, across it
EDGE_MARGIN = 0.10              # user 16:41: the drawer's outline stays >= 10 cm inside every table edge
SHIFT_MAX = 0.0185              # the aim may leave the target by this much (layout tolerance 2 cm)
EDGE_BUF = (0.004, 0.003, 0.002, 0.001)   # extra edge margin of the aim: the largest that stays within SHIFT_MAX
GAP_WANT = 0.06                 # drawer outline to the cabinet: at ~5 cm the side camera merged them (operator 19:45, resets 22, 25)
NEAR_BASE_R = 0.24               # knob closer to the base axis: pinch with the tool tilted 20 deg away from the base
NEAR_BASE_TILT = 20.0           # (attempt 97: vertical pinch at 0.20 m left joint 3 only 3.5 deg from its limit; 20 deg: 21 deg)
NUDGE = 0.06                    # next to the base: slide the drawer this far toward +x by its knob (operator 00:57), then pick
GAP_MIN = 0.044                 # a drawer closer than this is not left in place (the tightest target corner plans 4.5 cm)


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


def safe_aim(xy, axis=None, buf=EDGE_BUF[0]):
    """The point nearest xy whose outline keeps EDGE_MARGIN + buf from every edge."""
    c = np.array(xy, float)
    for _ in range(5):
        m, n = edge_margin(c, axis)
        if m >= EDGE_MARGIN + buf:
            break
        c = c + (EDGE_MARGIN + buf - m + 1e-4) * n
    return c


def aim_for(target, axis):
    """safe_aim with the largest edge buffer that keeps the aim within 1 cm of the target, else the smallest buffer
    (attempt 105: a 1.8 cm shift left 2 mm of the 2 cm tolerance for the placing error)."""
    c = np.array(target, float)
    for b in EDGE_BUF:
        c = safe_aim(target, axis, b)
        if float(np.linalg.norm(c - target)) <= 0.010:
            break
    return c


def clear_cabinet(c, axis, target):
    """Slide the aim along the camera-side edge toward the robot, inside SHIFT_MAX, until the gap to the cabinet is
    GAP_WANT: the edge margin stays, the cabinet gets further away."""
    n = EDGES[0][0]
    u = np.array([-n[1], n[0]])
    u = u if u[0] < 0 else -u
    best = np.array(c, float)
    for lam in np.arange(0.001, 0.0301, 0.001):
        if cab_gap(best, axis) >= GAP_WANT:
            break
        p = np.asarray(c, float) + lam * u
        if float(np.linalg.norm(p - target)) > SHIFT_MAX or edge_margin(p, axis)[0] < EDGE_MARGIN + EDGE_BUF[-1]:
            break
        best = p
    return best


def cab_gap(c, axis):
    """Distance (m) along the row axis from the drawer outline to the cabinet's near end."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    b = np.array([-a[1], a[0]])
    reach = _along_row(a, b)
    return float(-FX["size"][0] / 2 - ((np.asarray(c, float) - np.asarray(FX["centre"], float)) @ A + reach))


def _along_row(a, b):
    return HALF[0] * abs(a @ A) + HALF[1] * abs(b @ A)


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
                 if 0.05 <= b["height"] <= 0.14 and 0.05 <= max(b["extent"]) <= 0.26 and b["n"] > 1000]   # drawer ~4300 points; depth artifacts beside it ~200
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
            if LYING_H[0] <= float(b["height"]) <= LYING_H[1]:
                return dict(where="lying", xy=np.array(b["xy"], float), h_top=RIM_H)
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

    def look_pose(self, xy, h_top, h_tool):
        X = np.r_[xy, self.tz + h_top]
        poses = []
        for yaw in (YAW_DEG, YAW_DEG + 180.0):
            T = tool_down_T(np.r_[xy, self.tz + h_tool], yaw)
            Tc = T @ self.Ttc
            lam = (X[2] - Tc[2, 3]) / Tc[2, 2]
            T[:2, 3] += X[:2] - (Tc[:2, 3] + lam * Tc[:2, 2])
            poses.append(T)
        return poses[0] if np.hypot(*poses[0][:2, 3]) >= 0.23 else max(poses, key=lambda P: np.hypot(*P[:2, 3]))

    def look(self, xy, h_top, h_tool, tag):
        """Tool-down view with the optical axis through (xy, h_top); the knob in the base frame, or None."""
        s = self.s
        X = np.r_[xy, self.tz + h_top]
        poses = []
        for yaw in (YAW_DEG, YAW_DEG + 180.0):   # the camera sits 5 cm off the tool axis: near the robot base the default
            T = tool_down_T(np.r_[xy, self.tz + h_tool], yaw)   # look put the tool inside the base keep-out (attempts 94, 95)
            Tc = T @ self.Ttc
            lam = (X[2] - Tc[2, 3]) / Tc[2, 2]
            T[:2, 3] += X[:2] - (Tc[:2, 3] + lam * Tc[:2, 2])
            poses.append(T)
        T = poses[0] if np.hypot(*poses[0][:2, 3]) >= 0.23 else max(poses, key=lambda P: np.hypot(*P[:2, 3]))
        s.move_joint(f"look_{tag}", self.ik(T), "normal")
        time.sleep(0.7)
        img = s.wrist.latest()[0]
        Tn = self.T_now()
        px = float(self.K[0, 0]) / float((Tn @ self.Ttc)[2, 3] - X[2])
        uv0 = project_base_to_wrist(X[None], Tn)[0]
        r = knob.detect(img, px, uv0, 0.05)              # attempt 109: at 7 cm the cabinet's edge next to the drawer won
        if r is None:
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
        try:
            low = self.look(v["knob"], loc["h_top"], loc["h_top"] + 0.085, f"{tag}_lo")
        except Abort as e:                      # the low view is refused next to the base: the high view is enough
            if "model check failed" not in str(e):
                raise
            self.s.log(f"low look refused ({str(e)[:80]}): using the high view")
            low = None
        return low if low is not None and np.linalg.norm(low["knob"] - v["knob"]) < 0.012 else v

    # ------------------------------------------------------------ pick and place
    def grasp_poses(self, k, ax, h_g, body=False):
        """Jaws square to the knob axis: of both yaws and several IK seeds, the feasible plan with the smallest joint move.
        Attempt 97: next to the robot base the IK seeded from the look pose landed on a branch whose descent hit the
        joint limits, while the branch from a tool-down seed was clear."""
        s = self.s
        a = np.degrees(np.arctan2(ax[1], ax[0]))
        q0 = self.q_now()
        best = None
        tilt = 0.0 if body else (NEAR_BASE_TILT if np.hypot(*k) < NEAR_BASE_R else 0.0)
        gf = 0.0 if body else PRE
        if tilt:
            s.decide(f"knob {np.hypot(*k):.3f} m from the base axis: pinch with the tool tilted {tilt:.0f} deg away from the base")
        for yaw in ((wrap(a), wrap(a + 180.0)) if body else (wrap(a + 90.0), wrap(a - 90.0))):
            Tg = tool_down_T(np.zeros(3), yaw)
            if tilt:
                u = np.r_[np.asarray(k, float) / np.linalg.norm(k), 0.0]
                kx = np.cross([0.0, 0.0, 1.0], u)
                kx /= np.linalg.norm(kx)
                K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
                th = np.radians(tilt)
                Tg[:3, :3] = (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ Tg[:3, :3]
            tip = np.r_[k, self.tz + h_g - TIP_CLOSED]             # the fingertips where a vertical pinch puts them
            Tg[:3, 3] = np.r_[k, self.tz + h_g] if body else tip - TIP_CLOSED * Tg[:3, 2]
            Ta = Tg.copy()
            Ta[:3, 3] -= 0.09 * Tg[:3, 2]
            seeds = [q0, SEED_DOWN, np.array([0, -7, -134, 0, -53, yaw]), np.array([0, 40, -120, 0, -60, yaw])]
            seen = []
            for sd in seeds:
                try:
                    qa = s.planner.ik_exact(Ta, sd)[0]
                except Exception:
                    continue
                qa = q0 + wrap180(qa - q0)
                if any(np.abs(qa - p).max() < 1.0 for p in seen):
                    continue
                seen.append(qa)
                try:
                    seg = s.planner.joint_move("chk", q0, qa, NORMAL)
                    seg2 = s.planner.cart_line("chk2", qa, Tg, SLOW)
                except Exception as e:
                    s.log(f"grasp yaw {yaw:.0f}: not plannable ({str(e)[:70]})")
                    continue
                ok1, r1 = sim.check(seg.q, s.kin, NORMAL, gripper_frac=gf, table_z=self.tz)
                ok2, r2 = sim.check(seg2.q, s.kin, SLOW, gripper_frac=gf, table_z=self.tz)
                cost = float(np.abs(qa - q0).max())
                for ok, r, nm in ((ok1, r1, "move"), (ok2, r2, "descent")):
                    if not ok:
                        s.log(f"grasp yaw {yaw:.0f} {nm} refused: " + str({kk: v for kk, v in r.items() if kk in (
                            "joint_limit_violations", "keep_out_violations", "near_base", "min_tip_above_table",
                            "min_body_above_table", "min_link_z", "keep_in_violations", "joint_speed_ok", "max_radius")}))
                s.log(f"grasp yaw {yaw:.0f} branch {np.round(qa, 0).tolist()}: move {ok1} descent {ok2} cost {cost:.0f}")
                if ok1 and ok2 and (best is None or cost < best[0]):
                    best = (cost, Ta, Tg, yaw, qa)
        s.require(best is not None, "no feasible grasp yaw for the knob")
        self.grasp_q = best[4]
        return best[1], best[2], best[3]

    def pick(self, loc, tag):
        """Pinch the knob and lift the drawer LIFT m. Two tries; returns the grip description."""
        s = self.s
        nudges, body_tried = 0, False
        for trial in range(4):
            if trial - nudges >= 2:                   # two real grasp tries; slides away from the base do not count
                break
            s.phase(f"pick_{tag}_{trial}")
            v = self.find_knob(loc, f"{tag}_{trial}")
            s.require(v is not None, f"the pull is not visible from above at {np.round(loc['xy'], 3).tolist()} "
                                     f"({loc['where']}): the drawer is not pull-up")
            if USE_BODY and loc["where"] == "table" and np.hypot(*v["knob"]) >= NEAR_BASE_R and not body_tried:
                body_tried = True
                gb = self.pick_body(v, f"{tag}_{trial}")
                if gb is not None:
                    return gb
            self.hold_range = HOLD
            clear = TIP_CLEAR_TABLE if loc["where"] == "table" else TIP_CLEAR
            h_g = loc["h_top"] + TIP_CLOSED + clear - 0.0015 * trial    # second try a little lower on the knob
            Ta, Tg, yaw = self.grasp_poses(v["knob"], v["axis"], h_g)
            h_g = float(Tg[2, 3] - self.tz)                         # tool height of the (possibly tilted) pinch
            k_rel = np.asarray(v["knob"], float) - Tg[:2, 3]          # knob from the tool origin (non-zero when tilted)
            s.decide(f"pinch the knob at {np.round(v['knob'], 4).tolist()}, jaw yaw {yaw:.1f}, tool {h_g:.3f} m above the table")
            s.gripper(PRE, "pre-close around the knob")
            s.move_joint("above_knob", self.grasp_q, "normal")      # the branch that was checked
            if not self.contact_line("descend_knob", Tg):
                s.gripper(0.0, "open after contact")
                self.rise("up_after_contact", h_g + 0.09)
                continue
            g = float(s.gripper(GRIP, "pinch the knob"))
            self.g_pinch = g
            s.summary(**{f"pinch_{tag}_{trial}": round(g, 1)})
            if not s.dry and not HOLD[0] <= g <= HOLD[1]:
                s.decide(f"gripper reads {g:.1f}: the knob is not between the pads")
                s.gripper(PRE, "release")
                self.rise("up_after_miss", h_g + 0.09)
                continue
            if np.hypot(*v["knob"]) < NEAR_BASE_R and loc["where"] == "table" and nudges < 2:
                loc = self.nudge(loc, h_g)
                nudges += 1
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
            return dict(knob=v["knob"], axis=v["axis"], h_g=h_g, bottom=loc["h_top"] - DRAWER_L, yaw=yaw, k_rel=k_rel)
        s.require(False, f"could not grasp the knob after two tries ({loc['where']})")

    def pick_body(self, v, tag):
        """Attempts 92-106: picked by the knob from the table the drawer often hung tilted about the jaw axis (its centre
        is 17 mm off the knob), touched down 12 mm early and landed 2-5 cm off. Grip the body instead: open jaws along the
        knob axis around its 73-78 mm side, 2 cm below the top face; the off-centre weight then lies along the jaw axis.
        Returns the grip, or None (then the knob pinch is used)."""
        s = self.s
        ax = np.asarray(v["axis"], float)
        c = np.asarray(v["knob"], float) - KNOB_OFF * ax
        h_t = DRAWER_L - BODY_DEPTH + 0.027
        try:
            Ta, Tg, yaw = self.grasp_poses(c, ax, h_t, body=True)
        except Abort as e:
            s.log(f"body grasp not plannable: {str(e)[:80]}")
            return None
        s.decide(f"grip the drawer body at {np.round(c, 4).tolist()}, jaws along the knob axis (yaw {yaw:.1f})")
        s.gripper(0.0, "open fully around the drawer body")
        s.move_joint("above_body", self.grasp_q, "normal")
        if not self.contact_line("descend_body", Tg):
            s.decide("touched the drawer while reaching around it: use the knob instead")
            self.rise("up_after_body_contact", h_t + 0.12)
            return None
        g = float(s.gripper(GRIP, "grip the drawer body"))
        s.summary(**{f"body_grip_{tag}": round(g, 1)})
        if not s.dry and not BODY_HELD[0] <= g <= BODY_HELD[1]:
            s.decide(f"body grip reads {g:.1f}: not holding the body, use the knob instead")
            s.gripper(0.0, "open")
            self.rise("up_after_body_miss", h_t + 0.12)
            return None
        self.g_pinch, self.hold_range = g, BODY_HELD
        z0 = self.T_now()[2, 3]
        for name, dz in (("lift_body_start", LIFT_START), ("lift_body", LIFT)):
            T = self.T_now()
            T[2, 3] = z0 + dz
            try:
                with ForceLimit(s, LIFT_FORCE_TABLE):
                    self.line(name, T, "slow")
            except Abort as e:
                if "contact force" not in str(e):
                    raise
                s.decide("the body lift stopped at its force limit: put it back down and use the knob")
                self.contact_line("lower_back_body", Tg, 25.0)
                s.gripper(0.0, "open")
                self.rise("up_after_body_stuck", h_t + 0.12)
                return None
            if not self.still_held(f"{name}_{tag}"):
                s.decide("the body slipped in the pads: put it back down and use the knob")
                self.contact_line("lower_back_body", Tg, 25.0)
                s.gripper(0.0, "open")
                self.rise("up_after_body_slip", h_t + 0.12)
                return None
        return dict(knob=v["knob"], axis=ax, h_g=h_t, bottom=0.0, yaw=yaw, k_rel=np.zeros(2), off_len=0.0, body=True)

    def nudge(self, loc, h_g):
        """Attempts 97-99: next to the robot base the arm cannot lift the drawer (vertical: joint 3 at its limit; tilted:
        the knob slipped). Holding the knob, slide the drawer NUDGE m toward +x along the table (away from the base and
        from every table edge; it only needs to overcome sliding friction), let go, rise. Returns the updated location."""
        s = self.s
        T0 = self.T_now()
        dst = T0.copy()
        dst[0, 3] += NUDGE
        dst[2, 3] += 0.001
        c_new = np.asarray(loc["xy"], float) + [NUDGE, 0.0]
        s.require(edge_margin(c_new)[0] >= min(EDGE_MARGIN, edge_margin(loc["xy"])[0]), "the slide would bring the drawer nearer a table edge")
        s.decide(f"drawer next to the robot base: slide it {NUDGE * 100:.0f} cm toward +x by its knob, then pick it")
        try:
            with ForceLimit(s, 20.0):
                self.line("slide_from_base", dst, "slow")
        except Abort as e:
            if "contact force" not in str(e):
                raise
            s.decide("the slide stopped at the 20 N limit")
        moved = float(self.T_now()[0, 3] - T0[0, 3])
        s.gripper(PRE, "let go after the slide")
        self.rise("up_after_slide", h_g + 0.09)
        s.summary(slid_m=round(moved, 4))
        return dict(loc, xy=np.asarray(loc["xy"], float) + [moved, 0.0])

    def still_held(self, when):
        """Command the closure again: after a slip the reading stays where it was until the gripper is told to move
        (attempts 33, 34 carried nothing at a reading of 88), and then it runs on to ~98."""
        g = float(self.s.gripper(GRIP, f"re-close: is the knob still held ({when})?"))
        self.s.summary(**{f"held_{when}": round(g, 1)})
        ref = getattr(self, "g_pinch", None)
        lo, hi = getattr(self, "hold_range", HOLD)
        return self.s.dry or (lo <= g <= hi + 1.0 and (ref is None or g <= ref + SLIP))

    def lying_pose(self, loc):
        """Drawer lying on its bottom, open side up: its outline from a wrist look-down (wood colour) and its front wall
        = the long side with the dark metal knob just outside it. dict(c, u, nf) or None."""
        s = self.s
        T = self.look_pose(loc["xy"], RIM_H, RIM_H + 0.20)
        s.move_joint("look_lying", self.ik(T), "normal")
        time.sleep(0.7)
        img = s.wrist.latest()[0]
        Tn = self.T_now()
        z = self.tz + RIM_H
        px = float(self.K[0, 0]) / float((Tn @ self.Ttc)[2, 3] - z)
        uv0 = project_base_to_wrist(np.r_[loc["xy"], z][None], Tn)[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S_, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        wood = ((H >= 3) & (H <= 30) & (S_ >= 40) & (V >= 60)).astype(np.uint8)
        roi = np.zeros(wood.shape, np.uint8)
        if np.all(np.isfinite(uv0)):
            cv2.circle(roi, (int(uv0[0]), int(uv0[1])), int(0.10 * px), 1, -1)
        wood &= roi
        wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(wood, connectivity=8)
        res = None
        if n > 1:
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            box = cv2.boxPoints(cv2.minAreaRect(np.column_stack(np.nonzero(lab == j))[:, ::-1].astype(np.float32)))
            P = np.array([self.ray_hit(Tn, b, z) for b in box])
            e1, e2 = P[1] - P[0], P[2] - P[1]
            L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
            u = (e1 / L1) if L1 >= L2 else (e2 / L2)
            lo, sh = max(L1, L2), min(L1, L2)
            c = P.mean(0)
            v = np.array([-u[1], u[0]])
            s.log(f"lying drawer outline {lo:.3f} x {sh:.3f} m at {np.round(c, 3).tolist()}")
            if 0.095 <= lo <= 0.14 and 0.07 <= sh <= 0.11:
                counts = []
                for sg in (1.0, -1.0):
                    pts = [np.r_[c + sg * v * (sh / 2 + d) + u * tt, self.tz + 0.045] for d in (0.006, 0.012, 0.018, 0.024)
                           for tt in np.linspace(-0.03, 0.03, 9)]
                    uv = project_base_to_wrist(np.array(pts), Tn)
                    k = 0
                    for q in uv:
                        if np.all(np.isfinite(q)) and 0 <= q[0] < img.shape[1] and 0 <= q[1] < img.shape[0]:
                            k += int(V[int(q[1]), int(q[0])] < 95)
                    counts.append(k)
                s.log(f"dark pixels beside the two long sides: {counts}")
                if max(counts) >= 4 and max(counts) >= 2 * min(counts) + 2:
                    nf = v if counts[0] > counts[1] else -v
                    res = dict(c=c, u=u, nf=nf, depth=sh)
            cv2.drawContours(img, [box.astype(np.int32)], 0, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(s.A, "look_lying.jpg"), cv2.resize(img, (640, 360)))
        if res is None and s.dry:
            res = dict(c=np.asarray(loc["xy"], float), u=A.copy(), nf=B.copy(), depth=0.0885)
        if res is not None:
            s.decide(f"lying drawer: front wall faces {np.round(res['nf'], 2).tolist()}")
        return res

    def right_lying(self, loc):
        """Turn a drawer lying open side up onto its back wall, pull up: pinch the front wall's top edge, lift, turn it
        90 deg about its long axis in the air (front up), lower until the back wall touches, let go, back out."""
        s = self.s
        s.phase("right_lying")
        pose = self.lying_pose(loc)
        s.require(pose is not None, "drawer lying open side up, but its front wall (the knob side) is not clear from above")
        c, nf = pose["c"], pose["nf"]
        tipz = 0.027 + 0.020 * WALL_OPEN
        G = c + nf * (pose["depth"] / 2 - 0.005) + pose["u"] * WALL_SIDE
        Tg = tool_down_T(np.r_[G, self.tz + RIM_H - WALL_DEPTH + tipz], np.degrees(np.arctan2(nf[1], nf[0])))
        Ta = Tg.copy()
        Ta[2, 3] += 0.08
        Tup = Tg.copy()
        Tup[2, 3] = self.tz + 0.22
        P = Tup[:3, 3] + tipz * Tup[:3, 2]
        pl, kin = s.planner, s.kin
        plan = None
        a0 = np.degrees(np.arctan2(nf[1], nf[0]))
        # the turn is reachable for some front directions only (offline: ~30-90 and ~270-300 deg): turn the lifted drawer
        # about the vertical first, by the smallest angle that brings its front to a reachable direction
        phis = sorted({round(float(wrap(g - a0)), 1) for g in (45.0, 60.0, 75.0, 90.0, 270.0, 285.0)} | {0.0}, key=abs)
        for phi in phis:
            if abs(phi) > 150.0:
                continue
            Tyaw = Tup.copy()
            Tyaw[:3, :3] = rz(phi) @ Tup[:3, :3]
            nf2 = rz(phi)[:2, :2] @ nf
            ax3 = np.cross(np.r_[nf2, 0.0], [0.0, 0.0, 1.0])
            K = np.array([[0, -ax3[2], ax3[1]], [ax3[2], 0, -ax3[0]], [-ax3[1], ax3[0], 0]])
            Rr = (np.eye(3) + K + K @ K) @ Tyaw[:3, :3]      # Rodrigues, +90 deg about nf x Z: maps nf to +Z (front up)
            for roll in (0.0, 180.0):
                R2 = Rr @ rz(roll)
                Tr = np.eye(4)
                Tr[:3, :3] = R2
                Tr[:3, 3] = P - tipz * R2[:, 2]
                Td = Tr.copy()
                Td[2, 3] -= P[2] - (self.tz + 0.0885 - 0.005 - 0.006)       # grip point to the top face, 6 mm past touch
                To = Td.copy()
                To[:3, 3] -= 0.06 * R2[:, 2]
                try:
                    q0 = self.q_now()
                    qa = pl.ik_exact(Ta, q0 if self.T_now()[2, 2] < -0.9 else SEED_DOWN)[0]
                    segs = [(pl.joint_move("a", q0, qa, NORMAL), NORMAL, WALL_OPEN)]
                    segs.append((pl.cart_line("g", qa, Tg, SLOW), SLOW, WALL_OPEN))
                    segs.append((pl.cart_line("u", segs[-1][0].q[-1], Tup, SLOW), SLOW, 0.9))
                    qy = pl.ik_exact(Tyaw, segs[-1][0].q[-1])[0]
                    segs.append((pl.joint_move("y", segs[-1][0].q[-1], qy, SLOW), SLOW, 0.9))
                    qr = pl.ik_exact(Tr, qy)[0]
                    segs.append((pl.joint_move("r", qy, qr, SLOW), SLOW, 0.9))
                    segs.append((pl.cart_line("d", qr, Td, SLOW), SLOW, 0.9))
                    segs.append((pl.cart_line("o", segs[-1][0].q[-1], To, SLOW), SLOW, WALL_OPEN))
                    Tu2 = To.copy()
                    Tu2[2, 3] += 0.05
                    segs.append((pl.cart_line("u2", segs[-1][0].q[-1], Tu2, SLOW), SLOW, WALL_OPEN))
                    qready = pl.ik_exact(tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0), SEED_DOWN)[0]
                    segs.append((pl.joint_move("h", segs[-1][0].q[-1], qready, NORMAL), NORMAL, WALL_OPEN))
                except Exception as e:
                    s.log(f"righting yaw {phi:+.0f} roll {roll:.0f}: not plannable ({str(e)[:70]})")
                    continue
                bad = [i for i, (sg, sp, gf) in enumerate(segs) if not sim.check(sg.q, kin, sp, gripper_frac=gf, table_z=self.tz)[0]]
                s.log(f"righting yaw {phi:+.0f} roll {roll:.0f}: refused segments {bad}")
                if not bad:
                    plan = (qa, qy, qr, Tr, Td, To, phi, Tu2, qready)
                    break
            if plan is not None:
                break
        s.require(plan is not None, "the righting move for the lying drawer is not reachable from here")
        qa, qy, qr, Tr, Td, To, phi, Tu2, qready = plan
        s.decide(f"righting plan: turn the lifted drawer {phi:+.0f} deg about the vertical, then front up")
        s.gripper(WALL_OPEN, "open around the front wall")
        s.move_joint("above_front_wall", qa, "normal")
        if not self.contact_line("descend_wall", Tg, 20.0):
            s.gripper(0.0, "open")
            self.rise("up_after_wall_contact", 0.20)
            s.require(False, "touched something while reaching the front wall")
        g = float(s.gripper(GRIP, "pinch the front wall"))
        s.summary(wall_pinch=round(g, 1))
        if not s.dry and not 60.0 <= g <= 96.0:
            s.gripper(0.0, "open")
            self.rise("up_after_wall_miss", 0.20)
            s.require(False, f"the front wall is not between the pads (reading {g:.1f})")
        T1 = Tg.copy()
        T1[2, 3] += 0.025
        for name, Tl in (("lift_lying_start", T1), ("lift_lying", Tup)):
            with ForceLimit(s, LIFT_FORCE):
                self.line(name, Tl, "slow")
            g2 = float(s.gripper(GRIP, "re-close: still holding the wall?"))
            s.summary(**{f"held_{name}": round(g2, 1)})
            if not s.dry and g2 > g + SLIP:
                s.gripper(0.0, "open after the slip")
                self.rise("up_after_wall_slip", 0.22)
                s.require(False, f"the front wall slipped out of the pads during the lift ({g2:.1f})")
        if abs(phi) > 0.5:
            s.move_joint("turn_about_vertical", qy, "slow")
        s.move_joint("turn_front_up", qr, "slow")
        touched = not self.contact_line("lower_righted", Td, 20.0)
        s.require(touched or s.dry, "no table contact under the righted drawer: not letting go in the air")
        s.gripper(WALL_OPEN, "let go of the front wall")
        self.line("back_out", To, "slow", tip_margin=0.004)
        self.line("up_after_righting", Tu2, "slow", tip_margin=0.004)
        s.move_joint("ready_after_righting", qready, "normal")      # tool down again (a straight rise hit joint limits)
        s.gripper(0.0, "open")
        s.decide("drawer turned onto its back wall")

    def plan_yaw(self, ax, target):
        """(turn deg, aim): a random yaw within +/-YAW_RANGE of the cabinet axis among those needing the least shift of
        the aim to keep the outline EDGE_MARGIN inside the table."""
        nominal = np.degrees(np.arctan2(-A[1], -A[0]))
        now = wrap(np.degrees(np.arctan2(ax[1], ax[0])) - nominal)
        opts = []
        for goal in np.arange(-YAW_RANGE, YAW_RANGE + 0.1, 5.0):
            d = wrap(goal - now)
            d = d if abs(d) <= 90 else wrap(d + 180.0)
            axn = rz(d)[:2, :2] @ ax
            c = clear_cabinet(aim_for(target, axn), axn, target)
            opts.append((float(np.linalg.norm(c - target)), float(goal), d, c, cab_gap(c, axn)))
        least = min(o[0] for o in opts)
        fits = [o for o in opts if o[0] <= max(min(SHIFT_MAX, least + 0.003), 0.010)]   # edge rule, then the least shift
        wide = max(o[4] for o in fits)
        good = [o for o in fits if o[4] >= min(GAP_WANT, wide - 0.003)]        # then clear of the cabinet, yaw varies among these
        s_, goal, d, c, gap = good[int(self.rng.integers(len(good)))]
        self.s.summary(cabinet_gap_m=round(gap, 4))
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
        off = np.zeros(2) if ax is None else grip.get("off_len", KNOB_OFF) * ax
        m0 = edge_margin(aim, ax)[0]
        s.require(m0 >= EDGE_MARGIN, f"placement refused: outline {m0 * 100:.1f} cm from a table edge at {np.round(aim, 3).tolist()}")
        self.last_axis = ax
        Tc = T0.copy()
        Tc[:3, :3] = R @ T0[:3, :3]
        bias = PLACE_BIAS if aim[0] < 0.40 else np.zeros(2)
        kr = R[:2, :2] @ np.asarray(grip.get("k_rel", np.zeros(2)), float)   # tilted pinch: the tool is not above the knob
        Tc[:2, 3] = aim - bias + off - kr
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
        if not touched and not s.dry:
            # attempt 101: released at the planned height without contact, the drawer fell over. Go on down (tips stay
            # above the table) and never let go in the air.
            T2b = T2.copy()
            T2b[2, 3] = max(T2[2, 3] - 0.025, self.tz + TIP_CLOSED + 0.012)
            s.decide("no table contact at the planned height: keep lowering")
            touched = not self.contact_line("lower_more", T2b, 20.0)
            s.require(touched, "no table contact while lowering the drawer: not letting go in the air")
        early = float(self.T_now()[2, 3] - self.tz - (h_place + 0.004))
        s.summary(touch_early_mm=round(early * 1000, 1))
        if touched and early > 0.004 and not grip.get("unknown") and not s.dry:
            # attempts 92, 94: the drawer hung tilted in the pads, a corner touched 1 cm early and, let go there, it fell
            # flat 5 cm away (next to the robot base). Press it level first (it turns about the knob back under the pads).
            s.decide(f"touched the table {early * 1000:.0f} mm early: the drawer hangs tilted, press it level before letting go")
            T3 = Tc.copy()
            T3[2, 3] = self.tz + h_place + 0.002
            self.contact_line("level_drawer", T3, 35.0)
            s.summary(touch_after_level_mm=round(float(self.T_now()[2, 3] - self.tz - (h_place + 0.004)) * 1000, 1))
        s.require(touched or s.dry or not grip.get("unknown"), "held drawer: no table contact while lowering, not releasing")
        c_now = self.T_now()[:2, 3] + kr - off + bias
        m1 = edge_margin(c_now, ax)[0]
        s.summary(edge_margin_at_release_m=round(m1, 4))
        s.require(m1 >= EDGE_MARGIN, f"release refused: outline {m1 * 100:.1f} cm from a table edge")
        s.gripper(0.0 if grip.get("body") else PRE, "release the drawer")
        s.decide(f"released {'on contact' if touched else 'at the planned height'}, tool "
                 f"{self.T_now()[2, 3] - self.tz:.3f} m above the table")
        self.rise("rise_after_release", h_place + 0.07)
        self.rise("rise_clear", 0.25, "normal")
        s.gripper(0.0, "open")

    def measure(self):
        """The drawer's position from the side camera with the cabinet footprint removed (attempt 12: the plain blob search
        merged the drawer and the cabinet in one frame). Also records whether the plain search sees it as its own blob,
        which is what the orchestrator's layout check needs."""
        xy, hs, plain_ok, n = [], [], 0, 0
        for n in range(1, 7):                 # depth artifacts come and go per frame (attempts 21, 30 lost the blob in one)
            plain_ok += any(layout.matches(PRIOR, b) and float(np.linalg.norm(np.subtract(b["xy"], self.target))) < 0.06
                            for b in self.s.tabletop())
            blobs = self.scan("measure")["blobs"]
            if blobs:
                b = min(blobs, key=lambda c: float(np.linalg.norm(np.subtract(c["xy"], self.target))))
                xy.append(b["xy"])
                hs.append(float(b["height"]))
            if len(xy) >= 3:
                break
            time.sleep(0.25)
        self.s.summary(plain_blob_found=f"{plain_ok}/{n}", measure_frames=f"{len(xy)}/{n}")
        if not xy:
            return None, None
        return np.median(np.array(xy, float), axis=0), float(np.median(hs))

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
        gap = cab_gap(loc["xy"], v["axis"])
        ok = abs(turn) <= YAW_RANGE + 3.0 and margin >= EDGE_MARGIN and gap >= GAP_MIN
        s.decide(f"drawer near the target: turn {turn:+.0f} deg, outline {margin * 100:.1f} cm from the nearest edge, {gap * 100:.1f} cm from the cabinet: "
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
        righted, placed_once = 0, False
        for rnd in range(0 if placed is not None else 5):
            if grip is None:
                s.phase(f"scan_{rnd}")
                p = self.scan(str(rnd))
                loc = self.locate(p)
                s.require(loc is not None, f"drawer not found: no table blob, compartments {p['comps']}")
                s.decide(f"round {rnd}: drawer at {loc['where']} {np.round(loc['xy'], 3).tolist()}, top {loc['h_top']:.3f} m")
                if loc["where"] == "lying":
                    s.require(righted == 0, "the drawer is still lying after the righting move")
                    self.right_lying(loc)
                    righted += 1
                    self.go_home()
                    if s.dry:
                        s.decide("dry run: righting simulated; the scene did not change")
                        s.claim("fail")
                        return
                    continue
                grip = self.pick(loc, str(rnd))
            if not placed_once and grip.get("axis") is not None:
                delta, aim = self.plan_yaw(grip["axis"], self.target)
            else:
                delta, aim = 0.0, safe_aim(aim, grip.get("axis"), EDGE_BUF[-1])
            self.place(grip, aim, delta)
            placed_once = True
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
