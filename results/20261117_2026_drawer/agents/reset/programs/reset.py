"""Reset for drawer_rl: pull the drawer out of the bottom slot if needed, place it upright at the target.

States (from the side depth):
  table   - a drawer-sized blob in front of the cabinet: pick it by its end wall nearest the robot, place.
  slot    - the drawer front is flush (protrudes < PROTR_GRASP): grasp the pin pull with a tilted gripper,
            pull PULL m out, release; then 'out'.
  out     - the drawer front protrudes >= PROTR_GRASP: top-down grasp of the front wall (jaws along the
            insertion normal), slide it out along the normal, carry it to the target.
After placing: measure, correct up to twice by re-picking from the table, home, measure, claim.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj.perception import tabletop as tt
from rlpj.robot import calib
from rlpj.robot.kin import wrap180
from rlpj.robot.plan import tool_down_T
from rlpj.robot.session import Abort, RobotSession
import scene
from scene import N, TT, OFF, nt, xy_of, obb

HOME_Q = np.array([0, 15, -130, 0, 55, 90.0])   # camera-clear pose for perception
SEED_DOWN = np.array([0, 25, -110, 0, -45, 90.0])
TIP_OPEN = 0.027          # tips below the tool frame, open
PROTR_GRASP = 0.045       # front-wall grasp needs this much drawer in front of the cabinet
PULL = 0.065              # first pull with the pin grasp
PIN_T = scene.SLOT_C - 0.0055   # pin t (wrist triangulation, attempts 9-10: -0.088 with the slot centre at -0.0825)
PLACE_BIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notes", "place_bias.json")
TAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notes", "tab_position.json")
try:                              # 15:52 operator: start from the last measured tab position (same slot geometry)
    with open(TAB_FILE) as _f:
        _tab = json.load(_f)
    TAB_DT0 = float(_tab["t_from_slot"]) - (PIN_T - scene.SLOT_C)       # tab offset from PIN_T that held last time
    if abs(float(_tab["slot_c"]) - scene.SLOT_C) > 0.001 or abs(TAB_DT0) > 0.03:
        TAB_DT0 = None
except Exception:
    TAB_DT0 = None
PIN_H = 0.052             # pin height above the table
PIN_OUT = 0.010           # pad centre in front of the front plane
PIN_BASE = (0.0026, 0.022)   # pre-closed fingertip (out, h) before the tool-frame offset
PIN_SWEEP = [(0.0, 0.0025), (0.002, 0.0015), (0.0, 0.0015), (0.004, -0.001)]   # (dy, dz) tool frame; all pass the keep-out model with the gripper at 0.706 and 0.91
PINCH = 0.90
PIN_TILT = 50.0           # deg, gripper pointing down and into the cabinet
PIN_PHI0 = 20.0
PIN_SWEEP_DEEP = [(0.002, 0.0015), (0.0, 0.0015), (0.002, 0.0), (0.004, -0.001)]   # attempt 86: dz 3-5 mm pressed the face (29-36 N)   # recessed drawer (roll 0, +7 mm)
PIN_PHI = 20.0            # deg roll about the normal
PIN_ZT = 0.041            # fingertip along the tool z when pre-closed
PIN_MARGIN = 0.004        # model fingertip clearance at the pin (model points sit 8 mm outside the real fingers)
PRECLOSE = 0.706          # ~25 mm opening
BACK_PROTR = 0.009        # a slot drawer sticking out more than this is in BACKWARDS (plain back wall outside; a
                          # correct flush drawer reads -0.002..+0.004; 23:50 note: backwards at ~0.014)
BACK_TILT = 40.0          # oblique pinch of the protruding wall's top edge: jaws in the vertical plane through the
BACK_ZC = 0.035           # slot centre, tilted BACK_TILT; pinch centre BACK_ZC along the tool z
BACK_OPEN = 0.45
FLUSH_MAX_DEPTH = 0.008   # face depth beyond this = recessed (attempt 122)
DEEP_MAX = 0.015            # recessed deeper: fail fast (attempts 85-87: 2 cm ungrippable). 17:05 operator: handle <= 1.5 cm
DEEP_VARIANTS = [   # recessed > 1.5 cm. attempt 92: the tab sat ~1-1.5 cm toward +t (image left; camera x = -tool x)
    dict(name="level_dt+0.012", phi=0.0, h_add=0.0035, margin=0.006, dt=0.012, sweep=[(0.002, 0.0015), (0.0, 0.0)]),
    dict(name="level_dt+0.006", phi=0.0, h_add=0.0035, margin=0.006, dt=0.006, sweep=[(0.002, 0.0015), (0.0, 0.0)]),
    dict(name="level_dt+0.018", phi=0.0, h_add=0.0035, margin=0.006, dt=0.018, sweep=[(0.002, 0.0015), (0.0, 0.0)]),
    dict(name="rolled_dt+0.012", phi=20.0, h_add=0.0, margin=0.004, dt=0.012, sweep=[(0.002, 0.0015), (0.0, 0.0025)]),
]
DRAG = 0.03
DEEP_LEVEL = 0.016          # above this the level-finger variant (missed on the robot at 2 cm) would be used: off
T_C = scene.SLOT_C - 0.0045     # drawer centre line along the front (-0.087 before 2026-11-20)
CAB_T = scene.CAB_T             # cabinet body t-span from the task yaml keep-out box
TASK_LAYOUT = scene.TASK.get("layout", {"drawer": {"y": [-0.21, -0.126]}})
# attempt 133: the task's reset_orientation text also demands 'on the table away from the cabinet': at check1 the pull
# faced the robot but the drawer was still at the slot -> 'no' -> a needless turn made it backwards. Orientation only:
ORIENT_Q = ("Judge ONLY which way the wooden drawer is turned, not where it is (it may still be at or partly in the "
            "cabinet slot, or anywhere on the table). The drawer's metal pull - a small grey metal label frame with a "
            "short metal tab below it - is on the outside of the drawer wall that faces the lower left of this image, "
            "toward the robot base.")
SURVEY_GOAL = ("Survey photo from a camera mounted on the robot gripper (the gripper fingers may show at the image "
               "edges). Only decide whether the drawer wall nearest the camera carries the metal pull; ignore where the "
               "drawer is, whether it is held, and the viewing angle.")
WRIST_SIDE_BIAS = np.array([0.010, 0.058])   # 00:38 operator: wrist - side camera table positions, attempts 150-163
STAGE_XY = (0.34, -0.15)        # 18:39 operator: turn the drawer here (>= 25 cm from the cabinet face), then carry it
STAGE_GAP = 0.20                # never place/turn closer than this (centre to cabinet face) except at the target
FACE_GAP_MIN = 0.17           # 16:35 operator: drawer centre >= 17 cm in front of the cabinet face
Y_CENTRE = -0.10                # robot's centre line on the table side (01:12: slide only toward it)
DRIFT_MAX = 0.02                # 01:12: stop sliding/righting attempts after 2 cm of unintended drift
TABLE_OPEN = 0.15          # table pick opening (~72 mm): +-3 cm around a 1 cm wall (attempt 52 missed at 51 mm)
L_DEFAULT = 0.13          # drawer depth until measured
FIRM_MAX = 94.5           # table grasp: reading below this = a firm grasp of the wall (good ones 86-89.5)
HOLD_MAX = 97.0           # gripper reading below this = holding something
TOL = 0.05
GOOD = 0.03               # correct when the error is above this (151/158: 4.4 cm by mine = 6.1 cm by the orchestrator)
TIPPED_H = 0.088          # blob taller than this = drawer lying on a side wall (upright 0.065-0.072, on its side ~0.10)
TIP_CLOSED = 0.047        # closed fingertips below the tool frame
SAFE_H = 0.20             # tool height above the table for transfers
PIN_HI = 0.14             # tool height of the tilted pose above the pin
READY_XYZ = (0.40, -0.10, 0.25)   # tool-down waypoint (above table) between home and the work poses


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def rot2(v, deg):
    a = np.radians(deg); c, s_ = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])


def rot_axis(a, deg):
    a = unit(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


class Reset:
    def __init__(self, s, targets):
        self.s, self.targets = s, targets
        self.tz = calib.table_z()
        self.rng = np.random.default_rng(int(time.time()))
        self.k = 0
        self.backwards = False
        self.final_facing = None
        self.out_judged = None
        self.rgbd = (None, None, None)
        self.pin_dy = self.pin_dz = 0.0
        self.retried = False
        self.deep_cfg = None
        self.from_recess = False
        self.tab_img = None       # last close wrist view of the drawer front (pinch approach)
        self.last_conf = 0.0
        self.place_bias = np.zeros(2)       # learned systematic placement error (measured - aimed), persisted
        self.just_placed = False
        try:
            if not s.dry:
                with open(PLACE_BIAS_FILE) as f:
                    self.place_bias = np.clip(np.array(json.load(f)["bias"], float), -0.06, 0.06)
        except Exception:
            pass
        self.wall_log = []
        self.refine_off = self.refined_last = False
        self.pin_deeper = self.pin_higher = 0.0   # 22:34: extra pinch depth along N / height (m)
        self.knob_t = None
        self.survey_pending = self.survey_placed = False
        self.or_stage = 0         # judge guess sequence for an unseen pull (attempt 124)
        self.tab_dt = TAB_DT0     # tab t offset from PIN_T: saved from the last hold, updated by the wrist measurement
        self.pin_dt = 0.0         # sideways (t) offset of the tab: attempt 92 saw it ~1-1.5 cm toward +t in the recessed drawer
        self.depth = 0.0          # how far the drawer front sits behind flush (pushed in past flush: no stop)
        self.bias = np.array([0.004, -0.005])   # box centre - blob centre (moved camera, attempt 124: (0.002..0.006, -0.004..-0.006); old camera (0.02, 0.03))

    # ------------------------------------------------------------ helpers
    def log(self, m):
        self.s.log(m)

    def q_now(self):
        return np.asarray(self.s.feedback()["q"], float)

    def ik(self, T, seeds=None):
        best = None
        for sd in (seeds or [self.q_now(), SEED_DOWN]):
            try:
                q, _, _ = self.s.planner.ik_exact(T, sd)
            except Exception as e:
                self.log(f"IK from seed failed: {str(e)[:80]}")
                continue
            d = float(np.max(np.abs(q - self.q_now())))
            if best is None or d < best[1]:
                best = (q, d)
        self.s.require(best is not None, f"no IK for tool at {np.round(T[:3, 3], 3).tolist()}")
        return best[0]

    def joint_to(self, name, T, speed="normal"):
        self.s.move_joint(name, self.ik(T), speed)

    def line(self, name, T, speed="slow", tip_margin=0.008):
        self.s.move_line(name, T, speed, tip_margin=tip_margin)

    def to_ready(self):
        T = tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0)
        self.s.move_joint("ready", self.ik(T, [SEED_DOWN]), "normal")

    def go_home(self, name):
        self.to_ready()
        self.s.move_joint(name, HOME_Q, "normal")
        self.set_home()

    def set_home(self):
        """Make HOME_Q the session's (and watchdog's) return pose: a run that started elsewhere (a stopped run left
        the arm low) must not return there on abort (attempt 19 went back down onto the drawer)."""
        s = self.s
        if s.q_home is not None and np.abs(wrap180(np.asarray(s.q_home) - HOME_Q)).max() < 1:
            return
        s.q_home = HOME_Q.copy()
        if getattr(s, "wd", None) is not None:
            try:
                s.wd.stdin.write(("HOME " + " ".join(f"{v:.4f}" for v in HOME_Q) + "\n").encode()); s.wd.stdin.flush()
            except Exception as e:
                s.log(f"watchdog home update failed: {e}")
        s.log("session home set to HOME_Q")

    def escape(self):
        """Start away from home and low (a stopped run or an RL episode left the arm there): get up and clear.
        Tool pointing down (attempts 50-51: 3.5 cm in front of the cabinet, gripper around the drawer front wall):
        open fully first, then straight up, else back off along -N while rising, else straight back then up.
        Other orientations (tilted at the pull): back off along -N and up."""
        s = self.s
        T = self.T_now()
        if T[2, 3] > self.tz + 0.15:
            return
        s.decide(f"start pose low ({T[2, 3] - self.tz:.3f} m above the table): back off and up first")
        # 00:25 operator: attempt 162 rose with the pre-closed pads around the knob (61.9 N): open fully first,
        # whatever the tool direction
        g0 = float(s.feedback()["gripper"])
        if g0 > 3.0:
            s.gripper(0.0, "open fully before backing off (fingers away from any wall / the knob)")
        if T[2, 2] > -0.9:
            # tilted (a pinch pose): back off along the tool axis first, out of the knob, then the usual escape
            for back in (0.04, 0.025):
                Tb = T.copy(); Tb[:3, 3] -= back * T[:3, 2]
                try:
                    self.line("escape_tool_back", Tb, "slow", tip_margin=PIN_MARGIN)
                    T = self.T_now()
                    break
                except Abort as e:
                    if "check failed" not in str(e):
                        raise
                    self.log(f"escape along the tool axis ({back}) refused: {str(e)[:100]}")
        up = max(T[2, 3], self.tz + 0.10) + 0.06
        if T[2, 2] < -0.9:
            cands = [(0.04, T[2, 3]), (0.0, up), (0.04, up)]   # 01:40 note: straight back along -N first
        else:
            cands = [(d, up) for d in (0.08, 0.05, 0.03, 0.0)]
        for d, z in cands:
            T2 = T.copy(); T2[:2, 3] -= d * N; T2[2, 3] = z
            try:
                self.line("escape_up" if z > T[2, 3] else "escape_back", T2, "slow", tip_margin=PIN_MARGIN)
                if z <= T[2, 3]:
                    T3 = T2.copy(); T3[2, 3] = up
                    self.line("escape_up", T3, "slow", tip_margin=PIN_MARGIN)
                return
            except Abort as e:
                self.log(f"escape (back {d}, to {z - self.tz:.3f}) refused: {str(e)[:120]}")
        s.require(False, "no escape path from the start pose")

    def T_now(self):
        return calib.pose_to_T(self.s.feedback()["pose"])

    def shifted(self, dxyz):
        T = self.T_now(); T[:3, 3] += dxyz; return T

    def lift_to(self, name, h, speed="normal"):
        T = self.T_now(); T[2, 3] = self.tz + h; self.line(name, T, speed)

    def save(self, name, img, w=640):
        if img is None:
            return
        h = int(img.shape[0] * w / img.shape[1])
        path = os.path.join(self.s.A, f"{name}.jpg")
        cv2.imwrite(path, cv2.resize(img, (w, h)))
        return path

    # ------------------------------------------------------------ perception
    def perceive(self, tag):
        s = self.s
        D = s.side.depth_median(5)
        K, dist = calib.side_intrinsics(); Tbs = calib.T_base_side()
        Pc, uv = calib.deproject(np.asarray(D, np.float32), K, dist)
        kk, _ = tt.depth_correction(Pc, Tbs)
        P = calib.apply(Tbs, Pc * (kk[0] + kk[1] * Pc[:, 2:3]))
        self.rgbd = (s.side.latest()[0], P, uv)
        tz = tt.table_fit(P); self.tz_fit = tz
        blobs = tt.detect(P, table_z=tz)
        cand = []
        for b in blobs:
            q = nt(b["xy"]); ext = max(b["extent"])
            if q[0] < OFF - 0.03 and 0.03 <= b["height"] <= 0.12 and 0.06 <= ext <= 0.32 and b["n"] > 150:
                pts = b["points"]; hh = pts[:, 2] - tz
                rim = pts[hh > b["height"] - 0.012, :2]            # wall tops: the blob outline is inflated by merges
                c, (L, W), yaw = obb(rim if len(rim) > 60 else pts[:, :2])
                cand.append(dict(xy=[round(float(v), 4) for v in b["xy"]], obb_c=np.round(c, 4).tolist(), L=round(L, 3),
                                 W=round(W, 3), yaw=round(yaw, 1), h=b["height"], n=b["n"], pts=b["points"]))
        cand.sort(key=lambda d: -d["n"])
        # the bottom slot: what is in front of it, and whether the front is there at all
        q = nt(P[:, :2]); h = P[:, 2] - tz
        win = (q[:, 1] > scene.SLOT_T[0] - 0.005) & (q[:, 1] < scene.SLOT_T[1] + 0.005)
        front = win & (h > 0.025) & (h < 0.10) & (q[:, 0] > OFF - 0.30) & (q[:, 0] < OFF + 0.012)
        face = win & (h > 0.035) & (h < 0.085) & (np.abs(q[:, 0] - OFF) < 0.015)
        inside = win & (h > 0.035) & (h < 0.085) & (q[:, 0] > OFF + 0.025) & (q[:, 0] < OFF + 0.10)
        # the drawer front: the nearest dense layer of points in front of the slot
        protr, out = 0.0, None
        fq = q[front]
        if len(fq) > 30:
            protr = float(OFF - np.percentile(fq[:, 0], 3))
            sel = front & (q[:, 0] < OFF - protr + 0.02)
            if protr > 0.02 and sel.sum() > 20:
                # attempts 95-96: the drawer can sit skewed (~30-45 deg) across the opening: fit its front wall from the
                # wall-top points of the protruding part (wide t window), farthest rectangle edge from the cabinet
                ht = float(np.percentile(h[front & (q[:, 0] < OFF - 0.01)], 98))
                band = (q[:, 0] < OFF - 0.005) & (q[:, 0] > OFF - 0.16) & (q[:, 1] > scene.SLOT_C - 0.1175) & (q[:, 1] < scene.SLOT_C + 0.1125) & (h > ht - 0.012) & (h < ht + 0.01)
                self.front_fit = None
                if band.sum() > 40:
                    (cx, cy), (w0, h0), a0 = cv2.minAreaRect(np.ascontiguousarray(P[band, :2].astype(np.float32)))
                    c0 = np.array([cx, cy]); ua = np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))]); va = np.array([-ua[1], ua[0]])
                    edges = [(c0 + ua * w0 / 2, va), (c0 - ua * w0 / 2, va), (c0 + va * h0 / 2, ua), (c0 - va * h0 / 2, ua)]
                    mid, along = min(edges, key=lambda e: float(nt(e[0])[0]))
                    inward = unit(c0 - mid)
                    self.front_fit = dict(mid=mid, inward=inward, yaw=round(ang(inward), 1), size=[round(w0, 3), round(h0, 3)])
                out = dict(t=float(np.median(q[sel, 1])), h_top=ht,
                           h_bot=float(np.percentile(h[sel], 3)), n=int(sel.sum()))
        # drawer pushed in past flush (attempt 80: face points 53, points 2.5-10 cm inside 180): the recessed face
        deep = win & (h > 0.035) & (h < 0.085) & (q[:, 0] > OFF - 0.02) & (q[:, 0] < OFF + 0.10)
        face_depth = float(np.median(q[deep, 0]) - (OFF + 0.002)) if deep.sum() > 40 else None   # 16:45: median 25 mm
        # 22:14: median face position over a wide window (a few outlying points made a ~1 cm-out drawer read 'out')
        wide = win & (h > 0.035) & (h < 0.085) & (q[:, 0] > OFF - 0.10) & (q[:, 0] < OFF + 0.06)
        slot_med = float(np.median(q[wide, 0]) - (OFF + 0.002)) if wide.sum() > 40 else None
        # attempt 98: drawer standing ON THE TABLE pressed against the cabinet front (merged with the cabinet blob, so
        # no candidate): walls reach the table (h < 1.2 cm) in front of the plane; a drawer in the slot hangs >= 1.5 cm
        low = (q[:, 1] > scene.SLOT_C - 0.1775) & (q[:, 1] < scene.SLOT_C + 0.1625) & (q[:, 0] < OFF - 0.012) & (q[:, 0] > OFF - 0.25) & (h > 0.004) & (h < 0.012)
        near_tab = None
        if not cand and low.sum() > 40:
            body = (q[:, 1] > scene.SLOT_C - 0.1775) & (q[:, 1] < scene.SLOT_C + 0.1625) & (q[:, 0] < OFF - 0.012) & (q[:, 0] > OFF - 0.25) & (h > 0.004) & (h < 0.11)
            hb = float(np.percentile(h[body], 98))
            if 0.03 <= hb <= 0.11 and body.sum() > 150:
                pts = P[body]
                rim = pts[(pts[:, 2] - tz) > hb - 0.012, :2]
                c, (L, W), yaw = obb(rim if len(rim) > 60 else pts[:, :2])
                if W < 0.05:      # attempt 117: W 0.032 = the protruding part of a drawer still in the slot
                    self.log(f"near-cabinet blob too thin (W {W:.3f}): not a drawer on the table")
                    L = None
            if 0.03 <= hb <= 0.11 and body.sum() > 150 and L is not None:
                near_tab = dict(xy=np.round(pts[:, :2].mean(0), 4).tolist(), obb_c=np.round(c, 4).tolist(), L=round(L, 3),
                                W=round(W, 3), yaw=round(yaw, 1), h=hb, n=int(body.sum()), pts=pts, near_cabinet=True)
                cand = [near_tab]
        try:
            np.save(os.path.join(s.A, f"cloud_{tag}.npy"), P[(h > -0.01) & (h < 0.2)].astype(np.float32))
        except Exception:
            pass
        lay = front & (q[:, 0] < OFF - protr + 0.012)
        front_top = float(np.percentile(h[lay], 98)) if lay.sum() > 20 else None
        info = dict(tag=tag, table_z=round(tz, 4), front_top=None if front_top is None else round(front_top, 4),
                    face_depth=None if face_depth is None else round(face_depth, 4),
                    slot_med=None if slot_med is None else round(slot_med, 4), blobs=len(blobs), low_pts=int(low.sum()), near_cabinet=near_tab is not None, drawer_blobs=[{k: v for k, v in c.items() if k != "pts"} for c in cand],
                    protr=round(protr, 4), front_pts=int(front.sum()), face_pts=int(face.sum()), inside_pts=int(inside.sum()),
                    out=None if out is None else {k: round(v, 4) for k, v in out.items()})
        self.log(f"perceive {tag}: {info}")
        self.save(f"side_{tag}", s.side.latest()[0])
        return dict(info, cand=cand)

    def state_of(self, p):
        if p["cand"]:
            return "table"
        if p["out"] is not None and p["protr"] >= PROTR_GRASP:
            sm = p.get("slot_med")
            if sm is not None and sm > -0.04:
                return "slot_near"     # median face < 4 cm out: the front wall top is still under the rail (22:14)
            return "out"
        if p["out"] is not None and p["protr"] > BACK_PROTR:
            return "slot_back"           # sticks out 1-4.5 cm (attempt 93 after the recessed pull: 'missing' before)
        if p["face_pts"] > 40 and p["face_pts"] > 2 * p["inside_pts"]:
            if p["protr"] > BACK_PROTR:
                return "slot_back"
            # attempt 122 (new side camera): a drawer ~2 cm past flush read face depth 0.0156 with 241 face points and
            # no inside points (the face window is +-1.5 cm): flush reads -0.002..+0.004
            if p.get("face_depth") is not None and p["face_depth"] > FLUSH_MAX_DEPTH:
                return "slot_deep"
            return "slot"
        if p.get("face_depth") is not None and 0.008 < p["face_depth"] < 0.08:
            return "slot_deep"
        return "missing"

    # ------------------------------------------------------------ pin pull
    def pin_T(self, out_m):
        """Steep pinch pose: tool pointing PIN_TILT from vertical into the cabinet (nearly along the pin, so the pin
        lies in the pad plane), rolled PIN_PHI about the normal; the fingertip at PIN_ZT along z (pre-closed to
        PRECLOSE) is placed out_m in front of the yaml front plane at height PIN_H."""
        th = np.radians(PIN_TILT)
        z = np.r_[np.sin(th) * N, -np.cos(th)]
        x = np.r_[TT, 0.0]
        Rm = rot_axis(np.r_[N, 0.0], PIN_PHI) @ np.stack([x, np.cross(z, x), z], 1)
        margin = self.deep_cfg["margin"] if self.deep_cfg is not None else 0.002
        dshift = max(self.depth - margin, 0.0) if self.depth >= 0 else self.depth   # protruding: shift outward fully
        tip = np.r_[xy_of(OFF - out_m + dshift + self.pin_deeper, self.pin_t + self.pin_dt), self.tz + PIN_H + self.pin_higher]
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = tip - PIN_ZT * Rm[:, 2] + Rm @ np.r_[0.0, self.pin_dy, self.pin_dz]
        return T

    def judge_backwards(self):
        """18:14 operator: decide 'backwards' only from a close look at the drawer front: the wrist image taken at the
        pinch approach goes to the vision judge. Default (no image / judge unsure): the right way round."""
        if self.tab_img is None or self.s.dry:
            return False
        ans = self.ask(self.tab_img, "This close-up from a camera on the robot gripper shows the front wall of a wooden drawer "
                       "in a cabinet slot. That wall carries a metal pull: a grey metal label frame and/or a short metal tab "
                       "or knob is clearly visible on the wood.", "front_pull")
        back = ans is False
        self.s.decide(f"close wrist view: metal pull {'NOT visible -> treat as backwards' if back else 'visible (or unsure) -> right way round'}")
        return back

    def pick_high(self, cands, T_next, gf):
        """First (T, q, label) among cands whose joint move from now and line to T_next pass the model (no motion)."""
        from rlpj.robot import sim
        from rlpj.robot.plan import NORMAL, SLOW
        pl, kin = self.s.planner, self.s.kin
        q0 = self.q_now()
        seeds = [q0, SEED_DOWN] + [np.array([j1, 40, -120, 0, j5, j6], float) for j1 in (0.0, 30.0) for j5 in (20.0, 40.0, 60.0, -40.0) for j6 in (-90.0, 90.0)]
        q_ready = self.ik(tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0), [SEED_DOWN])
        for T, label in cands:
            sols = []
            for sd in seeds:
                try:
                    q, _, _ = pl.ik_exact(T, sd)
                except Exception:
                    continue
                if any(np.abs(wrap180(q - t)).max() < 2 for t in sols):
                    continue
                sols.append(q)
            # unflipped wrist first (22:12: a j4 ~ -170 branch could not get back out past the cabinet)
            for q in sorted(sols, key=lambda v: abs(wrap180(v[3]))):
                try:
                    segb = pl.joint_move("chk_back", q, q_ready, NORMAL)
                    okb, _ = sim.check(segb.q, kin, NORMAL, gripper_frac=gf, table_z=calib.table_z())
                    if not okb:
                        continue
                    seg = pl.joint_move("chk", q0, q, NORMAL)
                    ok, _ = sim.check(seg.q, kin, NORMAL, gripper_frac=gf, table_z=calib.table_z())
                    if not ok:
                        continue
                    seg2 = pl.cart_line("chk2", q, T_next, SLOW)
                    ok2, _ = sim.check(seg2.q, kin, SLOW, gripper_frac=gf, table_z=calib.table_z(), tip_margin=PIN_MARGIN)
                    if ok2:
                        return T, q, label
                except Exception:
                    continue
        return None

    def lift_cands(self, T_out):
        T_up = T_out.copy(); T_up[2, 3] += 0.12
        T_b = T_out.copy(); T_b[:3, 3] -= 0.04 * T_out[:3, 2]; T_b[2, 3] += 0.10
        T_u8 = T_out.copy(); T_u8[2, 3] += 0.08
        T_d = T_out.copy(); T_d[:3, 3] += np.r_[-0.05 * N, 0.04]
        T_u4 = T_out.copy(); T_u4[2, 3] += 0.04
        return [T_up, T_b, T_u8, T_d, T_u4]

    def feasible_pull(self, Tg, pds, gf):
        """Longest pd (m) whose straight pull from Tg along -N and a lift after it pass the model, else None."""
        from rlpj.robot import sim
        from rlpj.robot.plan import SLOW
        pl, kin = self.s.planner, self.s.kin
        try:
            qg = self.ik(Tg)
        except Exception:
            return None
        for pd in pds:
            T_out = Tg.copy(); T_out[:3, 3] -= np.r_[pd * N, 0.0]
            try:
                seg = pl.cart_line("chk_pull", qg, T_out, SLOW)
                ok, _ = sim.check(seg.q, kin, SLOW, gripper_frac=gf, table_z=calib.table_z(), tip_margin=PIN_MARGIN)
                if not ok:
                    continue
                for T_l in self.lift_cands(T_out):
                    seg2 = pl.cart_line("chk_lift", seg.q[-1], T_l, SLOW)
                    if sim.check(seg2.q, kin, SLOW, gripper_frac=0.45, table_z=calib.table_z(), tip_margin=PIN_MARGIN)[0]:
                        return pd
            except Exception:
                continue
        return None

    def pull_held(self, Tg, pull_d):
        """Pull the held tab along -N by the longest distance whose pull AND a lift afterwards pass the model
        (attempt 149 offline: a 6.5 cm flush pull passed but every lift after it hit joint limits)."""
        s = self.s
        from rlpj.robot import sim
        from rlpj.robot.plan import SLOW
        pl, kin = s.planner, s.kin
        gf = 0.9
        q0 = self.q_now()
        cands = {round(max(self.depth, 0.0) + 0.065, 3), round(max(self.depth, 0.0) + 0.045, 3), round(pull_d, 3),
                 0.035, 0.03, 0.025, 0.02}      # attempt 149: from 2 cm proud only short pulls fit (then pull again)
        for pd in sorted(cands, reverse=True):
            T_out = Tg.copy(); T_out[:3, 3] -= np.r_[pd * N, 0.0]
            try:
                seg = pl.cart_line("chk_pull", q0, T_out, SLOW)
                ok, _ = sim.check(seg.q, kin, SLOW, gripper_frac=gf, table_z=calib.table_z(), tip_margin=PIN_MARGIN)
                if not ok:
                    s.decide(f"pull of {pd:.3f} m refused by the model"); continue
                q1 = seg.q[-1]
                lift_ok = False
                for T_l in self.lift_cands(T_out):
                    try:
                        seg2 = pl.cart_line("chk_lift", q1, T_l, SLOW)
                        ok2, _ = sim.check(seg2.q, kin, SLOW, gripper_frac=0.65, table_z=calib.table_z(), tip_margin=PIN_MARGIN)
                    except Exception:
                        ok2 = False
                    if ok2:
                        lift_ok = True
                        break
                if not lift_ok:
                    s.decide(f"pull of {pd:.3f} m: no lift afterwards passes the model"); continue
            except Exception as e:
                s.decide(f"pull of {pd:.3f} m not plannable ({str(e)[:60]})"); continue
            self.line("pull_drawer", T_out, "slow", tip_margin=PIN_MARGIN)
            s.decide(f"pulled the tab {pd:.3f} m out")
            return True, T_out
        return False, None

    class _ForceLimit:
        """00:50 operator: lower the executor's contact-force stop for one approach (never above the task cap)."""
        def __init__(self, sess, limit):
            self.ex = getattr(sess, "executor", None); self.limit = limit; self.prev = None
        def __enter__(self):
            if self.ex is not None and getattr(self.ex, "force_limit", None) is not None:
                self.prev = self.ex.force_limit
                self.ex.force_limit = min(float(self.limit), float(self.prev))
            return self
        def __exit__(self, *a):
            if self.ex is not None and self.prev is not None:
                self.ex.force_limit = self.prev
            return False

    def low_force(self, limit=20.0):
        return Reset._ForceLimit(self.s, limit)

    def contact_line(self, name, T, speed="slow", limit=20.0, tip_margin=0.008):
        """Final approach toward the drawer / knob / table with a low force stop. Returns False on contact (the arm
        stopped there; the caller backs off), True when reached."""
        try:
            with self.low_force(limit):
                self.line(name, T, speed, tip_margin=tip_margin)
            return True
        except Abort as e:
            if "contact force" not in str(e):
                raise
            self.s.decide(f"{name}: contact at the {limit:.0f} N stop ({str(e)[:60]})")
            return False

    def back_off_open(self, name, T_back, limit=25.0):
        """01:17 operator: the pre-closed pads around the knob dragged it on every back-off (73-89 N): open to 0.4,
        then back off under a low force stop (a contact here is tolerated: re-open and try once more)."""
        s = self.s
        s.gripper(0.4, "open before backing off the knob")
        try:
            with self.low_force(limit):
                self.line(name, T_back, "slow", tip_margin=PIN_MARGIN)
        except Abort as e:
            if "contact force" not in str(e):
                raise
            s.decide(f"{name}: contact while backing off: open fully and retry")
            s.gripper(0.0, "open fully")
            self.line(name + "_2", T_back, "slow", tip_margin=PIN_MARGIN)

    def guarded_line(self, name, T_goal, fmax=25.0, last=0.02, step=0.005):
        """00:26 operator: approach the last `last` m to T_goal (along its tool z) in `step` increments, stopping at a
        contact force above fmax (the session's own stop is 65 N). Returns True if T_goal was reached; on contact it
        backs off 1 cm along the tool axis and returns False."""
        s = self.s
        from rlpj.robot.plan import SLOW
        z = T_goal[:3, 2]
        T0 = T_goal.copy(); T0[:3, 3] -= last * z
        # 01:10 operator: the _near move ran at the full 65 N (71.1 N measured in 168): low stop there too
        try:
            with self.low_force(max(fmax, 30.0)):
                self.line(f"{name}_near", T0, "slow", tip_margin=PIN_MARGIN)
        except Abort as e:
            if "contact force" not in str(e):
                raise
            s.decide(f"{name}_near: contact at the low stop: back off")
            Tb = self.T_now(); Tb[:3, 3] -= 0.01 * z
            self.line(f"{name}_off", Tb, "slow", tip_margin=PIN_MARGIN)
            return False
        n = int(round(last / step))
        for k in range(1, n + 1):
            Tk = T_goal.copy(); Tk[:3, 3] -= (last - k * step) * z
            try:
                with self.low_force(fmax):
                    r_ = s.run_segment(s.planner.cart_line(f"{name}_{k}", self.q_now(), Tk, SLOW), tip_margin=PIN_MARGIN)
                f = float((r_ or {}).get("max_force_n") or 0.0)
            except Abort as e:
                if "contact force" not in str(e):
                    raise
                f = fmax + 1.0
            if not s.dry and f > fmax:
                s.decide(f"{name}: contact {f:.1f} N at {(last - k * step) * 1000:.0f} mm before the goal: back off")
                Tb = self.T_now(); Tb[:3, 3] -= 0.01 * z
                self.line(f"{name}_off", Tb, "slow", tip_margin=PIN_MARGIN)
                return False
        return True

    def lift_any(self, name, cands, speed):
        """First model-feasible line among cands (a refused plan does not move the arm)."""
        for j, T in enumerate(cands):
            try:
                self.line(name if j == 0 else f"{name}_alt{j}", T, speed, tip_margin=PIN_MARGIN)
                return
            except (Abort, RuntimeError, ValueError) as e:
                if ("check failed" not in str(e) and "joint jump" not in str(e)) or j == len(cands) - 1:
                    raise
                self.s.decide(f"{name} option {j} refused by the model ({str(e)[:80]}); next option")

    def measure_tab(self, tag):
        """Close wrist view at the pinch pose (fingers pre-closed around the expected tab): the tab is the grey metal
        run (low saturation) on the orange wood between the label frame and the fingertips. Its image-x centre vs
        the pad centre -> t offset (m) to add to the aim (attempt 122: tab ~100 px image-left of the pads = ~+11 mm).
        None if not found / inconsistent."""
        s = self.s
        img = s.wrist.latest()[0]
        if img is None:
            return None
        u_pad, px_per_m, sgn = 625.0, 9500.0, -1.0          # fallback: attempt 122 crops (image-left = +t)
        try:
            from rlpj.perception.wrist import project_base_to_wrist
            T = calib.pose_to_T(s.feedback()["pose"])
            tip = T[:3, 3] + PIN_ZT * T[:3, 2]
            P0 = tip + 0.005 * np.r_[N, 0.0]
            uv = project_base_to_wrist(np.array([tip, P0 - 0.005 * np.r_[TT, 0.0], P0 + 0.005 * np.r_[TT, 0.0]]), T)
            if np.all(np.isfinite(uv)):
                du = (uv[2, 0] - uv[1, 0]) / 0.01
                if 3000 < abs(du) < 30000 and abs(uv[0, 0] - 625.0) < 120:
                    u_pad, px_per_m, sgn = float(uv[0, 0]), abs(du), float(np.sign(du))
        except Exception as e:
            s.log(f"measure_tab: projection unavailable ({str(e)[:80]}), fallback scale")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cents = []
        for y in range(420, 710, 10):     # attempt 140: knob near the image edge, higher up
            if y + 5 > img.shape[0]:
                break
            row = hsv[y - 4:y + 5].astype(float)
            S = np.median(row[..., 1], 0); V = np.median(row[..., 2], 0)
            g = (S < 70) & (V > 70)
            wood = (S > 110) & (V > 40)
            # attempt 132: grey table/tape runs were taken for the tab: a tab run has wood right beside it
            near_wood = np.convolve(wood.astype(float), np.ones(41), "same") > 5
            x, best = 0, None
            while x < len(g):
                if g[x]:
                    x0 = x
                    while x < len(g) and g[x]:
                        x += 1
                    w = x - x0; c = (x0 + x) / 2
                    # 22:27 operator: a run touching the image edge (the grey table beyond the cabinet in 147) is not
                    # the knob, and a knob cut by the edge gives a wrong centre anyway
                    edge = x0 < 5 or x > len(g) - 5
                    flank = (not edge) and near_wood[max(x0 - 25, 0)] and near_wood[min(x + 24, len(g) - 1)]   # wood on both sides
                    if 40 <= w <= 200 and flank and (best is None or w > best[1]):
                        best = (c, w)          # the knob is the widest metal run (frame bars are thin)
                x += 1
            if best is not None:
                cents.append(best)
        # attempt 132 check image: two modes (knob 535-546, frame edge 644-656): keep the group with the most metal
        groups = []
        for c, w in sorted(cents):
            if groups and c - groups[-1][-1][0] < 30:
                groups[-1].append((c, w))
            else:
                groups.append([(c, w)])
        grp = max(groups, key=lambda G: sum(w for _, w in G)) if groups else []
        cents = [c for c, _ in grp]
        res = dict(rows=len(cents), u_pad=round(u_pad, 1), px_per_m=round(px_per_m), cents=[round(c) for c in cents])
        dt = None
        if len(cents) >= 3:
            c = float(np.median(cents)); spread = float(np.median(np.abs(np.array(cents) - c)))
            res.update(u_tab=round(c, 1), spread=round(spread, 1))
            if spread < 25:
                dt_raw = float((c - u_pad) / px_per_m * sgn)
                res["dt_raw"] = round(dt_raw, 4)
                if abs(dt_raw) <= 0.035:          # 22:27: a saturated/implausible offset counts as not found (-> judge)
                    dt = dt_raw
                    res["dt"] = round(dt, 4)
        s.summary(**{f"tab_measure_{tag}": res})
        self.tab_img = self.save(f"wrist_tab_{tag}", img) or self.tab_img
        s.decide(f"wrist tab measurement {tag}: " + (f"tab {dt * 1000:+.1f} mm along t from the aim" if dt is not None else f"not found {res}"))
        return dt

    def explain_refusal(self, T, gf, tag):
        """21:58 operator: log which keep-out box and which gripper point a refused pose enters (model gripper at gf)."""
        try:
            from rlpj.robot import sim
            G = sim.gripper_points(0.085 * (1 - gf), gf)
            P = G @ T[:3, :3].T + T[:3, 3]
            names = ["tip L-", "tip L-hi", "tip L+", "tip L+hi", "tip R-", "tip R-hi", "tip R+", "tip R+hi"] + \
                [f"body z{z:+.2f} a{a:.0f}" for z in (-0.12, -0.08, -0.04) for a in np.linspace(0, 315, 8)]
            hits = []
            for name, c, half, yaw in sim.cell_keep_out():
                ca, sa = np.cos(yaw), np.sin(yaw)
                L = (P - c) @ np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
                inside = np.all(np.abs(L) < half, axis=1)
                for i in np.where(inside)[0]:
                    q = nt(P[i, :2])
                    hits.append(f"{names[i]} in {name} (t {q[1]:.3f}, n-OFF {q[0] - OFF:+.3f}, h {P[i, 2] - self.tz:.3f})")
            self.s.summary(**{f"refusal_{tag}": hits[:12]})
            self.s.decide(f"refused pose {tag}: " + ("; ".join(hits[:4]) if hits else "no keep-out point at the end pose (path/limits)"))
        except Exception as e:
            self.s.log(f"explain_refusal failed: {e}")

    def judge_tab_side(self, tag):
        """20:55 operator: when the grey-run detector finds nothing, ask the vision judge on the close wrist image
        whether the knob is left or right of the gap between the finger pads; step 12 mm that way (image left = +t)."""
        s = self.s
        if self.tab_img is None or s.dry:
            return None
        goal = ("Close-up photo from a camera on a robot gripper whose two dark finger pads are at the bottom of the "
                "image; the robot wants to pinch the small grey metal knob (tab) below a grey metal label frame on a "
                "wooden drawer front. Only judge where the knob is relative to the gap between the two finger pads.")
        for side, dt in (("LEFT", 0.012), ("RIGHT", -0.012)):
            q = (f"The grey metal knob of the drawer is visible and it is clearly to the {side} of the gap between the two "
                 f"finger pads (more than about 1 cm off, not between them).")
            if self.ask(self.tab_img, q, f"tab_side_{tag}_{side.lower()}", goal=goal):
                s.decide(f"judge: knob {side} of the finger gap -> step {dt * 1000:+.0f} mm along t")
                return dt
        s.decide("judge: knob not clearly left or right of the finger gap (or not visible)")
        return None

    def save_tab(self):
        """Persist the tab position that held (slot-centre relative) for later resets (15:52 operator)."""
        if self.s.dry or "/attempt_" not in os.environ.get("RLPJ_ATTEMPT_DIR", ""):   # dry runs / offline tests: never
            return
        try:
            extra = (self.deep_cfg.get("dt", 0.0) - 0.012) if self.deep_cfg is not None else 0.0   # variant spread
            # 00:26: a hold without a fresh wrist measurement saved +6.5 mm (aim -0.043 vs knob -0.055): store the
            # measured knob position when there is one; never a dry-run value
            t_abs = self.knob_t if self.knob_t is not None else float(self.pin_t + self.pin_dt - extra)
            t_abs = t_abs - (0.012 if self.deep_cfg is not None and self.knob_t is not None else 0.0) * 0
            d = dict(slot_c=scene.SLOT_C, t_from_slot=round(float(t_abs - scene.SLOT_C - (0.0 if self.knob_t is not None else 0.0)), 4),
                     depth=round(float(self.depth), 4), when=time.strftime("%Y-%m-%d %H:%M"))
            with open(TAB_FILE, "w") as f:
                json.dump(d, f)
            self.s.log(f"tab position saved: {d}")
        except Exception as e:
            self.s.log(f"tab position not saved: {e}")

    def pull_pin(self, k=0):
        """Pinch the pull tab, sweeping PIN_SWEEP tool-frame offsets (dy, dz) until the gripper holds it; pull PULL
        along -N, release, lift, back to READY. The pinch closes to PINCH so a held tab shows as a lower reading."""
        global PIN_OUT, PIN_H, PIN_PHI
        s = self.s
        PIN_OUT, PIN_H = PIN_BASE
        sweep = PIN_SWEEP
        if self.deep_cfg is not None:
            # recessed drawer variant (21:30: keep trying physical variants): roll, tip height, depth margin, offsets
            c = self.deep_cfg
            PIN_PHI, PIN_H = c["phi"], PIN_BASE[1] + c["h_add"]
            self.pin_dt = c.get("dt", 0.0)
            if self.tab_dt is not None:             # measured this run: centre the variants on it (keep their spread)
                self.pin_dt = self.tab_dt + (c.get("dt", 0.0) - 0.012)
            sweep = c["sweep"]
        elif self.depth > DEEP_LEVEL:
            PIN_PHI, PIN_H = 0.0, PIN_BASE[1] + 0.0035   # attempt 85: +7 mm put the pads on the label plate
            sweep = PIN_SWEEP_DEEP
        else:
            PIN_PHI = PIN_PHI0
            self.pin_dt = 0.0 if self.tab_dt is None else self.tab_dt
        self.pin_t = PIN_T
        self.pin_dy = self.pin_dz = 0.0
        T0 = self.pin_T(PIN_OUT)
        z = T0[:3, 2]
        phi_now = PIN_PHI; PIN_PHI = PIN_PHI0          # the high pose keeps the 20 deg roll (roll 0 there hits joint limits)
        d_now = self.depth; self.depth = max(self.depth, 0.0)    # and stays at the flush position (protruding: j3 folds)
        dt_now = self.pin_dt; self.pin_dt = float(np.clip(dt_now, -0.012, 0.012))   # (a far knob offset: joint limits)
        # 22:12 (slot centre -0.0495): the fixed high pose (roll 20, 14 cm) is out of joint limits there: pick the
        # first (roll, height, back-off, IK branch) whose joint move AND line down to the first pre-pose pass the model
        hi_cands = []
        for d_h in sorted({self.depth, d_now}, reverse=True):     # flush position first, then the real protrusion
            self.depth = d_h
            for phi_h in (PIN_PHI0, 0.0, 35.0, -15.0):
                PIN_PHI = phi_h
                Th0 = self.pin_T(PIN_OUT)
                for hh, back in ((PIN_HI, 0.04), (0.12, 0.04), (0.17, 0.06), (PIN_HI, 0.0), (0.12, 0.08)):
                    T_c = Th0.copy(); T_c[:3, 3] -= 0.05 * Th0[:3, 2]; T_c[2, 3] = self.tz + hh; T_c[:2, 3] -= back * N
                    hi_cands.append((T_c, f"roll {phi_h:.0f} h {hh:.2f} back {back:.2f} depth {d_h:+.3f}"))
        PIN_PHI = phi_now; self.depth = d_now; self.pin_dt = dt_now
        if np.abs(wrap180(self.q_now() - self.ik(tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0), [SEED_DOWN]))).max() > 3:
            self.to_ready()
        self.pin_dy, self.pin_dz = sweep[0]
        Tg0 = self.pin_T(PIN_OUT); T_pre0 = Tg0.copy(); T_pre0[:3, 3] -= 0.05 * Tg0[:3, 2]
        self.pin_dy = self.pin_dz = 0.0
        pick = self.pick_high(hi_cands, T_pre0, PRECLOSE)
        if pick is None and not s.dry:
            s.decide("no model-feasible high pose above the pull tab: skip this pinch")
            return False
        s.require(pick is not None, "no model-feasible high pose above the pull tab")
        T_hi, q, why = pick
        s.decide(f"high pose above the tab: {why}")
        s.gripper(PRECLOSE, "pre-close to ~25 mm for the pull tab")
        s.move_joint("above_pin", q, "normal")
        held, tries = False, []
        for i, (dy, dz) in enumerate(sweep):
            self.pin_dy, self.pin_dz = dy, dz
            Tg = self.pin_T(PIN_OUT)
            T_pre = Tg.copy(); T_pre[:3, 3] -= 0.05 * z
            s.decide(f"pin try {k}.{i}: offset dy {dy:+.3f} dz {dz:+.3f} (tool frame), aim t {self.pin_t + self.pin_dt:.4f}")
            try:
                if i > 0:
                    s.gripper(PRECLOSE, "pre-close again for the next offset")
                self.line(f"to_pin_pre{i}", T_pre, "normal" if i == 0 else "slow", tip_margin=PIN_MARGIN)
                reached = self.guarded_line(f"approach_pin{i}", Tg)
            except Abort as e:
                if "check failed" not in str(e):
                    raise
                s.decide(f"pin offset {i} refused by the model ({str(e)[:90]}); next offset")
                tries.append([dy, dz, "refused"])
                continue
            if not reached:
                tries.append([dy, dz, "contact"])
                self.back_off_open(f"back_pin_c{i}", T_pre)
                if i == 0 and not s.dry:          # re-measure the knob from the pre-pose view before the next offset
                    time.sleep(0.4)
                    dtm = self.measure_tab(f"{k}_contact")
                    if dtm is not None and abs(dtm) > 0.0025:
                        kt = float(self.pin_t + self.pin_dt + dtm)
                        if scene.SLOT_T[0] + 0.01 <= kt <= scene.SLOT_T[1] - 0.01:
                            self.pin_dt += dtm; self.knob_t = kt
                            s.decide(f"after the contact: knob re-measured, aim t {kt:.4f}")
                continue
            if i == 0 and not s.dry:
                # 15:52 operator: the knob needs ~2 mm; locate it in the close wrist view and centre it between the pads.
                # attempt 132: 34 mm off (capped at 25), the check saw the 9.7 mm rest but it was not applied: iterate
                extra = (self.deep_cfg.get("dt", 0.0) - 0.012) if self.deep_cfg is not None else 0.0
                aim0 = float(self.pin_t + self.pin_dt); prev_abs = None; best_dt = self.pin_dt
                for it_m in range(3):
                    time.sleep(0.4)
                    dt_m = self.measure_tab(f"{k}_{it_m}")
                    # 01:17 operator: +7.5, +6.8, +5.7 mm -> the aim walked 2 cm (the knob moved with the pads): a step
                    # must halve the offset, and the total walk stays within 12 mm of the first aim
                    if dt_m is not None and prev_abs is not None and abs(dt_m) > 0.5 * prev_abs:
                        s.decide(f"knob offset {dt_m * 1000:+.1f} mm did not halve ({prev_abs * 1000:.1f} mm before): stop stepping, pinch at the best aim")
                        if abs(self.pin_dt - best_dt) > 1e-4:
                            self.back_off_open(f"back_to_best{it_m}", T_pre)
                            self.pin_dt = best_dt
                            Tg = self.pin_T(PIN_OUT); T_pre = Tg.copy(); T_pre[:3, 3] -= 0.05 * z
                            s.gripper(PRECLOSE, "pre-close again")
                            self.line(f"to_best_pre{it_m}", T_pre, "slow", tip_margin=PIN_MARGIN)
                            if not self.guarded_line(f"approach_best{it_m}", Tg):
                                self.back_off_open(f"back_best_c{it_m}", T_pre)
                        break
                    if dt_m is not None:
                        cap = 0.012 - abs(self.pin_t + self.pin_dt - aim0)
                        if cap <= 0.0005:
                            s.decide("knob centring: 12 mm total walk reached: pinch here"); break
                        dt_m = float(np.clip(dt_m, -cap, cap))
                        prev_abs = abs(dt_m); best_dt = self.pin_dt + dt_m
                    lo_t, hi_t = scene.SLOT_T[0] + 0.01, scene.SLOT_T[1] - 0.01
                    if dt_m is not None:
                        kt = float(self.pin_t + self.pin_dt + dt_m)
                        if lo_t <= kt <= hi_t:            # 22:27: never keep a knob position outside the slot window
                            self.knob_t = kt
                            s.summary(knob_t_measured=round(kt, 4))
                        else:
                            s.decide(f"measured knob t {kt:.4f} is outside the slot window [{lo_t:.3f}, {hi_t:.3f}]: ignored")
                            dt_m = None
                    if dt_m is None:
                        dt_m = self.judge_tab_side(f"{k}_{it_m}")     # 20:55 operator: ask the judge where the knob is
                        if dt_m is not None and not (lo_t <= self.pin_t + self.pin_dt + dt_m <= hi_t):
                            s.decide("judge step would leave the slot window: not taken")
                            dt_m = None
                    if dt_m is None:
                        break
                    if abs(dt_m) <= 0.0025:
                        self.tab_dt = self.pin_dt - extra
                        break
                    tab_old = self.tab_dt
                    self.back_off_open(f"back_for_tab{it_m}", T_pre)
                    self.pin_dt += dt_m                       # aim at the measured tab itself
                    self.tab_dt = self.pin_dt - extra         # later variants keep their spread around it
                    Tg = self.pin_T(PIN_OUT)
                    T_pre = Tg.copy(); T_pre[:3, 3] -= 0.05 * z
                    aim_t = float(self.pin_t + self.pin_dt)
                    s.decide(f"commanded pinch aim t {aim_t:.4f} (yaml slot centre {scene.SLOT_C:.4f}, PIN_T {PIN_T:.4f}, step {dt_m * 1000:+.1f} mm)")
                    try:
                        s.gripper(PRECLOSE, "pre-close again for the approach")
                        self.line(f"to_tab_pre{it_m}", T_pre, "slow", tip_margin=PIN_MARGIN)
                        if not self.guarded_line(f"approach_tab{it_m}", Tg):
                            self.back_off_open(f"back_tab_c{it_m}", T_pre)
                            s.gripper(PRECLOSE, "pre-close again")
                            break
                    except Abort as e:
                        if "check failed" not in str(e):
                            raise
                        self.explain_refusal(Tg, PRECLOSE, f"tab{it_m}")
                        s.decide(f"measured tab position refused by the model ({str(e)[:90]}); back to the previous aim")
                        self.pin_dt -= dt_m; self.tab_dt = tab_old
                        Tg = self.pin_T(PIN_OUT); T_pre = Tg.copy(); T_pre[:3, 3] -= 0.05 * z
                        self.line(f"to_pin_pre{i}b", T_pre, "slow", tip_margin=PIN_MARGIN)
                        self.line(f"approach_pin{i}b", Tg, "slow", tip_margin=PIN_MARGIN)
                        break
            g = s.gripper(PINCH, "pinch the tab")
            img = s.wrist.latest()[0]
            if img is not None:
                cv2.imwrite(os.path.join(s.A, f"wrist_pin_closed{k}_{i}.jpg"), img[360:720, 320:960])
            tries.append([dy, dz, round(g, 1)])
            s.summary(**{f"pin_sweep{k}": tries})
            held = g < PINCH * 100 - 2.5 or s.dry
            if held:
                s.decide(f"tab held at offset {i} (reading {g:.1f})")
                self.save_tab()
                s.gripper(min(g / 100 + 0.008, 0.90), "keep squeezing (sets the model gripper close to the real one)")
                # recessed: pull only to ~2 cm proud (a full 6.5 cm + depth folds j3); the wall-edge clamp takes over
                pull_d = (self.depth + 0.02) if (self.depth > DEEP_LEVEL or self.deep_cfg is not None) else PULL
                if self.depth < 0:      # already protruding (18:14: 1-7 cm out): pull only to ~7 cm out (j3/j5 limits)
                    pull_d = float(np.clip(0.07 + self.depth, 0.025, PULL))
                # 22:12 geometry: pull by the tab as far as the model allows INCLUDING a lift afterwards
                pulled, T_out = self.pull_held(Tg, pull_d)
                if pulled:
                    break
                s.decide(f"no pull distance passed the model at offset {i}; release and try the next offset")
                held = False
            s.gripper(0.4, "open, missed (wide: do not drag the knob)")
            if i == 0 and not s.dry and not held:
                # 22:34 operator: centred along t but the pads closed in front of the knob's end: step deeper along N
                # (and a few mm higher) until the pinch holds; the reset may reach 4 cm into the slot opening
                for deeper, higher in ((0.004, 0.0), (0.008, 0.0), (0.008, 0.003), (0.012, 0.0), (0.012, 0.003)):
                    self.pin_deeper, self.pin_higher = deeper, higher
                    Tgd = self.pin_T(PIN_OUT)
                    # 01:10 operator: back out to this target's pre-pose first, never shove the pads sideways
                    T_pre_d = Tgd.copy(); T_pre_d[:3, 3] -= 0.05 * Tgd[:3, 2]
                    try:
                        self.back_off_open(f"deeper_pre_{int(deeper * 1000)}_{int(higher * 1000)}", T_pre_d)
                        s.gripper(PRECLOSE, "pre-close for the deeper pinch")
                    except Abort as e:
                        if "check failed" not in str(e):
                            raise
                        s.decide(f"deeper pre-pose refused ({str(e)[:60]})"); continue
                    s.decide(f"pinch {deeper * 1000:.0f} mm deeper, {higher * 1000:.0f} mm higher (aim t {self.pin_t + self.pin_dt:.4f})")
                    try:
                        if not self.guarded_line(f"deeper_pin_{int(deeper * 1000)}_{int(higher * 1000)}", Tgd, last=0.02, step=0.005):
                            s.decide("contact while going deeper: stop stepping deeper")
                            break
                    except Abort as e:
                        if "check failed" not in str(e):
                            raise
                        s.decide(f"deeper pinch refused by the model ({str(e)[:80]})")
                        continue
                    g = s.gripper(PINCH, "pinch the tab (deeper)")
                    img = s.wrist.latest()[0]
                    if img is not None:
                        cv2.imwrite(os.path.join(s.A, f"wrist_pin_deep{k}_{int(deeper * 1000)}_{int(higher * 1000)}.jpg"), img[360:720, 320:960])
                    tries.append([dy, dz, deeper, higher, round(g, 1)])
                    s.summary(**{f"pin_sweep{k}": tries})
                    if g < PINCH * 100 - 2.5:
                        held = True
                        Tg = Tgd
                        s.decide(f"tab held {deeper * 1000:.0f} mm deeper (reading {g:.1f})")
                        break
                    s.gripper(0.4, "open, missed (deeper)")
                if held:
                    self.save_tab()
                    s.gripper(min(g / 100 + 0.008, 0.90), "keep squeezing (sets the model gripper close to the real one)")
                    pull_d = (self.depth + 0.02) if (self.depth > DEEP_LEVEL or self.deep_cfg is not None) else PULL
                    pulled, T_out = self.pull_held(Tg, pull_d)
                    if pulled:
                        break
                    held = False
                    s.gripper(PRECLOSE, "release, pull refused")
                self.pin_deeper = self.pin_higher = 0.0
            self.back_off_open(f"back_pin{i}", T_pre)
        if held:
            s.gripper(0.65, "release the tab (partly, keeps the lower finger off the table)")
            # the drawer now sticks out (the model does not know it): lift 12 cm straight up so the joint move to
            # READY passes above it (attempt 17 hit the drawer after a 6 cm lift)
            self.lift_any("up_from_pin", self.lift_cands(T_out), "slow")    # (not toward the cabinet: the drawer sticks out)
        else:
            T_hi2 = T_pre.copy(); T_hi2[2, 3] = self.tz + PIN_HI; T_hi2[:2, 3] -= 0.04 * N
            T_st = T_pre.copy(); T_st[2, 3] += 0.10
            # attempts 85/122 (model): rolled high pose hits joint limits for recessed: lift straight up first
            cands = [T_st, T_hi2, T_hi] if (self.depth > DEEP_LEVEL or self.deep_cfg is not None) else [T_hi2, T_st, T_hi]
            self.lift_any("up_from_pin", cands, "normal")
        # 22:12 geometry: the joint move to READY from beside the cabinet can sweep through its keep-out: back off
        # straight away from the cabinet and up first
        Tn = self.T_now()
        T_r1 = Tn.copy(); T_r1[:3, 3] += np.r_[-0.10 * N, 0.0]; T_r1[2, 3] = max(Tn[2, 3], self.tz + 0.16)
        T_r2 = Tn.copy(); T_r2[:3, 3] += np.r_[-0.06 * N, 0.0]; T_r2[2, 3] = max(Tn[2, 3], self.tz + 0.12)
        try:
            self.lift_any("retreat_from_cabinet", [T_r1, T_r2], "normal")
        except (Abort, RuntimeError, ValueError) as e:
            if "check failed" not in str(e) and "joint jump" not in str(e):
                raise
            s.decide(f"retreat line refused ({str(e)[:80]}); joint move to READY directly")
        self.to_ready()
        return held

    def press_drag(self, k=0):
        """Recessed drawer, last resort: closed fingertips (level, tilted 50 deg) set on the top of the pull tab's outer
        end, pressed ~3 mm down, dragged DRAG out along -N slowly (friction), lifted straight up."""
        global PIN_PHI, PIN_H, PIN_OUT
        s = self.s
        PIN_PHI, PIN_OUT, PIN_H = 0.0, PIN_BASE[0], 0.031 + 0.004 * k      # tip end ~on the tab top (h ~0.03)
        self.pin_t, self.pin_dy, self.pin_dz = PIN_T, 0.0, 0.0
        self.pin_dt = 0.012                                                  # attempt 92: tab ~1.2 cm toward +t (image left = +t: camera x = -tool x)
        if self.knob_t is not None:          # attempt 143: the pinch at the measured knob was refused; drag there instead
            self.pin_dt = self.knob_t - PIN_T
        self.deep_cfg = dict(margin=-0.002)                                  # 2 mm deeper than the tab end
        Tg = self.pin_T(PIN_OUT)
        z = Tg[:3, 2]
        Tg[:3, 3] += (0.047 - PIN_ZT) * z * -1.0                              # closed tips reach 6 mm further than pre-closed
        T_pre = Tg.copy(); T_pre[2, 3] += 0.03
        PIN_PHI = PIN_PHI0
        T0h = self.pin_T(PIN_OUT); PIN_PHI = 0.0
        T_hi = T0h.copy(); T_hi[:3, 3] -= 0.05 * T0h[:3, 2]; T_hi[2, 3] = self.tz + PIN_HI; T_hi[:2, 3] -= 0.04 * N
        s.decide(f"press-drag try {k}: closed tips onto the tab top at depth {self.depth:.3f}, press 3 mm, drag {DRAG} m out")
        self.to_ready()
        s.gripper(1.0, "close for press-drag")
        q = self.ik(T_hi, [np.array([0, 40, -120, 0, j5, j6], float) for j5 in (20.0, 40.0, 60.0) for j6 in (-90.0, 90.0)])
        s.move_joint("above_tab_drag", q, "normal")
        self.line("to_tab_drag_pre", T_pre, "slow", tip_margin=PIN_MARGIN)
        if not self.contact_line("touch_tab", Tg, "slow", limit=20.0, tip_margin=PIN_MARGIN):
            s.decide("touch_tab: contact early: press from here")
            Tg = self.T_now()
        # 22:34 operator: a fixed 3 mm press never loaded the tab (1.5 N): press in 1 mm steps until the force rises
        # by ~6 N (or 6 mm)
        T_press = Tg.copy()
        f0 = None
        from rlpj.robot.plan import SLOW
        for mm in range(1, 7):
            T_n = T_press.copy(); T_n[2, 3] -= 0.001
            try:
                r_ = s.run_segment(s.planner.cart_line(f"press_tab_{mm}", self.q_now(), T_n, SLOW), tip_margin=PIN_MARGIN)
            except Abort as e:
                if "check failed" not in str(e):
                    raise
                s.decide(f"press step {mm} refused by the model ({str(e)[:80]})"); break
            T_press = T_n
            f = float((r_ or {}).get("max_force_n") or 0.0)
            f0 = f if f0 is None else min(f0, f)
            s.log(f"press {mm} mm: max force {f:.1f} N")
            if not s.dry and f - f0 >= 6.0:
                s.decide(f"tab loaded after {mm} mm press ({f:.1f} N)"); break
        img = s.wrist.latest()[0]
        if img is not None:
            cv2.imwrite(os.path.join(s.A, f"wrist_press{k}.jpg"), img[360:720, 320:960])
        T_out = T_press.copy(); T_out[:3, 3] -= np.r_[DRAG * N, 0.0]
        self.line("drag_out", T_out, "slow", tip_margin=PIN_MARGIN)
        T_up = T_out.copy(); T_up[2, 3] += 0.10
        self.line("up_from_drag", T_up, "slow", tip_margin=PIN_MARGIN)
        self.deep_cfg = None
        self.to_ready()

    # ------------------------------------------------------------ backwards drawer: pinch the protruding wall's top
    def back_T(self, protr, top, dn=0.0):
        th = np.radians(BACK_TILT)
        z = np.r_[np.sin(th) * N, -np.cos(th)]; x = np.r_[np.cos(th) * N, np.sin(th)]
        Pc = np.r_[xy_of(OFF - (protr - 0.005) + dn, T_C), self.tz + top - 0.010]
        T = np.eye(4); T[:3, :3] = np.stack([x, np.cross(z, x), z], 1); T[:3, 3] = Pc - BACK_ZC * z
        return T

    def pull_back(self, p):
        """The drawer sits backwards (plain wall outside, ~1.4 cm proud, no pull): clamp the protruding wall's top edge
        diagonally (outer finger on its outer face, inner finger over the top into the drawer; jaws in the vertical
        plane through the slot centre, which the slot-opening keep-out allows) and pull it PULL out."""
        s = self.s
        top = float(np.clip(p.get("front_top") or 0.072, 0.06, 0.085))
        protr = float(p["protr"])
        held = False
        for i, dn in enumerate((0.0, 0.003)):
            Tg = self.back_T(protr, top, dn)
            z = Tg[:3, 2]
            T_pre = Tg.copy(); T_pre[:3, 3] -= 0.05 * z
            T_hi = T_pre.copy(); T_hi[2, 3] += 0.05
            s.decide(f"backwards drawer: oblique pinch of the outer wall top (protr {protr:.3f}, top {top:.3f}, dn {dn:+.3f})")
            self.to_ready()
            s.gripper(BACK_OPEN, "open for the wall edge")
            try:
                q = self.ik(T_hi, [np.array([0, 40, -120, 0, j5, j6], float) for j5 in (20.0, 40.0, 60.0) for j6 in (-90.0, 90.0)])
                s.move_joint("above_back_wall", q, "normal")
                self.line("to_back_pre", T_pre, "normal")
            except Abort as e:
                if "check failed" not in str(e) and "no IK" not in str(e):
                    raise
                s.decide(f"wall-edge clamp approach refused by the model ({str(e)[:80]}): skip the clamp")
                break
            # attempt 149: the clamp held (83.4) and THEN its pull was refused -> stuck on the wall. Plan first.
            pull_d0 = float(np.clip(0.085 - protr, 0.03, PULL))
            pd_ok = self.feasible_pull(Tg, sorted({pull_d0, 0.04, 0.03, 0.02}, reverse=True), 0.95)
            if pd_ok is None and not s.dry:
                s.decide("wall-edge clamp: no pull distance passes the model from here: skip the clamp")
                self.line("up_from_wall_edge", T_hi, "normal")
                break
            self.line("approach_back_wall", Tg, "slow", tip_margin=PIN_MARGIN)
            g = s.gripper(1.0, "clamp the wall edge")
            img = s.wrist.latest()[0]
            if img is not None:
                cv2.imwrite(os.path.join(s.A, f"wrist_back_closed{i}.jpg"), img[360:720, 320:960])
            s.summary(**{f"back_grip{i}": round(g, 1)})
            held = g < HOLD_MAX or s.dry
            if held:
                s.gripper(min(g / 100 + 0.01, 0.95), "keep squeezing (model gripper close to the real one)")
                pull_d = pd_ok if pd_ok is not None else float(np.clip(0.085 - protr, 0.03, PULL))   # model-checked above
                T_out = Tg.copy(); T_out[:3, 3] -= np.r_[pull_d * N, 0.0]
                self.line("pull_back_drawer", T_out, "slow", tip_margin=PIN_MARGIN)
                s.gripper(BACK_OPEN, "release the wall")
                T_up = T_out.copy(); T_up[2, 3] += 0.10 if protr < 0.025 else 0.05   # 10 cm: wrist singularity from 3-4 cm out
                self.lift_any("up_from_back", [T_up] + self.lift_cands(T_out), "slow")
                break
            s.decide(f"wall edge not held (reading {g:.1f})")
            s.gripper(BACK_OPEN, "open, missed")
            self.line("back_from_wall", T_pre, "slow", tip_margin=PIN_MARGIN)
            self.line("up_from_wall_edge", T_hi, "normal")
        self.to_ready()
        return held

    def tab_score(self, img, T, e, d_out, h, s0=0.006, s1=0.03, half=0.03):
        """Share of 'pull tab' pixels (metal: S 60..175, V < 165; table is S < 50, wood S > 175) in a band outside a
        wall: from s0 to s1 beyond the point e along d_out, +-half along the wall, at height h (base frame).
        Attempt 28 (pull wall) 0.10, attempt 41 (plain wall) 0.00."""
        from rlpj.perception.wrist import project_base_to_wrist
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        w = np.array([-d_out[1], d_out[0]])
        P = np.array([np.r_[e + sv * d_out + a * w, self.tz + h] for sv in np.linspace(s0, s1, 13) for a in np.linspace(-half, half, 25)])
        uv = project_base_to_wrist(P, T)
        ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0] - 1)
        if ok.sum() < 50:
            return None, uv[ok]
        v = hsv[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].astype(float)
        return float(((v[:, 1] > 60) & (v[:, 1] < 175) & (v[:, 2] < 165)).mean()), uv[ok]

    def overview_pull(self, c, tag):
        """Wrist view from ~28 cm above the drawer on the table; score the band outside each of the 4 walls for the
        pull tab; a second view at the opposite wrist yaw if unclear. Returns unit vector centre -> pull wall or None."""
        s = self.s
        ctr = np.array(c["obb_c"]); u = np.array([np.cos(np.radians(c["yaw"])), np.sin(np.radians(c["yaw"]))]); v = np.array([-u[1], u[0]])
        walls = [(u, c["L"] / 2), (-u, c["L"] / 2), (v, c["W"] / 2), (-v, c["W"] / 2)]
        best = None
        for k, yaw in enumerate((ang(u), ang(u) + 180.0)):
            if s.dry:
                break
            T = tool_down_T(np.r_[ctr, self.tz + 0.28], wrapd(yaw))
            self.to_ready()
            self.joint_to(f"overview{k}", T)
            time.sleep(0.5)
            img = s.wrist.latest()[0]
            if img is None:
                continue
            Tm = calib.pose_to_T(s.feedback()["pose"])
            scores, vis = [], img.copy()
            for d, hlf in walls:
                sc, uvs = self.tab_score(img, Tm, ctr + d * (hlf - 0.012), d, 0.02, s0=0.0, s1=0.035)
                scores.append(sc)
                for q in uvs[::3]:
                    cv2.circle(vis, (int(q[0]), int(q[1])), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(s.A, f"overview_{tag}{k}.jpg"), cv2.resize(vis, (640, 360)))
            sc = [(-1 if x is None else x) for x in scores]
            order = np.argsort(sc)[::-1]
            s.summary(**{f"overview_{tag}{k}": [round(x, 3) for x in sc]})
            s.log(f"overview {tag}{k}: tab scores {[round(x, 3) for x in sc]}")
            if sc[order[0]] > 0.04 and sc[order[0]] > 2.5 * max(sc[order[1]], 0.005):
                best = walls[order[0]][0]
                break
        self.to_ready()
        return best

    # ------------------------------------------------------------ orientation by the orchestrator's vision judge
    def side_full(self, tag):
        """Full-size settled side image (the session has had the camera open for a while; wait a little anyway)."""
        time.sleep(1.0)
        img = self.s.side.latest()[0]
        path = os.path.join(self.s.A, f"vlm_{tag}.jpg")
        if img is not None:
            m = float(img.mean()) / 255.0
            if 0.05 < m < 0.4:                 # dim room (attempt 76: mean 77 vs 116): gamma-brighten to ~0.47
                g = np.log(0.47) / np.log(m)
                img = cv2.LUT(img, ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8))
                self.s.log(f"side image dim (mean {m * 255:.0f}): gamma {g:.2f} for the judge")
            cv2.imwrite(path, img)
        return path if img is not None else None

    def ask(self, path, success, tag, goal=None):
        try:
            from rlpj import config
            from rlpj.success import vlm_judge
            t = config.load()["task"]
            r = vlm_judge.judge({"after": path}, dict(goal=goal or t.get("reset", ""), success=success), timeout=120)
        except Exception as e:
            r = dict(success=None, reason=f"judge error {e}")
        self.s.log(f"vision judge {tag}: success={r.get('success')} conf={r.get('confidence')} reason={str(r.get('reason'))[:300]}")
        self.s.summary(**{f"vlm_{tag}": dict(success=r.get("success"), conf=r.get("confidence"), reason=str(r.get("reason"))[:200])})
        try:
            self.last_conf = float(r.get("confidence") or 0.0)
        except Exception:
            self.last_conf = 0.0
        return r.get("success")

    def walls_of(self, c):
        ctr = np.array(c["obb_c"]); u = np.array([np.cos(np.radians(c["yaw"])), np.sin(np.radians(c["yaw"]))]); v = np.array([-u[1], u[0]])
        return ctr, [(u, c["L"] / 2), (-u, c["L"] / 2), (v, c["W"] / 2), (-v, c["W"] / 2)]

    def image_label(self, ctr, d):
        """Image-quadrant words ('lower right', ...) for the wall of the drawer at ctr whose outward normal is d."""
        K, dist = calib.side_intrinsics(); Tsb = np.linalg.inv(calib.T_base_side())
        P = np.array([np.r_[ctr, self.tz + 0.035], np.r_[ctr + 0.07 * d, self.tz + 0.035]])
        uv = calib.project(calib.apply(Tsb, P), K, dist)
        du, dv = uv[1] - uv[0]
        return ("lower" if dv > 0 else "upper") + " " + ("right" if du > 0 else "left")

    def vlm_pull(self, c, tag):
        """(ok, pull) from the vision judge: ok True when the pull already faces the robot; otherwise the pull wall
        (unit vector) if seen on the other visible wall, else the wall opposite the robot-facing one (it is on a
        far wall: a 180 deg turn brings it into view). (None, None) when the judge is unavailable."""
        path = self.side_full(tag)
        if path is None:
            return None, None
        from rlpj import config
        t = config.load()["task"]
        ok = self.ask(path, ORIENT_Q, f"{tag}_q1")
        if ok is None:
            return None, None
        ctr, walls = self.walls_of(c)
        robot = max(walls, key=lambda w: float(w[0] @ unit(-N)))[0]
        if ok:                                   # the pull is on the robot-facing wall: keep it there when re-placing
            return True, robot
        cam = calib.T_base_side()[:3, 3][:2]
        others = [w for w in walls if w[0] is not robot]
        vis = max(others, key=lambda w: float(w[0] @ unit(cam - ctr)))[0]
        lab = self.image_label(ctr, vis)
        q2 = (f"The drawer's metal pull (a small grey metal label frame with a short tab below it) is on the outside of the "
              f"drawer wall that faces the {lab} of this image.")
        on_vis = self.ask(path, q2, f"{tag}_q2")
        if on_vis:
            msg, pull = f"pull on the visible {lab} wall", vis
        else:
            # attempts 77/78: in dim light the same scene got 'yes' (0.6) and 'no' (0.75): one more look before a turn
            path2 = self.side_full(tag + "b")
            if path2 is not None and self.ask(path2, ORIENT_Q, f"{tag}_q1b"):
                self.s.decide("vision judge: second look confirms the pull faces the robot")
                return True, robot
            # attempt 124 (new side camera): the lower-right wall is cut off by the frame bottom (q2 'can't see'), and
            # repeated 180 deg guesses just swapped a side-wall pull between the two side walls. Guess sequence over
            # the picks: far wall (180), then the camera-side side wall (90), then far again (covers the 4th wall).
            stage = self.or_stage; self.or_stage += 1
            if stage % 3 == 1:
                msg, pull = f"pull not on the robot-facing or far wall (180 deg turn did not show it): assume the {lab} side wall, turn 90 deg", vis
            else:
                msg, pull = "pull on a far (hidden) wall: assume opposite the robot-facing wall, turn 180 deg and re-check", -robot
        self.s.decide(f"vision judge: {msg}")
        return False, pull

    # ------------------------------------------------------------ tipped drawer
    def wrist_find(self, tag):
        """attempt 121: the drawer ended off the side camera's view (near edge, y < ~-0.25). Look down with the wrist
        camera, find the brown wood on the grey table (grid of table points at h 0.09 projected into the image), and
        slide it back toward the middle with a low push on its near (-y) side. Returns True if it pushed."""
        s = self.s
        if s.dry:
            return False
        from rlpj.perception.wrist import project_base_to_wrist
        T = tool_down_T(np.r_[0.42, -0.24, self.tz + 0.40], 0.0)
        self.to_ready()
        self.joint_to("look_down_wrist", T)
        time.sleep(0.8)
        img = s.wrist.latest()[0]
        if img is None:
            return False
        self.save(f"wrist_find_{tag}", img)
        Tm = calib.pose_to_T(s.feedback()["pose"])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 90) & (hsv[..., 2] > 40)
        xs = np.arange(0.25, 0.63, 0.005); ys = np.arange(-0.43, -0.02, 0.005)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        G = np.stack([X.ravel(), Y.ravel(), np.full(X.size, self.tz + 0.09)], 1)
        uv = project_base_to_wrist(G, Tm)
        ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        occ = np.zeros(X.size, np.uint8)
        occ[ok] = mask[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        q = nt(G[:, :2])
        occ[(q[:, 0] > OFF - 0.03) & (q[:, 1] > CAB_T[0] - 0.025) & (q[:, 1] < CAB_T[1] + 0.025)] = 0        # the cabinet
        occ = occ.reshape(X.shape)
        n, lab, st, cen = cv2.connectedComponentsWithStats(occ, connectivity=8)
        if n < 2:
            s.decide("wrist look: no wood on the table"); return False
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        area = int(st[k, cv2.CC_STAT_AREA])
        cells = np.argwhere(lab == k)
        xy = np.stack([xs[cells[:, 0]], ys[cells[:, 1]]], 1)
        c = xy.mean(0)
        s.summary(**{f"wrist_find_{tag}": dict(area_cells=area, centre=np.round(c, 3).tolist(), y_min=round(float(xy[:, 1].min()), 3),
                                               seen_frac=round(float(ok.mean()), 2))})
        if area < 120:
            s.decide(f"wrist look: wood blob too small ({area} cells)"); return False
        s.decide(f"wrist look: drawer at {np.round(c, 3).tolist()} (y {xy[:, 1].min():.3f}..{xy[:, 1].max():.3f})")
        y_goal = float(np.mean(TASK_LAYOUT["drawer"]["y"]))
        if c[1] > y_goal - 0.05:
            return False
        sel = np.abs(xy[:, 0] - c[0]) < 0.06          # the whole width under the gripper body
        y_min = float(xy[sel, 1].min()) if sel.any() else float(xy[:, 1].min())
        dist = float(np.clip(y_goal - c[1], 0.03, 0.12))
        d = np.array([0.0, 1.0])                                  # toward the robot's centre line (01:12 rule)
        if not self.slide_ok(c, d, dist) and c[1] + dist > Y_CENTRE:
            s.decide("wrist look: slide not allowed by the edge rule"); return False
        if self.slide(np.array([c[0], y_min]), d, dist, "drawer off the side camera's view") is False:
            return False
        return True

    def open_side(self, c):
        """Depth test for a drawer lying on its side (attempt 121: the judge said 'open side toward the camera' for a
        drawer whose camera-facing face was the solid bottom panel; 120 said the opposite). Returns (o, ctr_m, D):
        o = unit open-side direction, ctr_m = centre of the depth span, D = depth (from the top wall's points).
        On the camera-facing face: points >= 2 cm inside the face plane at mid heights = an open side."""
        s = self.s
        ctr, walls = self.walls_of(c)
        u, v = walls[0][0], walls[2][0]
        P = np.asarray(c["pts"], float)
        h = P[:, 2] - self.tz; h_top = float(c["h"])
        d = P[:, :2] - ctr; a_ = d @ u; b_ = d @ v
        top = (h > h_top - 0.015) & (np.abs(a_) < 0.06)
        if top.sum() > 30:
            b_lo, b_hi = float(np.percentile(b_[top], 3)), float(np.percentile(b_[top], 97))
        else:
            b_lo, b_hi = -float(c["W"]) / 2, float(c["W"]) / 2
        D = float(np.clip(b_hi - b_lo, 0.055, 0.09)); b_m = (b_lo + b_hi) / 2
        ctr_m = ctr + v * b_m
        cam = calib.T_base_side()[:3, 3][:2] if not s.dry else np.array([0.3, -0.9])
        sc = 1.0 if float(v @ (cam - ctr_m)) >= 0 else -1.0
        mid = (np.abs(a_) < 0.04) & (h > 0.02) & (h < h_top - 0.025)
        depth_in = sc * (b_m + sc * D / 2 - b_)            # distance inside the camera-side face plane
        face = mid & (depth_in > -0.02) & (depth_in < 0.012)
        inside = mid & (depth_in >= 0.02) & (depth_in < D + 0.01)
        n_f, n_i = int(face.sum()), int(inside.sum())
        is_open = n_i > 40 and n_i > 0.3 * (n_f + n_i)
        o = sc * v if is_open else -sc * v
        fb = float(np.median(b_[face])) if n_f > 30 else b_m + sc * D / 2
        self.cam_face = ctr + v * fb                      # camera-side face (well measured: seen straight on)
        self.cam_face_is_open = is_open
        s.summary(open_side=dict(face=n_f, inside=n_i, cam_side_open=is_open, D=round(D, 3), b=[round(b_lo, 3), round(b_hi, 3)]))
        s.decide(f"tipped drawer depth test: camera-side face {n_f} pts, {n_i} pts >= 2 cm inside -> open side "
                 f"{'toward' if is_open else 'away from'} the camera ({ang(o):.0f} deg), depth {D:.3f}")
        return o, ctr_m, D

    def landing_ok(self, l):
        """Drawer centre l (xy) safe on the table: off the cabinet front, not toward the table edges (01:12: the
        drawer went off the table beyond y ~ -0.25 before the 2026-11-20 base shift), within the arm's easy reach."""
        ql = nt(l)
        if ql[0] > OFF - 0.10 and CAB_T[0] - 0.055 < ql[1] < CAB_T[1] + 0.065:
            return False
        # 16:35: the moved side camera cuts off y < ~-0.18 (attempt 124: drawer at y -0.19 cut off at the frame bottom)
        return -0.17 <= l[1] <= 0.05 and 0.30 <= l[0] <= 0.56

    def slide_ok(self, ctr, d, dist):
        """01:12 rule: a slide must move the drawer toward the robot's centre line (y ~ -0.1), never toward an edge."""
        end = ctr + d * dist
        # the edge the drawer went off (01:12) is the -y side; +y is the table middle / the new target area (16:35)
        toward_edge = d[1] < -0.2 and end[1] < Y_CENTRE
        return (not toward_edge) and self.landing_ok(end) and dist <= 0.125

    def right_side(self, c, tag):
        """Drawer lying on a side wall (00:43 / 01:01): it is light and the table slippery, so pushes slide it and a
        pinch across its ~7.5 cm depth misses (85 mm stroke). Operator option (b): hook the closed fingertips inside
        the opening under the upper wall's free edge and lift along an arc about the far bottom edge (bottom-panel
        side): it rolls bottom-down. First slide it (low push on the solid bottom panel) if the landing zone would be
        out of the side camera's view or at the cabinet."""
        s = self.s
        o, ctr, D = self.open_side(c)
        ctr0 = ctr.copy()
        h_top = float(c["h"])
        land = ctr - o * (D / 2 + h_top / 2)               # centre of the upright drawer after the roll
        need = 0.0
        for k in range(6):
            if self.landing_ok(land + o * need):
                break
            need += 0.02
        s.require(self.landing_ok(land + o * need), f"no safe landing zone for the tipped drawer at {np.round(ctr, 3).tolist()} "
                  f"(open side {ang(o):.0f} deg): needs a person")
        if need > 0:
            # 01:12 operator: never slide it toward a table edge; only toward the robot's centre line (y ~ -0.1)
            s.require(self.slide_ok(ctr, o, need), f"landing zone needs a {need:.2f} m slide toward {ang(o):.0f} deg, "
                      "away from the robot's centre line (toward a table edge): not allowed, needs a person")
            push_face = self.cam_face - o * 0.07 if self.cam_face_is_open else self.cam_face
            self.slide(push_face, o, need, "tipped drawer: landing zone at the cabinet / table edge")
            ctr = ctr + o * need
        self.right_shift = ctr - ctr0
        D = 0.07                                           # operator: ~7 cm deep (the top-wall span reads short)
        f = self.cam_face + (ctr - ctr0)                   # camera-side face, moved with any slide
        edge = f if self.cam_face_is_open else f + o * D   # upper wall's free edge (open side)
        P = np.r_[edge - o * D, self.tz]                   # far bottom edge (bottom-panel side): the pivot
        best = None
        for az in (0.0, 35.0, -35.0, 60.0, -60.0):
            for tilt in (70.0, 60.0):
                for arc in (60.0, 50.0):
                    seq = self.hook_seq(edge, o, az, h_top, tilt, P, arc)
                    ok, why = self.seq_ok(seq)
                    s.log(f"hook plan az {az:.0f} tilt {tilt:.0f} arc {arc:.0f}: {'ok' if ok else why[:150]}")
                    if ok:
                        best = (az, tilt, arc, seq)
                        break
                if best:
                    break
            if best:
                break
        s.require(best is not None, "no model-feasible hook-and-lift for the tipped drawer")
        az, tilt, arc, seq = best
        s.decide(f"tipped drawer at {np.round(ctr, 3).tolist()} (height {h_top:.3f}, depth {D:.3f}): hook under the upper "
                 f"wall from the open side ({ang(o):.0f} deg, approach az {az:.0f}), tool tilt {tilt:.0f}, lift {arc:.0f} deg about the far bottom edge")
        s.gripper(1.0, "close the fingers (hook)")
        self.to_ready()
        for kind, name, T, speed, grip in seq:
            if kind == "grip":
                s.gripper(grip, name)
            elif kind == "joint":
                self.joint_to(name, T, speed)
            else:
                self.line(name, T, speed)
        self.lift_to("up_from_hook", SAFE_H, "normal")

    def hook_T(self, tip, w, x, tilt):
        """Closed-finger hook pose: tool z tilted `tilt` from straight down toward the horizontal direction w."""
        z = unit(np.r_[w * np.sin(np.radians(tilt)), -np.cos(np.radians(tilt))])
        xx = np.r_[x, 0.0]; xx = unit(xx - (xx @ z) * z)
        T = np.eye(4); T[:3, :3] = np.stack([xx, np.cross(z, xx), z], 1); T[:3, 3] = tip - TIP_CLOSED * z
        return T

    def hook_seq(self, edge, o, az, h_top, tilt, P, arc):
        """edge: xy of the upper wall's free edge (open side); approach azimuth az deg from -o (body clear of the cabinet)."""
        ca, sa = np.cos(np.radians(az)), np.sin(np.radians(az))
        w = np.array([ca * -o[0] - sa * -o[1], sa * -o[0] + ca * -o[1]])
        x = np.array([-w[1], w[0]])
        tip_in = np.r_[edge - o * 0.02, self.tz + h_top - 0.028]
        tip_out = tip_in - np.r_[w * 0.07, 0.0]
        T_in = self.hook_T(tip_in, w, x, tilt); T_out = self.hook_T(tip_out, w, x, tilt)
        T_hi = T_out.copy(); T_hi[2, 3] = self.tz + SAFE_H
        seq = [("joint", "above_hook", T_hi, "normal", 1.0), ("line", "down_to_hook_pre", T_out, "normal", 1.0),
               ("line", "hook_in", T_in, "slow", 1.0)]
        ax = np.cross(np.r_[o, 0.0], [0.0, 0.0, 1.0])
        T = T_in
        for th in np.linspace(arc / 3, arc, 3):
            Rr = rot_axis(ax, th)
            T = T_in.copy(); T[:3, 3] = P + Rr @ (T_in[:3, 3] - P)
            seq.append(("line", f"hook_lift_{int(round(th))}", T, "slow", 1.0))
        Tb = T.copy(); Tb[:3, 3] -= 0.05 * T[:3, 2]
        seq.append(("line", "hook_out", Tb, "slow", 1.0))
        return seq

    def slide(self, face_pt, d, dist, why):
        """Slide the drawer by dist along d: closed fingertips 2 cm above the table pushing its face at face_pt."""
        s = self.s
        yaw = ang(np.array([-d[1], d[0]]))
        yaw = yaw - 180 if yaw > 90 else yaw + 180 if yaw < -90 else yaw
        # 00:48 operator (165/166: 65 N on down_slide): the gripper BODY (4.5 cm radius, from ~4 cm above the tips)
        # came down on the drawer's top: start 6 cm outside the pushed edge, descend and push with low force stops
        p0 = face_pt - d * 0.06
        T0 = tool_down_T(np.r_[p0, self.tz + 0.02 + TIP_CLOSED], yaw)
        T0h = T0.copy(); T0h[2, 3] = self.tz + SAFE_H
        T1 = T0.copy(); T1[:2, 3] += d * (dist + 0.05)
        s.decide(f"{why}: slide it {dist:.2f} m toward {ang(d):.0f} deg (low push at {np.round(face_pt, 3).tolist()})")
        s.gripper(1.0, "close the fingers to push")
        self.to_ready()
        try:
            self.joint_to("above_slide", T0h)
        except Abort as e:
            if "check failed" not in str(e):
                raise
            s.decide(f"slide refused by the model ({str(e)[:80]}): no slide")
            return False
        if not self.contact_line("down_slide", T0, "slow", limit=20.0):
            s.decide("slide: contact while coming down (on the drawer?): abort the slide")
            self.lift_to("up_from_slide", SAFE_H, "slow")
            return False
        if not self.contact_line("slide_drawer", T1, "slow", limit=25.0):
            s.decide("slide: the drawer resists (> 25 N): stop pushing")
            Tb = self.T_now(); Tb[:2, 3] -= d * 0.01
            self.line("off_slide", Tb, "slow")
        self.lift_to("up_from_slide", SAFE_H, "normal")
        return True

    def roll_seq(self, ctr, h_g, yw, P, ax, roll, jm=False):
        """Motion list for the pinch-and-roll of a tipped drawer (see right_side)."""
        Tg = tool_down_T(np.r_[ctr, self.tz + h_g + TIP_OPEN], yw)
        Th = Tg.copy(); Th[2, 3] = self.tz + SAFE_H
        Tpre = Tg.copy(); Tpre[2, 3] += 0.05
        T0 = Tg.copy(); T0[2, 3] += 0.012
        seq = [("joint", "above_roll", Th, "normal", 0.0), ("line", "down_to_roll_pre", Tpre, "normal", 0.0),
               ("line", "down_to_roll", Tg, "slow", 0.0), ("grip", "pinch across the drawer", None, None, 1.0),
               ("line", "lift_for_roll", T0, "slow", 1.0)]
        T = T0
        n = 5 if jm else 3          # jm: short joint moves (cartesian lines can hit a wrist joint jump)
        for th in np.linspace(roll / n, roll, n):
            Rr = rot_axis(ax, th)
            T = np.eye(4); T[:3, :3] = Rr @ T0[:3, :3]; T[:3, 3] = P + Rr @ (T0[:3, 3] - P)
            seq.append(("joint" if jm else "line", f"roll_{int(round(th))}", T, "slow", 1.0))
        seq.append(("grip", "let go: it falls bottom-down", None, None, TABLE_OPEN))
        Tb = T.copy(); Tb[:3, 3] -= 0.05 * T[:3, 2]
        seq.append(("line", "back_from_roll", Tb, "slow", TABLE_OPEN))
        Tu = Tb.copy(); Tu[2, 3] = self.tz + SAFE_H
        seq.append(("line", "up_from_roll", Tu, "normal", TABLE_OPEN))
        return seq

    def seq_ok(self, seq):
        """Model-check a motion list from READY without moving (planner + sim.check, as the session does)."""
        from rlpj.robot import sim
        from rlpj.robot.plan import NORMAL, SLOW
        pl, kin = self.s.planner, self.s.kin
        q = self.ik(tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0), [SEED_DOWN])
        gf = 0.0
        try:
            for kind, name, T, speed, grip in seq:
                if kind == "grip":
                    gf = grip
                    continue
                sp = NORMAL if speed == "normal" else SLOW
                if kind == "joint":
                    qg, _, _ = pl.ik_exact(T, q)
                    seg = pl.joint_move(name, q, qg, sp)
                else:
                    seg = pl.cart_line(name, q, T, sp)
                ok, rep = sim.check(seg.q, kin, sp, gripper_frac=gf, table_z=calib.table_z())
                if not ok:
                    bad = {k: v for k, v in rep.items() if (k.endswith("violations") and v) or k in ("joint_speed_ok",) and not v}
                    return False, f"{name}: {bad}"
                q = seg.q[-1].copy()
        except Exception as e:
            return False, f"{name}: {str(e)[:120]}"
        return True, ""

    # ------------------------------------------------------------ front-wall grasp and extraction
    def extract(self, p):
        s = self.s
        o = p["out"]
        wall_n = OFF - p["protr"] + 0.006
        h_top = float(np.clip(o["h_top"], 0.05, 0.11))
        tip_h = max(h_top - 0.012, 0.03)
        xy = xy_of(wall_n, o["t"])
        yaw = ang(N)
        inward = N.copy()
        ff = getattr(self, "front_fit", None)
        if ff is not None and abs(wrapd(ff["yaw"] - ang(N))) > 12:
            # skewed drawer: grasp the fitted front wall, jaws across it
            inward = ff["inward"]; xy = ff["mid"] + 0.006 * inward; yaw = ang(inward)
            s.decide(f"drawer skewed: front wall fitted at {np.round(ff['mid'], 3).tolist()}, yaw {ff['yaw']:.0f} deg (normal {ang(N):.0f})")
        Tg = tool_down_T(np.r_[xy, self.tz + tip_h + TIP_OPEN], yaw)
        T_hi = Tg.copy(); T_hi[2, 3] = self.tz + SAFE_H
        T_pre = Tg.copy(); T_pre[2, 3] += 0.05
        s.decide(f"front wall at n-OFF={wall_n - OFF:.3f} t={o['t']:.3f} top {h_top:.3f}: top-down grasp, jaws along the normal")
        s.gripper(0.45, "open to straddle the front wall")
        self.to_ready()                  # attempt 40: the direct joint move from home failed the keep-in check
        self.joint_to("above_front_wall", T_hi)
        self.line("down_to_front_pre", T_pre, "normal")
        time.sleep(0.4)
        img = s.wrist.latest()[0]
        self.save("wrist_front_pre", img)
        if img is not None and not s.dry:
            sc, _ = self.tab_score(img, calib.pose_to_T(s.feedback()["pose"]), xy_of(wall_n, o["t"]), -N, 0.035)
            s.summary(front_tab_score=None if sc is None else round(sc, 3))
            if sc is not None and self.out_judged is None:
                self.backwards = sc < 0.03
                s.decide(f"outer wall {'has NO pull tab: drawer is in backwards' if self.backwards else 'carries the pull tab'} (tab score {sc:.3f})")
        sh = self.refine_wall(img, xy, inward, h_top) if img is not None else None
        if sh is not None:
            xy = xy + sh * inward; Tg[:2, 3] = xy; T_pre[:2, 3] = xy
            self.line("front_wall_correct", T_pre, "slow")
        self.line("down_to_front_wall", Tg, "slow")
        g = s.gripper(1.0, "grasp the front wall")
        s.summary(front_grip=round(g, 1))
        if g >= HOLD_MAX and not s.dry:
            s.decide(f"front wall grasp reading {g:.1f}: nothing held")
            s.gripper(0.45, "open, missed the wall")
            self.line("up_from_front_wall", T_pre, "slow")
            return None
        T1 = self.shifted(np.r_[-0.20 * N, 0.004])
        self.line("slide_out", T1, "slow")
        self.lift_to("lift_drawer", SAFE_H, "normal")
        g2 = float(s.feedback()["gripper"])
        s.require(g2 < HOLD_MAX or s.dry, f"drawer lost while sliding out (gripper reading {g2:.1f})")
        # drawer bottom below the tool frame, from the in-slot front face bottom
        # drawer bottom below the tool: the side-depth front bottom is clipped at 2.5 cm and the drawer can tilt when
        # hanging from its front wall (attempt 18 touched the table early): assume the bottom at 1.0 cm in the slot
        drop = (Tg[2, 3] - self.tz) - 0.010
        pull = (inward if self.backwards else -inward).copy()   # the grasped wall is the pull wall unless in backwards
        return dict(yaw_pick=yaw, grasp_off=-inward * (L_DEFAULT / 2 - 0.007), pull=pull, drop=drop, release=0.025)

    # ------------------------------------------------------------ table pick
    def pick_table(self, c, pull=None):
        s = self.s
        ctr = np.array(c["obb_c"]); u = np.array([np.cos(np.radians(c["yaw"])), np.sin(np.radians(c["yaw"]))])
        L = c["L"]
        # attempts 75-77 (old side camera, partial outline): L forced to 0.155. The moved camera sees the whole drawer
        # (attempts 124/133/134: 0.12 x 0.105): forcing 0.155 put the grasp ~1.8 cm outside the end wall (2 misses in
        # 134). Trust the measurement unless it is clearly partial.
        if L < 0.10:
            L = 0.12
        ends = [ctr + u * (L / 2 - 0.007), ctr - u * (L / 2 - 0.007)]

        def cab_d(v):      # distance of a grasp point from the cabinet footprint (attempts 100-101: the end near the
            n_, t_ = nt(v)   # cabinet's side post made the gripper body enter the cabinet_front keep-out)
            return float(np.hypot(max(OFF - n_, 0.0), max(t_ - (CAB_T[1] + 0.005), (CAB_T[0] - 0.005) - t_, 0.0)))
        if min(cab_d(v) for v in ends) >= 0.08:
            ends.sort(key=lambda v: np.hypot(*v))
        else:
            ends.sort(key=lambda v: -cab_d(v))
        h_top = float(np.clip(c["h"], 0.03, 0.11))
        tip_h = max(h_top - 0.014, 0.018)
        s.gripper(TABLE_OPEN, "open wide (~72 mm) to straddle the end wall")
        for k, e in enumerate(ends):
            d0 = unit(ctr - e)
            yaw = ang(d0)
            yaw = yaw - 180 if yaw > 90 else yaw + 180 if yaw < -90 else yaw
            Tg = tool_down_T(np.r_[e, self.tz + tip_h + TIP_OPEN], yaw)
            T_hi = Tg.copy(); T_hi[2, 3] = self.tz + SAFE_H
            T_pre = Tg.copy(); T_pre[2, 3] += 0.05
            s.decide(f"table drawer obb {np.round(ctr, 3).tolist()} L {L:.3f} W {c['W']:.3f} yaw {c['yaw']:.0f}; grasp end wall at {np.round(e, 3).tolist()} (cabinet dist {cab_d(e):.3f}) top {h_top:.3f}")
            try:
                self.to_ready()
                self.joint_to("above_drawer_wall", T_hi)
                self.line("down_to_wall_pre", T_pre, "normal")
                time.sleep(0.4)
                img = s.wrist.latest()[0]
                self.save(f"wrist_wall_pre{k}", img)
                # 23:12 operator: refine shifts of 18-24 mm missed 4x in attempt 150; with the wrist footprint the
                # outline is already good: cap the refinement at 12 mm (attempt 151: -11 mm, held)
                # 23:43 operator: 10 refined picks missed with |shift| 0.033-0.058 (cabinet edge / far wall taken for
                # the near end wall): cap at 2 cm, and after a refined miss use the unrefined point once
                # 00:00 operator: closed loop - refine, move, re-image, until the residual is <= 6 mm (max 3)
                e_side = e.copy()
                self.refined_last = False
                if self.refine_off:
                    s.decide("previous refined grasp missed: this one uses the unrefined outline point")
                else:
                    for it_r in range(3):
                        sh = self.refine_wall(img, e, d0, h_top, max_shift=0.045)
                        if sh is None:
                            break
                        self.refined_last = True
                        if abs(sh) <= 0.006:
                            break
                        e = e + sh * d0
                        Tg[:2, 3] = e; T_pre[:2, 3] = e
                        self.line(f"wall_correct{it_r}", T_pre, "slow")
                        time.sleep(0.4)
                        img = s.wrist.latest()[0]
                        self.save(f"wrist_wall_pre{k}_{it_r + 1}", img)
                wall_log = dict(side=np.round(e_side, 4).tolist(), wrist=np.round(e, 4).tolist(),
                                along_d0=round(float((e - e_side) @ d0), 4))
                self.wall_log.append(wall_log)
                s.summary(wall_side_vs_wrist=self.wall_log)
                s.decide(f"end wall: side-camera point {wall_log['side']} -> wrist {wall_log['wrist']} ({wall_log['along_d0'] * 1000:+.0f} mm toward the centre)")
                if not self.contact_line("down_to_wall", Tg, "slow", limit=20.0):
                    Tb = self.T_now(); Tb[2, 3] += 0.03
                    self.line("up_from_contact", Tb, "slow")
                    self.refine_off = self.refined_last and not self.refine_off
                    self.lift_to("lift_empty", SAFE_H, "normal")
                    return None
                break
            except Abort as ex:
                if "check failed" not in str(ex) or k == len(ends) - 1:
                    raise
                s.decide(f"end {k} refused by the model ({str(ex)[:90]}): try the other end wall")
                if self.T_now()[2, 3] < self.tz + SAFE_H - 0.01:
                    self.lift_to("lift_other_end", SAFE_H, "slow")
        g = s.gripper(1.0, "grasp the end wall")
        s.summary(table_grip=round(g, 1))
        if g >= FIRM_MAX and not s.dry:
            self.refine_off = self.refined_last and not self.refine_off   # never refine-miss twice in a row
            # attempt 134: 96.5 = a marginal edge grasp; the drawer tilted in the hand and hit the table (58 N) on place
            s.decide(f"end wall grasp reading {g:.1f}: {'nothing held' if g >= HOLD_MAX else 'only a marginal grasp'}")
            s.gripper(TABLE_OPEN, "open, missed")
            self.line("up_from_wall", T_pre, "slow")
            self.lift_to("lift_empty", SAFE_H, "normal")
            return None
        self.line("lift_drawer_slow", T_pre, "slow")
        self.lift_to("lift_drawer", SAFE_H, "normal")
        return dict(yaw_pick=yaw, grasp_off=e - ctr, pull=pull, drop=tip_h + TIP_OPEN)

    def refine_wall(self, img, e, d_in, h_top, half=0.07, max_shift=0.06):
        """Wrist-camera refinement of a drawer wall seen from above: intensity profile along d_in (toward the drawer
        centre) across the expected wall top line at height h_top; the outer edge (bright table -> darker wood,
        going inward) is the strongest negative step. Returns the shift (m, along d_in) of the wall centre, or None."""
        s = self.s
        if img is None or s.dry:
            return None
        try:
            from rlpj.perception.wrist import project_base_to_wrist
            T = calib.pose_to_T(s.feedback()["pose"])
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
            if float(np.median(g)) < 40:     # attempt 78: at night the wrist image is nearly black (median ~1-16) and
                s.log(f"refine_wall: wrist image too dark (median {np.median(g):.0f}), no refinement")   # the gripper
                s.summary(wall_refine=dict(skipped="dark", median=round(float(np.median(g)), 1)))       # shadow
                return None                                                                               # fakes edges
            w = np.array([-d_in[1], d_in[0]])
            ss = np.arange(-half, half + 1e-9, 0.001)
            prof = []
            for sv in ss:
                P = np.array([np.r_[e + sv * d_in + a * w, self.tz + h_top] for a in np.linspace(-0.02, 0.02, 9)])
                uv = project_base_to_wrist(P, T)
                ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < g.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < g.shape[0] - 1)
                prof.append(np.median(g[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]) if ok.sum() >= 3 else np.nan)
            prof = np.array(prof)
            if np.isnan(prof).mean() > 0.3:
                s.log("refine_wall: profile out of the image"); return None
            prof = np.where(np.isnan(prof), np.nanmedian(prof), prof)
            # attempt 52: the outer step (table -> wall) is blurred by the wall's outer face seen obliquely; the drop into
            # the drawer's shadowed interior is sharp: inner edge = first sample where the profile stays dark
            table = float(np.median(prof[:8]))
            dark = prof < 0.6 * table
            k = None
            for i in range(len(prof) - 6):
                if dark[i:i + 6].all():
                    k = i; break
            grad = np.correlate(prof, [1, 1, 0, -1, -1], "same"); grad[:3] = grad[-3:] = 0
            if k is None:
                s.log("refine_wall: no dark interior along the profile"); return None
            inner = ss[k]
            shift = inner - 0.005 - 0.0          # e at the wall centre, ~5 mm outside the inner face
            step = float(table - prof[k:k + 6].mean())
            s.summary(wall_refine=dict(profile=np.round(prof[::4], 0).tolist(), table=round(table, 0), inner=round(inner, 4), step=round(step, 0), shift=round(shift, 4)))
            s.log(f"refine_wall: inner edge at {inner:+.3f} along d_in, step {step:.0f}, shift {shift:+.3f}")
            k = 0; grad[k] = step
            # 00:00 operator: positive shifts (toward the drawer centre) up to 4.5 cm were right when the step is
            # strong; the misses came from negative ones (away from the centre, onto the cabinet / table)
            pos_ok = 0.0 <= shift <= max(max_shift, 0.045) and step >= 70
            neg_ok = -0.015 <= shift < 0.0 and step >= 50
            if not (pos_ok or neg_ok):
                s.decide(f"wall refinement rejected (step {step:.0f}, shift {shift:+.3f})")
                return None
            s.decide(f"wall refined by wrist camera: shift {shift:+.3f} m toward the drawer centre")
            return float(shift)
        except Exception as ex:
            s.log(f"refine_wall failed: {ex}")
            return None

    # ------------------------------------------------------------ place
    def holding(self, when):
        g = float(self.s.feedback()["gripper"])
        if self.s.dry or g < HOLD_MAX:
            return True
        self.s.decide(f"drawer no longer in the gripper {when} (reading {g:.1f}): skip the placement, pick it from the table")
        self.s.gripper(0.4, "open")
        return False

    def cab_clear(self, centre, axis):
        """Smallest distance (m) from the drawer footprint (0.14 x 0.14 box at centre, axis = its front-back axis) to the
        cabinet's footprint (front plane - 1 cm .. +30 cm, t over CAB_T from the task yaml)."""
        w = np.array([-axis[1], axis[0]])
        pts = np.array([centre + i * 0.07 * axis + j * 0.07 * w for i in np.linspace(-1, 1, 9) for j in np.linspace(-1, 1, 9)])
        q = nt(pts)
        dn = np.maximum(OFF - 0.01 - q[:, 0], 0.0)
        dt = np.maximum(np.maximum(CAB_T[0] - q[:, 1], q[:, 1] - CAB_T[1]), 0.0)
        return float(np.min(np.hypot(dn, dt)))

    def place(self, held, target, delta=None):
        s = self.s
        target = np.asarray(target, float)
        if delta is None:
            # attempt 54: a drawer placed next to the cabinet corner merged with it in the orchestrator's measurement
            # ("not found"): pick a yaw that keeps the footprint >= 5 cm from the cabinet; else nudge the aim away
            box = target + self.bias
            # attempt 73: the judges disagree when the pull points 10 deg off -N but ~40 deg off the robot base
            # ('lower left' in the side image): aim the pull at the robot base, clipped to +-25 deg of -N, +-5 jitter
            # attempts 75-76: a pull aimed straight at the robot base faces image-left, edge-on to the side camera,
            # and no judge can see it (worse in the dim evening light); attempts 46/47 at ~-138 deg were seen and
            # judged 'lower left': aim the pull at -N + 20 deg (toward the camera), +-5 jitter
            want = 20.0 + float(self.rng.uniform(-5, 5))
            cands = sorted([float(d) for d in np.arange(-25, 26, 5)] + [want], key=lambda d: abs(d - want))
            ok = [d for d in cands if self.cab_clear(box, rot2(N, d)) >= 0.05]
            delta = ok[0] if ok else max(cands, key=lambda d: self.cab_clear(box, rot2(N, d)))
            clr = self.cab_clear(box, rot2(N, delta))
            if clr < 0.05:
                away = unit(-N) * min(0.025, 0.05 - clr)
                s.decide(f"placement {clr:.3f} m from the cabinet: aim {np.linalg.norm(away):.3f} m away from it")
                target = target + away
            # 16:35 operator: keep the drawer centre >= 17 cm in front of the cabinet face (separate depth blob)
            n_c = float(nt(target + self.bias + self.place_bias)[0])    # where it will really land (learned bias)
            if OFF - n_c < FACE_GAP_MIN:
                back = FACE_GAP_MIN - (OFF - n_c)
                s.decide(f"placement centre {OFF - n_c:.3f} m from the cabinet face: aim {back:.3f} m further out")
                target = target - N * back
            s.summary(cabinet_clearance=round(self.cab_clear(target + self.bias, rot2(N, delta)), 3),
                      face_gap=round(float(OFF - nt(target + self.bias + self.place_bias)[0]), 3))
        a = rot2(N, delta)            # drawer axis from its front (pull) wall toward its back: pull faces the robot
        if held.get("pull") is not None:
            turn = wrapd(ang(-a) - ang(held["pull"]))
        else:                         # pull side unknown: keep the grasped wall toward the robot
            turn = wrapd(ang(a) - ang(-held["grasp_off"]))
        yaw = wrapd(held["yaw_pick"] + turn)
        # aim so that the side camera's blob centre (what the orchestrator measures) lands on the target: the blob
        # centre sits self.bias away from the drawer's box centre (attempt 19: ~2-4 cm toward the camera)
        g = np.asarray(target) + self.bias + rot2(held["grasp_off"], turn)
        h_tool = held["drop"] + held.get("release", 0.008)
        Tp = tool_down_T(np.r_[g, self.tz + h_tool], yaw)
        T_hi = Tp.copy(); T_hi[2, 3] = self.tz + SAFE_H
        T_pre = Tp.copy(); T_pre[2, 3] += 0.05
        s.decide(f"place: pull {'known' if held.get('pull') is not None else 'UNKNOWN'}, turn {turn:.0f} deg, pull wall toward {ang(-a):.0f} deg (robot side, normal {ang(N):.0f} + {delta:.0f}), grasp point {np.round(g, 3).tolist()}, tool {h_tool:.3f} above table")
        if not self.holding("before the carry"):
            return delta
        self.joint_to("above_target", T_hi)
        if not self.holding("after the carry"):
            return delta
        self.line("down_to_place_pre", T_pre, "normal")
        if not self.contact_line("place_down", Tp, "slow", limit=25.0, tip_margin=0.0):
            s.decide("place_down: contact before the planned height: release here")
        s.gripper(0.4, "release the drawer")
        self.line("up_from_drawer", T_pre, "slow", tip_margin=0.0)
        self.lift_to("clear_up", SAFE_H, "normal")
        return delta

    # ------------------------------------------------------------ wrist survey of a drawer on the table (18:39)
    def slot_or_table(self, p, tag):
        """Attempts 155-156: a drawer still ~2.5 cm proud in the slot was taken for a table drawer against the cabinet
        (near-cabinet blob) and 5 end-wall picks closed on nothing. Ask the judge; a wrist look-down that finds no
        drawer-sized wood in front of the cabinet confirms it. Returns 'slot_near' or 'table'."""
        s = self.s
        path = self.side_full(f"{tag}_inslot")
        ans = None
        if path is not None:
            ans = self.ask(path, "The wooden drawer is still partly inside the cabinet's bottom slot (its back part is in "
                           "the slot and only its front sticks out); it is NOT standing free on the table.", f"{tag}_inslot",
                           goal="Decide whether the drawer is still in the cabinet slot or standing free on the table.")
        fp = self.wrist_footprint(p["cand"][0], f"{tag}_ns")
        in_slot = bool(ans) or (ans is None and fp is None)
        if ans and fp is not None and self.last_conf < 0.7:
            in_slot = False                        # a full footprint in front of the cabinet beats a weak 'yes'
        sm = p.get("slot_med")
        s.decide(f"near-cabinet drawer: judge in-slot={ans} (conf {self.last_conf:.2f}), wrist footprint "
                 f"{'found' if fp is not None else 'not found'} -> {'IN THE SLOT (tab path)' if in_slot else 'on the table'}")
        if in_slot:
            p["slot_med"] = float(min(sm if sm is not None else -float(p["protr"]), -0.01))
            p["cand"] = []
            return "slot_near"
        return "table"

    def side_biased(self, c):
        """00:38 operator: never use a raw side-camera table point: add the measured wrist-side bias."""
        if c.get("wrist_fp") or c.get("side_biased"):
            return c
        return dict(c, obb_c=(np.array(c["obb_c"], float) + WRIST_SIDE_BIAS).tolist(),
                    xy=(np.array(c["xy"], float) + WRIST_SIDE_BIAS).tolist(), side_biased=True)

    def wrist_footprint(self, c, tag):
        """Look straight down from above the drawer: wood pixels projected onto the rim plane -> footprint OBB.
        Attempt 163: centred on the side camera's estimate the drawer sat cut off at the image edge (side camera off by
        several cm near the cabinet): if the biggest wood blob touches the image border or is implausible, re-centre
        on it and look again (max 3 looks)."""
        s = self.s
        if s.dry:
            return None
        xy_look = np.array(c["xy"], float) + (0.0 if c.get("wrist_fp") else WRIST_SIDE_BIAS)
        h_top = float(np.clip(c["h"], 0.05, 0.085))
        for look in range(3):
            res = self.footprint_look(xy_look, h_top, f"{tag}_{look}")
            if res is None:
                return None
            ctr, L, W, yaw, n_cells, border = res
            out = dict(obb_c=np.round(ctr, 4).tolist(), L=round(L, 3), W=round(W, 3), yaw=round(yaw, 1), cells=n_cells, border=border)
            s.summary(**{f"wrist_footprint_{tag}_{look}": out})
            plausible = 0.09 <= L <= 0.17 and 0.05 <= W <= 0.15
            if plausible and not border:
                s.decide(f"wrist footprint {tag}: centre {out['obb_c']} L {L:.3f} W {W:.3f} yaw {yaw:.0f} (side camera {np.round(c['obb_c'], 3).tolist()})")
                return dict(c, obb_c=out["obb_c"], L=out["L"], W=out["W"], yaw=out["yaw"], xy=out["obb_c"], wrist_fp=True)
            if np.linalg.norm(ctr - xy_look) < 0.015 and not border:
                break
            s.decide(f"wrist footprint {tag} look {look}: {'cut by the image border' if border else 'implausible'} {out}: re-centre on it")
            xy_look = ctr
        s.decide(f"wrist footprint {tag}: no clean drawer outline: keep the side-camera outline")
        return None

    def footprint_look(self, xy0, h_top, tag):
        """One look-down at xy0: (centre, L, W, yaw, cells, touches_border) of the biggest wood blob, or None."""
        s = self.s
        from rlpj.perception.wrist import project_base_to_wrist
        T = tool_down_T(np.r_[xy0, self.tz + 0.42], 0.0)
        try:
            self.to_ready()
            self.joint_to(f"look_down_{tag}", T)
        except Abort as e:
            if "check failed" not in str(e):
                raise
            s.decide(f"look-down pose refused ({str(e)[:80]})"); return None
        time.sleep(0.6)
        img = s.wrist.latest()[0]
        if img is None:
            return None
        self.save(f"wrist_down_{tag}", img)
        Tm = calib.pose_to_T(s.feedback()["pose"])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 80) & (hsv[..., 2] > 25)
        g = np.arange(-0.18, 0.1801, 0.003)
        X, Y = np.meshgrid(xy0[0] + g, xy0[1] + g, indexing="ij")
        G = np.stack([X.ravel(), Y.ravel(), np.full(X.size, self.tz + h_top)], 1)
        uv = project_base_to_wrist(G, Tm)
        H_, W_ = img.shape[:2]
        ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H_)
        near_edge = np.zeros(X.size, bool)
        near_edge[ok] = (uv[ok, 0] < 12) | (uv[ok, 0] > W_ - 13) | (uv[ok, 1] < 12) | (uv[ok, 1] > H_ - 13)
        occ = np.zeros(X.size, np.uint8)
        occ[ok] = mask[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        q = nt(G[:, :2])
        occ[(q[:, 0] > OFF - 0.015) & (q[:, 1] > CAB_T[0] - 0.03) & (q[:, 1] < CAB_T[1] + 0.03)] = 0
        occ = cv2.morphologyEx(occ.reshape(X.shape), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
        if n < 2:
            s.decide("wrist footprint: no wood seen"); return None
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cells = np.argwhere(lab == k)
        xy = np.stack([X[cells[:, 0], cells[:, 1]], Y[cells[:, 0], cells[:, 1]]], 1)
        border = bool(near_edge.reshape(X.shape)[lab == k].any())
        ctr, (L, W), yaw = obb(xy)
        return np.asarray(ctr, float), float(L), float(W), float(yaw), int(len(xy)), border

    def wall_views(self, c, tag):
        """18:39 operator: find the pull wall directly: a close wrist view of each wall from outside (camera 20 cm out,
        16 cm up, looking at the wall's outer face), judged one by one. Returns the pull wall's outward normal or None."""
        s = self.s
        if s.dry:
            return None
        ctr, walls = self.walls_of(c)
        # robot-facing wall first, then the two sides, then the far one
        walls = sorted(walls, key=lambda w: -float(w[0] @ unit(-N)))
        Q = ("This photo is from a camera on the robot gripper, looking at one outer wall of a wooden drawer that stands "
             "upright on a table. The drawer wall facing the camera (the nearest wall, in the middle of the image) carries "
             "the drawer's metal pull: a small grey metal label frame with a short metal tab or knob below it.")
        best = None
        for j, (d, half) in enumerate(walls):
            target = np.r_[ctr + d * (half + 0.005), self.tz + 0.035]
            moved = False
            for out_, up_ in ((0.10, 0.26), (0.14, 0.22), (0.06, 0.30)):
                p_cam = np.r_[ctr + d * (half + out_), self.tz + up_]
                z = unit(target - p_cam)
                for sx in (1.0, -1.0):
                    x = sx * np.r_[-d[1], d[0], 0.0]; x = unit(x - (x @ z) * z)
                    T = np.eye(4); T[:3, :3] = np.stack([x, np.cross(z, x), z], 1); T[:3, 3] = p_cam
                    try:
                        self.to_ready()
                        q = self.ik(T, [self.q_now(), SEED_DOWN] + [np.array([0, 40, -120, 0, j5, j6], float) for j5 in (30.0, 60.0) for j6 in (-90.0, 90.0)])
                        s.move_joint(f"view_wall_{tag}_{j}", q, "normal")
                        moved = True
                        break
                    except Abort as e:
                        if "check failed" not in str(e) and "no IK" not in str(e):
                            raise
                if moved:
                    break
            if not moved:
                s.decide(f"wall view {j} ({ang(d):.0f} deg) not reachable"); continue
            time.sleep(0.6)
            img = s.wrist.latest()[0]
            path = self.save(f"wrist_wall_{tag}_{j}", img, w=960) if img is not None else None
            if path is None:
                continue
            # attempt 136: with the reset goal as context the judge rejected views that DID show the pull ("fingers in
            # view", "looks into the drawer"): a neutral survey goal
            ans = self.ask(path, Q, f"wall_{tag}_{j}", goal=SURVEY_GOAL)
            s.decide(f"wall view {j}: wall facing {ang(d):.0f} deg -> pull {'YES' if ans else 'no' if ans is False else '?'} (conf {self.last_conf:.2f})")
            if ans and (best is None or self.last_conf > best[1]):
                best = (d, self.last_conf)
                if self.last_conf >= 0.7:
                    break
        self.to_ready()
        if best is None:
            return None
        s.summary(**{f"pull_wall_{tag}": dict(dir_deg=round(ang(best[0]), 1), conf=best[1])})
        return best[0]

    def place_plain(self, held, xy, tag):
        """Set the held drawer down at xy without turning it (staging spot for the survey)."""
        s = self.s
        g = np.asarray(xy, float) + held["grasp_off"]
        h_tool = held["drop"] + held.get("release", 0.008)
        Tp = tool_down_T(np.r_[g, self.tz + h_tool], held["yaw_pick"])
        T_hi = Tp.copy(); T_hi[2, 3] = self.tz + SAFE_H
        T_pre = Tp.copy(); T_pre[2, 3] += 0.05
        s.decide(f"stage: set the drawer down at {np.round(xy, 3).tolist()} without turning it")
        if not self.holding("before staging"):
            return False
        self.joint_to(f"above_stage_{tag}", T_hi)
        self.line(f"down_to_stage_pre_{tag}", T_pre, "normal")
        self.line(f"stage_down_{tag}", Tp, "slow", tip_margin=0.0)
        s.gripper(0.4, "release the drawer")
        self.line(f"up_from_stage_{tag}", T_pre, "slow", tip_margin=0.0)
        self.lift_to(f"clear_stage_{tag}", SAFE_H, "normal")
        return True

    # ------------------------------------------------------------ main
    def run(self):
        s = self.s
        tgt = np.asarray(self.targets["drawer"], float)
        s.phase("perceive")
        if np.abs(wrap180(self.q_now() - HOME_Q)).max() > 3:
            g0 = float(s.feedback()["gripper"])
            if 75.0 < g0 < HOLD_MAX and self.T_now()[2, 3] > self.tz + 0.11:
                # attempt 118: the arm started holding the drawer (stopped mid-transfer in 117); moving home with it
                # dropped it onto its side: lower it toward the table first, then let go
                s.decide(f"start: gripper reads {g0:.1f} (holding the drawer?): lower it toward the table and let go first")
                T = self.T_now(); T[2, 3] = self.tz + 0.10
                try:
                    self.line("put_down_start", T, "slow")
                except Abort as e:
                    if "check failed" not in str(e) and "contact force" not in str(e):
                        raise
                    s.decide(f"put-down stopped ({str(e)[:80]}): open where it is")
                s.gripper(PRECLOSE, "let go of the drawer")
                g0 = float(s.feedback()["gripper"])
            if 3.0 < g0 < 97.0 and self.T_now()[2, 3] < self.tz + 0.15:
                s.decide(f"start: gripper reads {g0:.1f} low near the cabinet (maybe still on the pull tab): open fully first")
                s.gripper(0.0, "let go fully before backing off (00:25: pre-closed pads stayed around the knob)")
            self.escape()
            self.go_home("home_start")
        p = self.perceive("start")
        st = self.state_of(p)
        if st in ("missing", "slot_deep") and (p.get("face_depth") is None or p["face_depth"] > 0.03):
            # an empty slot reads ~0.035-0.045 deep: the drawer may be on the table out of the side camera's view
            if self.wrist_find("start"):
                self.go_home("home_after_find")
                p = self.perceive("after_find")
                st = self.state_of(p)
        if st == "table" and p["cand"] and p["cand"][0].get("near_cabinet") and not s.dry:
            st = self.slot_or_table(p, "start")
        s.summary(start_state=st, start_perception={k: v for k, v in p.items() if k != "cand"})
        s.decide(f"start state: {st}")
        s.require(st != "missing", "drawer not found (no drawer blob on the table and the bottom slot looks empty)")
        held = None
        self.to_ready()
        if st == "slot_near":
            sm = float(p["slot_med"])
            # 18:14 operator: after an RL episode the drawer is the right way round; 1-7 cm out -> the knob pinch-and-pull
            # (attempt 131: the 2-4 cm wall-edge clamp was a false 'backwards' path and missed twice)
            self.depth = sm
            s.decide(f"drawer face median {-sm:.3f} m out: tab pinch-and-pull (right way round by default)")
            st = "slot"
        if st == "slot_deep":
            self.depth = float(p["face_depth"])
            # the tab pinch fits the keep-out (3 cm into the slot opening) only up to ~1.5 cm past flush: the model's
            # forward finger points sit ~7 mm ahead of the fingertip (offline sweep: 2.5 cm needs the tab 5 mm beyond the tips)
            if self.depth > DEEP_MAX:
                s.phase("recessed")
                # attempt 143: the pinch at the measured knob was refused by the keep-out every time; the press-drag
                # at the measured knob t stays outside it: interleave the drags early
                V = DEEP_VARIANTS
                tries = [("pinch", V[0]), ("drag", 0), ("pinch", V[1]), ("drag", 1)] + [("pinch", v) for v in V[2:]]
                for kind, v in tries:
                    s.decide(f"drawer recessed {self.depth:.3f} m: try {kind} {v['name'] if kind == 'pinch' else v}")
                    try:
                        if kind == "pinch":
                            self.deep_cfg = v
                            self.pull_pin(0)
                            self.deep_cfg = None
                        else:
                            self.press_drag(v)
                    except Abort as e:
                        if "check failed" not in str(e):
                            raise
                        s.decide(f"{kind} refused by the model: {str(e)[:100]}")
                        self.deep_cfg = None
                        self.to_ready()
                    self.go_home("home_after_try")
                    p = self.perceive(f"after_{kind}")
                    st = self.state_of(p)
                    s.summary(**{f"recessed_after_{kind}_{v['name'] if kind == 'pinch' else v}": dict(state=st, depth=p.get("face_depth"), protr=p["protr"])})
                    if st != "slot_deep":
                        break
                    self.depth = float(p["face_depth"])
                    if self.depth <= DEEP_MAX:
                        break
                s.require(st != "slot_deep" or self.depth <= DEEP_MAX,
                          f"drawer still recessed {self.depth:.3f} m after all variants; a person must pull it out to about flush")
            if st == "slot_deep":
                s.decide(f"drawer pushed {self.depth:.3f} m past flush: pinch the tab that much deeper")
                st = "slot"
            elif st == "slot":
                self.depth = 0.0
            self.from_recess = st in ("slot_back", "out", "table")   # pulled out by its tab: the right way round
        if st == "slot_near":     # (after a recessed pull) median face < 4 cm out: the tab pinch-and-pull again
            self.depth = float(p["slot_med"])
            s.decide(f"drawer face median {-self.depth:.3f} m out after the recessed pull: tab pinch-and-pull")
            st = "slot"
        if st == "slot_back":     # 22:12 geometry: also after a recessed pull (the ~2 cm-proud clamp is out of joint limits)
            # 18:14 operator: 'no pull seen' from the moved side camera does not mean backwards: knob pinch first
            sm = p.get("slot_med")
            self.depth = float(sm) if sm is not None and sm < 0 else -float(p["protr"])
            s.decide(f"drawer sticking out ({-self.depth:.3f} m by the face median): tab pinch-and-pull (right way round by default)")
            st = "slot"
        if st == "slot_back":
            s.phase("pull_back")
            self.backwards = False
            self.pull_back(p)
            self.go_home("home_after_back")
            p = self.perceive("after_back")
            st = self.state_of(p)
            s.require(st in ("out", "table") or s.dry, f"backwards drawer not pulled out (state {st}, protrusion {p['protr']:.3f})")
            if s.dry and st != "out":
                p = dict(p, out=dict(t=T_C, h_top=0.075, h_bot=0.03, n=0), protr=PULL + 0.012); st = "out"
        if st == "slot":
            s.phase("pull_out")
            for k in range(3):
                ok = self.pull_pin(k)
                self.go_home(f"home_after_pull{k}")
                p = self.perceive(f"after_pull{k}")
                st = self.state_of(p)
                if st in ("slot_near", "slot_back") and ok:
                    # attempt 149: from ~2 cm proud only short pulls fit: pinch the tab again and pull further
                    sm = p.get("slot_med")
                    self.depth = float(sm) if sm is not None and sm < 0 else -float(p["protr"])
                    s.decide(f"pulled, still {-self.depth:.3f} m out by the face: pinch the tab again")
                    continue
                if st != "slot":
                    break
                s.decide(f"drawer still in the slot after pull {k} (protrusion {p['protr']:.3f})")
            if st in ("slot", "slot_back", "slot_near") and p["protr"] > 0.006 and not s.dry:
                s.decide(f"drawer still in the slot, sticking out {p['protr']:.3f}: clamp the protruding wall edge and pull")
                self.backwards = self.judge_backwards()  # 18:14: only if a close wrist view shows no metal pull
                self.pull_back(p)
                self.go_home("home_after_back")
                p = self.perceive("after_back")
                st = self.state_of(p)
            s.require(st in ("out", "table") or s.dry,
                      f"drawer not pulled out (protrusion {p['protr']:.3f}" + (f", it sat {self.depth:.3f} m past flush: the pull tab "
                      "may be out of reach inside the keep-out" if self.depth > 0 else "") + ")")
            if s.dry and st == "slot":
                p = dict(p, out=dict(t=self.pin_t, h_top=0.088, h_bot=0.03, n=0), protr=PULL - 0.002); st = "out"
        delta = None
        if st == "out":
            # which way round is it? ask the vision judge (arm at home, drawer sticking out toward the lower left)
            if np.abs(wrap180(self.q_now() - HOME_Q)).max() > 3:
                self.go_home("home_for_judge")            # attempt 72: the arm at READY hid the drawer from the camera
            path = self.side_full("out")
            if path is not None:
                ans = self.ask(path, "A wooden drawer sticks out of the cabinet's bottom slot toward the robot. The outer wall of "
                               "that drawer (the one facing the lower left of the image, toward the robot base) carries the "
                               "metal pull: a small grey metal label frame with a short tab below it.", "out_q")
                if ans is not None:
                    self.out_judged = bool(ans)
                    self.backwards = not ans
                    s.decide(f"vision judge: the drawer is in {'the right way round' if ans else 'BACKWARDS'}")
            s.phase("extract")
            for k in range(2):
                held = self.extract(p)
                if held:
                    break
                p = self.perceive(f"retry_front{k}")
                s.require(self.state_of(p) == "out", f"drawer state changed to {self.state_of(p)} after a missed front grasp")
            s.require(held is not None, "front wall not grasped after two tries")
            s.phase("place")
            delta = self.place(held, tgt - self.place_bias)
            self.just_placed = True
        # table: pick and place; then up to two corrections
        tries = 0
        for it in range(6):
            s.phase(f"check{it}")
            self.go_home("home_for_check")
            p = self.perceive(f"check{it}")
            st = self.state_of(p)
            if st != "table":
                s.require(s.dry and held is not None, f"drawer not seen on the table (state {st})")
                break
            c = p["cand"][0]
            # 00:43 operator: a drawer tipped onto its side (blob 0.106 tall, upright ~0.07): tip it back upright first
            k_r = 0
            while c["h"] > TIPPED_H and k_r < 3 and not s.dry:
                s.phase(f"right{it}_{k_r}")
                xy0 = np.array(c["xy"], float)
                self.right_shift = np.zeros(2)
                self.right_side(c, f"r{it}_{k_r}")
                k_r += 1
                self.go_home("home_after_right")
                p = self.perceive(f"after_right{it}_{k_r}")
                st = self.state_of(p)
                if st != "table" and self.wrist_find(f"right{it}_{k_r}"):
                    self.go_home("home_after_find")
                    p = self.perceive(f"after_find{it}_{k_r}")
                    st = self.state_of(p)
                s.require(st == "table", f"drawer not seen on the table after tipping it upright (state {st})")
                c = p["cand"][0]
                drift = float(np.linalg.norm(np.array(c["xy"], float) - xy0 - self.right_shift))
                s.summary(**{f"right_drift_{it}_{k_r}": round(drift, 3)})
                if c["h"] > TIPPED_H:
                    s.require(drift <= DRIFT_MAX, f"tipped drawer drifted {drift:.3f} m without righting (01:12 rule: stop "
                              "after 2 cm of drift, never push it toward a table edge): needs a person")
            s.require(c["h"] <= TIPPED_H or s.dry, f"drawer still on its side after {k_r} tip pushes (height {c['h']:.3f})")
            self.bias = np.array(c["obb_c"]) - np.array(c["xy"])
            err = float(np.linalg.norm(np.array(c["xy"]) - tgt))
            if self.just_placed and not s.dry:
                # 151/158: placements land ~4 cm toward -y every time: learn the systematic error and aim against it
                ev = np.array(c["xy"], float) - tgt
                self.place_bias = np.clip(self.place_bias + 0.6 * ev, -0.06, 0.06)
                try:
                    with open(PLACE_BIAS_FILE, "w") as f:
                        json.dump(dict(bias=np.round(self.place_bias, 4).tolist(), last_err=np.round(ev, 4).tolist(),
                                       when=time.strftime("%Y-%m-%d %H:%M")), f)
                except Exception:
                    pass
                s.decide(f"placement error {np.round(ev, 3).tolist()} -> placement bias now {np.round(self.place_bias, 3).tolist()}")
                self.just_placed = False
            ok_or, pull = self.vlm_pull(c, f"check{it}")
            if not ok_or and self.survey_placed and not s.dry:
                # placed with the turn from a close wrist view of the pull wall: the side judge may simply not see it
                s.decide("side judge does not confirm, but the last placement used the pull wall seen up close: trust it")
                ok_or = True
            facing = None if ok_or is None else (0.0 if ok_or else 180.0)
            s.summary(**{f"err_check{it}": round(err, 4), f"pull_vs_robot_deg{it}": facing})
            self.final_facing = facing
            if (err <= GOOD and ok_or) or (held is not None and it >= 5):
                break
            if tries >= 5:
                s.decide(f"drawer {err:.3f} m from the target after {tries} picks: stop correcting, verify")
                break
            if not ok_or and not s.dry:
                # 18:39 operator: find the pull wall directly with close wrist views, at a spot with room (>= 25 cm
                # from the cabinet face); turn by exactly the needed angle when carrying it to the target
                gap = OFF - float(nt(np.array(c["obb_c"]))[0])
                if gap < STAGE_GAP + 0.03 or np.linalg.norm(np.array(c["xy"]) - np.array(STAGE_XY)) > 0.12:
                    s.phase(f"stage{it}")
                    fp = self.wrist_footprint(c, f"s{it}")
                    held = self.pick_table(fp or self.side_biased(c), None)
                    tries += 1
                    if held is None:
                        continue
                    self.place_plain(held, STAGE_XY, f"{it}")
                    self.go_home("home_after_stage")
                    p = self.perceive(f"staged{it}")
                    st = self.state_of(p)
                    s.require(st == "table", f"drawer not seen on the table after staging (state {st})")
                    c = p["cand"][0]
                    self.bias = np.array(c["obb_c"]) - np.array(c["xy"])
                s.phase(f"survey{it}")
                c = self.wrist_footprint(c, f"v{it}") or c
                pw = self.wall_views(c, f"{it}")
                if pw is not None:
                    pull = pw
                    self.survey_pending = True
                    s.decide(f"pull wall found by the close views: facing {ang(pw):.0f} deg; turn it to the robot side at the target")
                else:
                    s.decide("close views did not find the pull: fall back to the side-judge guess")
            s.phase(f"pick{it}")
            if not s.dry:
                # attempt 150: 4 end-wall misses in a row from the side-camera outline; the wrist look-down footprint
                # (attempt 136) got the grasps right
                c = self.wrist_footprint(c, f"p{it}") or self.side_biased(c)
            try:
                held = self.pick_table(c, pull)
            except Abort as e:
                # attempt 31: a single bad joint sample (j5 read 0 instead of -47 deg) aborted a segment as a
                # 47 deg "tracking deviation"; retry the pick once from a lifted pose, anything else is a failure
                if "tracking deviation" not in str(e) or self.retried:
                    raise
                self.retried = True
                s.decide(f"segment stopped by a tracking deviation ({str(e)[:100]}); lift and retry the pick once")
                time.sleep(1.0)
                self.lift_to("lift_after_glitch", SAFE_H, "slow")
                held = None
            tries += 1
            if held is None:
                continue
            s.phase(f"place{it}")
            if np.linalg.norm(self.place_bias) > 0.002:
                s.decide(f"aim corrected by the learned placement bias {np.round(self.place_bias, 3).tolist()}")
            self.place(held, tgt - self.place_bias)
            self.just_placed = True
            self.survey_placed = self.survey_pending
            self.survey_pending = False
        # final measurement (arm at home)
        s.phase("verify")
        self.go_home("home_final")
        p = self.perceive("final")
        st = self.state_of(p)
        s.require(st == "table" or s.dry, f"final: drawer not seen on the table (state {st})")
        if st == "table":
            c = p["cand"][0]
            self.bias = np.array(c["obb_c"]) - np.array(c["xy"])
            err = float(np.linalg.norm(np.array(c["xy"]) - tgt))
            yaw_axis = c["yaw"]
            dev = wrapd(yaw_axis - ang(N)); dev = dev - 180 if dev > 90 else dev + 180 if dev < -90 else dev
            facing = self.final_facing
            s.summary(pull_vs_robot_deg=None if facing is None else round(facing, 0))
            ok = err <= TOL and c["h"] <= 0.10 and (facing is None or facing <= 40)
            s.summary(measured={"drawer": c["xy"]}, errors={"drawer": round(err, 4)}, target={"drawer": tgt.tolist()},
                      verify=dict(obb_center=c["obb_c"], L=c["L"], W=c["W"], height=c["h"], yaw_vs_normal_deg=round(dev, 1), tol=TOL))
            s.claim("success" if ok else "fail")
        else:
            s.summary(measured={"drawer": None}, note="dry run: drawer not moved")
            s.claim("fail")


def load_targets(arg):
    """--targets: inline JSON or a path to a JSON file (the orchestrator passes a file)."""
    arg = (arg or "").strip()
    if arg and not arg.startswith("{") and os.path.exists(arg):
        with open(arg) as f:
            return json.load(f)
    return json.loads(arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    a, _ = ap.parse_known_args()
    with RobotSession(program="reset") as s:
        try:
            targets = load_targets(a.targets)
            s.require(isinstance(targets, dict) and "drawer" in targets, f"targets without 'drawer': {targets}")
        except (ValueError, OSError) as e:
            s.summary(measured=None)
            raise Abort(f"cannot read --targets {a.targets!r}: {e}")
        s.summary(targets=targets, measured=None)
        Reset(s, targets).run()


if __name__ == "__main__":
    main()
