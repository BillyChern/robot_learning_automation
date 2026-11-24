"""drawer_lying, one episode (notes/design.md): scripted wrist look-downs, pinch of the pull, carry over the opening;
learned alignment and descent (bounded residuals on a straight-down prior); release, rise, verification with the
orchestrator's geometric check. Env: RLPJ_RL_DIR, RLPJ_RL_MODE (train|eval), RLPJ_EPISODE (-1 = test run)."""
import contextlib
import json
import os
import re
import sys
import time

os.environ.setdefault("RLPJ_TASK", "drawer_lying")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from rlpj import config
from rlpj.perception import tabletop as tt
from rlpj.robot import calib, sim
from rlpj.robot.kin import wrap180
from rlpj.robot.plan import SLOW, tool_down_T
from rlpj.robot.session import Abort, RobotSession
from rlpj.success import checks
import rlcommon as rc
import touch
import vision as V

TASK = config.load("drawer_lying")["task"]
CAB = TASK["fixtures"]["cabinet"]
RIM = float(CAB["rim_h"])
OPEN_SIZE = [float(v) for v in CAB["compartment"]]
OPEN_C = np.array([0.4519, 0.0029])            # wrist-measured (attempt 4); the yaml's centre/yaw are ~1.5 cm / 3 deg off
OPEN_YAW = 18.3
OPEN_AIM = np.array([0.0, -0.028])             # look-down aim shift: cy is 251 of 720, the opening's -y long side was at the image edge

SEED_DOWN = np.array([0, 25, -110, 0, -45, 90.0])
CARRY = dict(name="carry", jv=12.0, lin=0.06)
LOOK_H = 0.30                                  # tool height of the look-downs above the table
F_GRASP, F_INSERT = 20.0, 25.0                 # N: operator's low-force limits for the approach and the descent
FACE_H, FACE_TOL = 0.0912, 0.008                # m: drawer top on the table (median of the first-try entries), largest believable error
TIP_CLEAR = 0.004                              # fingertips above the drawer's top face when closed on the tab
TIP_CLOSED = 0.027 + 0.020 * 0.85              # tool to fingertip with the jaws on the ~12 mm tab
GRIP_PRE = 0.50
HOLD = (60.0, 97.0)                            # gripper reading when the tab is held
HOVER = 0.008                                  # drawer bottom above the rim when the learned part starts
LIFT_CLEAR = 0.030                             # drawer bottom above the rim while carrying
FLOOR = 0.005                                  # compartment floor above the table (operator 18:32, sim worker; the yaml's 0.013 is too high)
B_MIN = 0.005                                  # lowest commanded drawer bottom above the table
K_ALIGN, K_DESC = 2, 90
DZ0 = 0.007                                    # nominal descent per step (m); the action scales it by 0.5 .. 1.5
SC_ALIGN = np.array([0.003, 0.003, 1.5])       # m, m, deg per unit action, align steps
SC_DESC = np.array([0.001, 0.001, 0.5])        # descent steps
MAX_C = np.array([0.018, 0.018, 4.0])          # bound of the accumulated corrections
PEAK_SEAT = 12.0                               # N: a force rise this large near the floor means the drawer is seated
MAX_CONTACTS = 36
FIRST_STD = 0.5                                # extra exploration before the first contact, learned arm only (units of the action, 1 mm / 0.5 deg)
SIM_EPISODES = {31, 34, 37, 40, 43}             # operator 20:27: zero-shot sim policy (sim/drawer_insert, README "Lying cabinet")
F_SIM = 35.0                                   # N: force stop of the sim arm (operator: 25-40 N)
SIM_DRIFT_MAX = 0.014                          # m from the start while still on the rim
SIM_OVER_N, SIM_OVER_MAX, SIM_AWAY_MAX, SIM_STOPS_MAX = 40.0, 2, 3, 20   # end the sim episode: stops above 40 N, steps against the force, stops
SCRIPTED_REV = "r4 (r3 + start offset from first-descent entries only)"   # bump with every change of the scripted part; both arms use it
UNLOAD = 0.004                                 # m lifted after a touch (operator 18:30)
DROPPED = 0.015                                # m below the rim: this point is the opening, remember it
OBS_C = np.array([0.008, 0.008, 4.0])          # scale of the residual in the observation
FIRST_TILT = 0.0                               # deg about b; tilts -5 .. +2.5 at the rim all failed (attempt 26)
ENTERED = 0.004                                # m below the rim: the drawer is in the opening, no more lateral moves
FIX_MAX = np.array([0.010, 0.010, 4.0])         # bound of the measured correction (drawer vs opening, wrist image)
CAB_MOVE_MAX = 0.02                            # m, the task's limit
CAB_SHIFT_PX = 6.0                             # side image at 640 px, ~1.6 mm per px: ~1 cm (the task allows 2 cm)
LEVEL_H = 0.025                                # drawer bottom above the rim while it is levelled
LEVEL_TOL, LEVEL_MAX = 0.8, 6.0                # deg: tilt left alone, largest correction about the jaw axis
LEVEL_ITERS = 0                                # measure only: the side-depth tilt repeats to ~2 deg only (episode 2)
SEARCH_STD = 0.35                              # extra exploration on alignment steps (units of the action), train mode


def run_identity():
    """(mode, episode). The orchestrator's attempt.json names the run ("rl episode 10", "rl_eval episode 12") and wins over the
    environment: it adopted my test command with its RLPJ_EPISODE=-1 RLPJ_RL_MODE=eval prefix, which overrode its own values."""
    mode, ep = os.environ.get("RLPJ_RL_MODE", "train"), int(os.environ.get("RLPJ_EPISODE", "-1"))
    try:
        with open(os.path.join(os.environ["RLPJ_ATTEMPT_DIR"], "attempt.json")) as f:
            m = re.fullmatch(r"(rl|rl_eval) episode (\d+)", str(json.load(f).get("what", "")).strip())
        if m:
            mode, ep = ("eval" if m.group(1) == "rl_eval" else "train"), int(m.group(2))
    except (OSError, KeyError, ValueError):
        pass
    return mode, ep


def grasp_face(measured):
    """Height of the drawer's top face for the look-down plane and the pinch. Side depth scatters it by 3 mm, and the pinch height
    follows: at the +3 mm offset, <= 91.7 mm entered on the first descent (4 of 4), >= 92.7 mm did not (0 of 4). Far from the known
    height the measurement is kept."""
    return FACE_H if abs(measured - FACE_H) <= FACE_TOL else float(measured)


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def rot2(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Episode:
    def __init__(self, s):
        self.s = s
        self.mode, self.ep = run_identity()
        self.rng = np.random.default_rng(int(time.time() * 1000) % 2 ** 31)
        self.actor = rc.Actor()
        # operator 19:10: every third training episode is the scripted baseline (no learned residual, no exploration), same program
        self.arm = "learned_eval" if self.mode == "eval" else "sim" if self.ep in SIM_EPISODES else "scripted" if self.ep >= 0 and self.ep % 3 == 2 else "learned"
        if self.ep < 0 and os.environ.get("RLPJ_ARM") == "sim":   # own test runs only: an orchestrator episode always has ep >= 0
            self.arm = "sim"
        self.sim_log, self.sim_head = [], {}
        self.trans, self.events = [], []
        self.ret = 0.0
        self.terminal = None
        self._saved = False
        self.tilt = {}
        self.centre = touch.frozen_centre()  # fixed start offset (a, b, yaw) of scripted r4, the centre of the RL action
        self.contact_map = []                       # [da mm, db mm, bottom mm at the contact] around the measured alignment
        self.c0 = np.zeros(3)                       # in-hand correction (a, b, yaw), the start of the learned part
        self.g_ref = np.zeros(2)                    # side-depth slope of the standing drawer (base frame)
        self.phi, self.R_fix = 0.0, np.eye(3)       # levelling rotation about the opening's b axis

    # ---------------------------------------------------------------- helpers
    def q_now(self):
        # wrapped like the session's dry-run state: the robot reports 0..360 (attempt 8: joints 360 / 230 made the IK
        # seed choice differ from the dry run and picked a branch that left the keep-in box)
        return wrap180(np.asarray(self.s.feedback()["q"], float))

    def T_now(self):
        return calib.pose_to_T(self.s.feedback()["pose"])

    def force(self):
        f = self.s.feedback().get("force")
        return np.zeros(3) if f is None else np.asarray(f, float)

    def ik(self, T, extra=()):
        q_now, best = self.q_now(), None
        for sd in [q_now, SEED_DOWN, *extra]:
            try:
                q, _, _ = self.s.planner.ik_exact(T, sd)
            except Exception as e:
                self.s.log(f"IK from a seed failed: {str(e)[:80]}")
                continue
            d = float(np.max(np.abs(wrap180(q - q_now))))
            if best is None or d < best[1]:
                best = (q, d)
        self.s.require(best is not None, f"no IK for the tool at {np.round(T[:3, 3], 3).tolist()}")
        return best[0]

    @contextlib.contextmanager
    def low_force(self, f):
        ex = self.s.executor
        old = ex.force_limit
        try:
            if ex.force_cap:
                ex.force_limit = min(float(f), float(ex.force_cap))
            yield
        finally:
            if ex.force_cap:
                ex.force_limit = old if old is not None else ex.force_cap

    def save(self, name, img, w=640):
        if img is not None:
            cv2.imwrite(os.path.join(self.s.A, f"{name}.jpg"), cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1]))))

    def look_T(self, xyz, yaw=0.0):
        Ttc = V.wrist_cam()[2]
        T = tool_down_T(np.array([xyz[0], xyz[1], self.tz + LOOK_H]), yaw)
        Tc = T @ Ttc
        o, zc = Tc[:3, 3], Tc[:3, 2]
        T[:2, 3] += np.asarray(xyz[:2]) - (o + (xyz[2] - o[2]) / zc[2] * zc)[:2]
        return T

    def look(self, name, xyz):
        self.s.move_joint(f"look_{name}", self.ik(self.look_T(xyz)), "normal")
        time.sleep(0.7)
        img = self.s.wrist.latest()[0]
        self.save(f"wrist_{name}", img)
        return img, calib.pose_to_T(self.s.feedback()["pose"])

    # ---------------------------------------------------------------- perception
    def side_drawer(self, blobs):
        c0 = np.asarray(CAB["centre"], float)
        cand = [b for b in blobs if 0.05 < b["height"] < 0.14 and 0.06 < max(b["extent"]) < 0.17 and b["n"] > 500
                and np.linalg.norm(np.asarray(b["xy"]) - c0) > 0.10]
        if not cand:
            return None
        Q = np.asarray(max(cand, key=lambda b: b["n"])["points"], float)
        h = Q[:, 2] - self.tz
        H, e = np.histogram(h[h > 0.05], bins=np.arange(0.05, 0.125, 0.002))
        k = int(np.argmax(H))
        face = float(np.median(h[(h > e[k] - 0.002) & (h < e[k + 1] + 0.002)]))
        F = Q[np.abs(h - face) < 0.004, :2]
        (cx, cy), (w, l), ang = cv2.minAreaRect(np.ascontiguousarray(F.astype(np.float32)))
        if w > l:
            w, l, ang = l, w, ang + 90
        return dict(c=[float(cx), float(cy)], yaw=float(((ang + 90) + 90) % 180 - 90), size=[float(w), float(l)], face=face)

    def perceive(self):
        s = self.s
        s.phase("perceive")
        P = tt.cloud(s.side, n=7)
        self.tz = tt.table_fit(P)
        self.before = tt.detect(P, table_z=self.tz)
        self.img_before = s.side.latest()[0].copy()
        self.save("side_before", self.img_before)
        # the drawer alone: the full-scene blobs sometimes merge it with the cabinet (attempt 6: one blob 0.446 m long)
        sd = self.side_drawer(tt.detect(P, dict(x=(0.20, 0.375), y=(-0.13, 0.12)), table_z=self.tz))
        if sd is None and s.dry:                         # dry runs get no reset: the drawer may still be in the cabinet, so test the flow on a nominal one
            s.decide("dry run: no drawer on the table, using a nominal standing drawer")
            sd = dict(c=[0.285, -0.005], yaw=82.0, size=[0.0723, 0.1131], face=FACE_H)
        s.require(sd is not None, "no drawer-sized blob on the table")
        face = grasp_face(sd["face"])

        s.phase("look")
        img, Tm = self.look("opening", np.r_[OPEN_C + OPEN_AIM, self.tz + RIM])
        op = V.find_opening(img, Tm, OPEN_C, OPEN_YAW, OPEN_SIZE, self.tz + RIM)
        if op.get("ok") and np.linalg.norm(np.asarray(op["c"]) - OPEN_C) < 0.012 and abs(op["yaw"] - OPEN_YAW) < 2.5:
            c_o, yaw_o = np.asarray(op["c"], float), float(op["yaw"])
        else:
            s.decide(f"opening not measured ({ {k: op.get(k) for k in ('ok', 'why', 'c', 'yaw')} }): using the stored rim")
            c_o, yaw_o = OPEN_C.copy(), OPEN_YAW

        img, Tm = self.look("drawer", np.r_[sd["c"], self.tz + face])
        d, tab = self.pull_detect(img, Tm, sd["c"], self.tz + face)
        if s.dry and not (d.get("ok") and tab.get("ok")):          # the home view shows no drawer: flow test only
            d = dict(ok=True, c=sd["c"], yaw=sd["yaw"], size=[0.0723, 0.1131])
            a_s = V.axes(sd["yaw"] - 90.0)[0]
            tab = dict(ok=True, xy=(np.asarray(sd["c"]) - 0.0195 * a_s).tolist(), s=-0.0195, t=0.0, side=-1)
        s.summary(objects=dict(drawer=dict(c=d.get("c"), yaw=d.get("yaw"), size=d.get("size"), face_h=round(sd["face"], 4), face_used=round(face, 4)),
                               tab={k: tab.get(k) for k in ("ok", "xy", "s", "t", "side", "why")},
                               opening=dict(c=np.round(c_o, 4).tolist(), yaw=round(yaw_o, 2), measured=bool(op.get("ok")))))
        s.require(bool(d.get("ok")), f"drawer not found in the wrist view: {d}")
        s.require(bool(tab.get("ok")), f"pull tab not found in the wrist view: {tab}")
        try:                                         # reference for the held-drawer tilt: the same fit, standing on the table
            self.tilt = dict(standing=V.plane_tilt(P, d["c"], float(d["yaw"]) - 90.0, (0.030, 0.050), self.tz + face, 0.006,
                                                   exclude=(tab["s"], tab["t"], 0.016, 0.016)))
        except Exception as e:
            self.tilt = dict(standing=dict(ok=False, err=str(e)[:80]))
        st = self.tilt["standing"]
        if st.get("ok"):
            ds, dt = V.axes(float(d["yaw"]) - 90.0)
            self.g_ref = np.tan(np.radians(st["tilt_s_deg"])) * ds + np.tan(np.radians(st["tilt_t_deg"])) * dt
        return dict(face=face, d=d, tab=tab, c_o=c_o, yaw_o=yaw_o)

    def pull_detect(self, img, Tm, c, z):
        """Drawer and tab from the wrist look-down. A blurred frame can break the thin label frame in the mask (episode 8): try the
        frame, a smoothed copy of it, then fresh frames."""
        s = self.s

        def once(im):
            d = V.find_drawer(im, Tm, c, z)
            return d, (V.find_tab(im, Tm, d, z) if d.get("ok") else dict(ok=False, why="no drawer"))

        d, tab = once(img)
        tries = 1
        for k in range(0 if s.dry else 4):
            if d.get("ok") and tab.get("ok"):
                break
            H, W_ = img.shape[:2]
            d, tab = once(cv2.resize(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA), (W_, H), interpolation=cv2.INTER_CUBIC))
            tries += 1
            if d.get("ok") and tab.get("ok"):
                break
            time.sleep(0.5)
            img = s.wrist.latest()[0]
            self.save("wrist_drawer", img)
            d, tab = once(img)
            tries += 1
        s.summary(pull_detect_tries=tries)
        return d, tab

    # ---------------------------------------------------------------- scripted grasp and carry
    def grasp(self, w):
        s = self.s
        s.phase("grasp")
        xy = np.asarray(w["tab"]["xy"], float)
        # the jaws fit the tab either way round: take the one that ends with the camera on the near-wall (-a) side, where the
        # drawer and the cabinet's near end are both in view
        yaw = wrapd(w["yaw_o"] - 90.0 - self.drawer_turn(w))
        h_g = w["face"] + TIP_CLEAR + TIP_CLOSED
        Tg = tool_down_T(np.r_[xy, self.tz + h_g], yaw)
        T_pre = Tg.copy(); T_pre[2, 3] += 0.05
        T_hi = Tg.copy(); T_hi[2, 3] = self.tz + 0.22
        s.gripper(GRIP_PRE, "pre-open for the tab")
        s.move_joint("above_tab", self.ik(T_hi), "normal")
        s.move_line("down_pre", T_pre, "slow")
        with self.low_force(F_GRASP):
            s.move_line("down_tab", Tg, "slow")
        g = s.gripper(1.0, "pinch the tab")
        held = s.dry or HOLD[0] <= g <= HOLD[1]
        s.summary(grasp=dict(reading=round(float(g), 1), held=bool(held), h_g=round(h_g, 4)))
        if not held:
            s.gripper(GRIP_PRE, "open, tab missed")
            s.move_line("up_missed", T_pre, "slow")
            s.move_line("up_missed_high", T_hi, "normal")
            return None
        T_up = Tg.copy(); T_up[2, 3] += 0.03
        s.move_line("lift_slow", T_up, "slow")
        T_c = Tg.copy(); T_c[2, 3] = self.tz + h_g + RIM + LIFT_CLEAR
        s.move_line("lift", T_c, CARRY)
        g = float(s.feedback()["gripper"])
        if not s.dry and not HOLD[0] <= g <= HOLD[1]:
            s.decide(f"tab lost while lifting (reading {g:.1f})")
            return None
        return dict(h_g=h_g, xy_g=xy, yaw_g=yaw, T_g=Tg)

    @staticmethod
    def drawer_turn(w):
        """Rotation (deg) that puts the tab on the near-wall side of the opening with the drawer's long side along b."""
        tab_dir = float(w["d"]["yaw"]) - 90.0 + (0.0 if w["tab"]["side"] > 0 else 180.0)
        return wrapd(w["yaw_o"] + 180.0 - tab_dir)

    def plan_place(self, w, g):
        """Rotate the drawer so that its long side lies along b and the tab points to the near end wall (-a), the side
        away from the divider; tool pose = opening centre + the rotated pinch offset."""
        d, tab = w["d"], w["tab"]
        dyaw = self.drawer_turn(w)
        return dict(dyaw=dyaw, yaw_f=g["yaw_g"] + dyaw, off=np.asarray(tab["xy"], float) - np.asarray(d["c"], float))

    def pose(self, w, g, pl, c, b):
        """Tool pose for corrections c = (a, b, yaw) with the drawer's bottom b above the table."""
        ax, bx = V.axes(w["yaw_o"])
        xy = w["c_o"] + c[0] * ax + c[1] * bx + rot2(pl["off"], pl["dyaw"] + c[2])
        T = tool_down_T(np.r_[xy, self.tz + g["h_g"] + b], pl["yaw_f"] + c[2])
        pinch = T[:3, 3] + T[:3, :3] @ np.array([0, 0, TIP_CLOSED])  # levelling turns the tool about the pinch point
        T[:3, :3] = self.R_fix @ T[:3, :3]
        T[:3, 3] = pinch - T[:3, :3] @ np.array([0, 0, TIP_CLOSED])
        return T

    def carry(self, w, g, pl):
        s = self.s
        s.phase("carry")
        T = self.pose(w, g, pl, np.zeros(3), RIM + LIFT_CLEAR)
        q = self.q_now()
        seeds = [q + np.array([0, 0, 0, 0, 0, sgn * pl["dyaw"]]) for sgn in (1, -1)]
        s.move_joint("carry", self.ik(T, seeds), CARRY)
        s.move_line("to_level", self.pose(w, g, pl, np.zeros(3), RIM + LEVEL_H), "slow")
        self.level(w, g, pl)
        self.align_fix(w, g, pl)
        s.move_line("to_hover", self.pose(w, g, pl, self.c0, RIM + HOVER), "slow")

    def align_fix(self, w, g, pl):
        """Operator 17:58: fix the alignment before the descent. 2.5 cm above the rim, measure the drawer against the opening
        in one wrist image and correct, twice; the third look only records what is left."""
        s = self.s
        hist = []
        for it in range(3):
            try:
                time.sleep(0.4)
                img = s.wrist.latest()[0]
                self.save(f"wrist_align{it}", img)
                m = V.drawer_vs_opening(img, self.T_now(), w["c_o"], w["yaw_o"], self.tz + RIM, self.tz + RIM + LEVEL_H + w["face"])
            except Exception as e:
                m = dict(ok=False, err=str(e)[:80])
            hist.append(dict(ok=m.get("ok"), ea=m.get("ea"), eb=m.get("eb"), eyaw=m.get("eyaw"), c0=np.round(self.c0, 4).tolist(),
                             why=(m.get("drawer") or {}).get("why") or (m.get("cabinet") or {}).get("why") or m.get("err")))
            break                                    # logged only (operator 18:30): the touch search finds the opening
        self.c0 = self.centre.copy()
        s.summary(align=hist, touch_centre=np.round(self.centre, 4).tolist())
        s.log(f"drawer vs opening: {hist}")

    def measure_held(self, w, b):
        """Top-face plane of the held drawer from side depth (s along a, t along b); never stops the episode."""
        try:
            time.sleep(0.4)
            P = tt.cloud(self.s.side, n=5)
            ax, bx = V.axes(w["yaw_o"])
            xy = self.T_now()[:2, 3] - w["c_o"]
            return V.plane_tilt(P, w["c_o"], w["yaw_o"], (0.033, 0.053), self.tz + b + w["face"], 0.018,
                                exclude=(float(xy @ ax), float(xy @ bx), 0.016, 0.032))
        except Exception as e:
            return dict(ok=False, err=str(e)[:80])

    def level(self, w, g, pl):
        """The pinch on the tab's ends leaves the drawer free about the jaw axis (b), and its centre of mass is ~2 cm off
        the tab: it hangs with the far side low and its near edge catches the rim (episodes 0, 1). Turn the tool about
        the pinch point until the top face is level along a, relative to the same fit on the standing drawer."""
        s = self.s
        ax, bx = V.axes(w["yaw_o"])
        hist = []
        for it in range(LEVEL_ITERS + 1):
            m = self.measure_held(w, RIM + LEVEL_H)
            rel = None if not m.get("ok") else float(np.degrees(np.arctan(np.tan(np.radians(m["tilt_s_deg"])) - self.g_ref @ ax)))
            hist.append({k: m.get(k) for k in ("ok", "n", "tilt_s_deg", "tilt_t_deg", "z0")} | dict(rel_s=None if rel is None else round(rel, 2), phi=round(self.phi, 2)))
            if rel is None or abs(rel) < LEVEL_TOL or abs(rel) > 10.0 or it == LEVEL_ITERS:
                break
            self.phi = float(np.clip(self.phi + rel, -LEVEL_MAX, LEVEL_MAX))
            self.R_fix = cv2.Rodrigues(np.r_[bx, 0.0] * np.radians(self.phi))[0]
            s.move_line(f"level{it}", self.pose(w, g, pl, np.zeros(3), RIM + LEVEL_H), "slow")
        self.tilt["held"] = hist
        s.summary(tilt=self.tilt)
        s.log(f"levelling: {hist}")

    # ---------------------------------------------------------------- learned alignment and descent
    def try_step(self, name, T, limit=None):
        s = self.s
        try:
            seg = s.planner.cart_line(name, s.measured_q(s.last_ref()), T, SLOW)
        except Exception as e:
            return "refused", dict(reason=f"plan: {str(e)[:100]}")
        ok, rep = sim.check(seg.q, s.kin, seg.speed, gripper_frac=s.grip_frac, tip_margin=0.008,
                            table_z=max(s.table_z, calib.table_z()), max_radius=0.80)
        if not ok:
            return "refused", dict(reason="model", keep_out=rep.get("keep_out_violations"), tip=round(rep["min_tip_above_table"], 3))
        try:
            with self.low_force(F_INSERT if limit is None else limit):
                return "ok", s.run_segment(seg, tip_margin=0.008)
        except Abort as e:
            msg = str(e)
            return ("force" if "contact force" in msg else "abort"), dict(reason=msg[:160])

    def obs(self, phase, b, c, f, peak, k, w, n_contact):
        t = w["tab"]
        o = [phase, (RIM + HOVER - b) / 0.09, c[0] / OBS_C[0], c[1] / OBS_C[1], c[2] / OBS_C[2], *(f / 20.0), peak / 30.0,
             k / 20.0, t["s"] / 0.03, t["t"] / 0.03, n_contact / 3.0]
        return np.clip(np.array(o, float), -5, 5)

    def set_tilt(self, w, phi):
        """Tilt of the tool about the opening's b axis through the pinch point (used by pose())."""
        self.phi = float(phi)
        self.R_fix = cv2.Rodrigues(np.r_[V.axes(w["yaw_o"])[1], 0.0] * np.radians(self.phi))[0]

    def insert(self, w, g, pl):
        s = self.s
        s.phase("insert_learned")
        self.set_tilt(w, FIRST_TILT)
        c, b = self.c0.copy(), RIM + HOVER
        ax, bx = V.axes(w["yaw_o"])
        A3 = np.array([[*ax, 0.0], [*bx, 0.0], [0, 0, 1.0]])
        f0, peak, n_contact, end = self.force(), 0.0, 0, "steps"
        last_force = False
        ts, next_pt = touch.TouchSearch(MAX_CONTACTS), None
        c_entry, n_entry = None, 0                       # offset at which the drawer passed into the opening, rim touches before it
        for k in range(K_ALIGN + K_DESC):
            phase = 0 if k < K_ALIGN or last_force else 1   # after a contact: one lateral alignment step at the unloaded height
            o = self.obs(phase, b, c - self.c0, A3 @ (self.force() - f0), peak, k, w, n_contact)  # residual on the measured alignment
            if self.arm == "scripted":
                a, info = np.array(rc.prior(o), float), dict(mu=[0.0] * rc.ACT_DIM, std=[0.0] * rc.ACT_DIM)
            else:
                a, info = self.actor.act(o, explore=self.arm == "learned", rng=self.rng)
            if self.arm == "learned" and phase == 0 and n_contact > 0:   # wide exploration only at the search points
                a = np.r_[np.clip(a[:3] + self.rng.normal(0.0, SEARCH_STD, 3), -1.0, 1.0), a[3]]
            if self.arm == "learned" and n_contact == 0 and b > RIM:     # first descent: explore around the start offset (operator 20:21)
                a = np.r_[np.clip(a[:3] + self.rng.normal(0.0, FIRST_STD, 3), -1.0, 1.0), a[3]]
            if b <= RIM - ENTERED:                       # inside: the walls guide it, lateral residuals would only rub on them
                a = np.r_[0.0, 0.0, 0.0, a[3]]
            if phase == 0 and n_contact > 0:             # search: a new point around the measured alignment + the policy's residual
                c2 = self.c0 + next_pt + SC_DESC * a[:3]     # the search point + the policy's residual (zero in the scripted arm)
            else:
                c2 = c + SC_DESC * a[:3]                 # first try: stay on the measured alignment (clearance ~1 mm)
            c2 = np.clip(c2, self.c0 - MAX_C, self.c0 + MAX_C)
            b2 = b if phase == 0 else max(b - DZ0 * (1.0 + 0.5 * a[3]), B_MIN)
            status, res = self.try_step(f"ins{k:02d}", self.pose(w, g, pl, c2, b2))
            r = -0.02
            if status == "ok":
                peak = float(res.get("max_force_n", 0.0))
                r += 30.0 * (b - b2) - 0.05 * max(0.0, peak - 15.0) / 5.0
                c, b = c2, b2
            elif status == "force":
                r -= 0.15                               # contacts are part of the search now
                peak = F_INSERT
                n_contact += 1
                f_c = A3 @ (self.force() - f0)          # sideways force while resting on the rim (operator 18:32: does it point to the opening?)
                b_now = float(self.T_now()[2, 3]) - self.tz - g["h_g"]
                ts.report(c2 - self.c0, b_now)
                next_pt = ts.next()
                # no descent reward on a force stop: at the rim the tool sinks while the drawer stays (attempt 10: 8 mm over
                # three stops at 25 N with the drawer on the rim), and paying it would teach pressing on the rim
                b = min(b, b_now)
                self.contact_map.append([round(c2[0] * 1000, 1), round(c2[1] * 1000, 1), round(c2[2], 1), round(b_now * 1000, 1), round(float(f_c[0]), 1), round(float(f_c[1]), 1)])
                if b <= FLOOR + 0.012:
                    end = "seated"
                elif b_now < RIM - ENTERED:             # partly in: a lateral search step would push on the walls
                    end = "partial"
                elif next_pt is None:
                    end = "jam"
                else:                                   # unload: 3 mm up, the policy sees the contact and may correct
                    st2, _ = self.try_step(f"unload{k:02d}", self.pose(w, g, pl, c, b + UNLOAD))
                    if st2 == "ok":
                        b += UNLOAD
                    else:
                        end = "stuck"
            else:
                r -= 1.0
                end = status
            last_force = status == "force"
            if c_entry is None and b <= RIM - ENTERED:
                c_entry, n_entry = c.copy(), n_contact
            self.trans.append(dict(obs=o, act=a, rew=r))
            self.ret += r
            self.events.append([k, status, np.round(a, 2).tolist(), round(r, 3), round(b, 4), np.round(c, 4).tolist(), round(peak, 1), self.phi])
            s.log(f"step {k}: a {np.round(a, 2).tolist()} (mu {info['mu']}) -> {status} {res.get('reason', '')} bottom {b:.4f} "
                  f"c {np.round(c, 4).tolist()} tilt {self.phi:+.1f} peak {peak:.1f} N r {r:+.3f}")
            if end != "steps":
                break
            if not s.dry and not HOLD[0] <= float(s.feedback()["gripper"]) <= HOLD[1]:
                end = "dropped"
                break
            if status == "ok" and phase == 1 and (b <= B_MIN + 1e-4 or (peak >= PEAK_SEAT and b <= FLOOR + 0.012)):
                end = "seated"
                break
        if end == "seated" and c_entry is not None and b <= RIM - DROPPED and not s.dry:
            msg = f"drawer entered at a {c_entry[0] * 1000:.1f} mm, b {c_entry[1] * 1000:.1f} mm, yaw {c_entry[2]:.2f} deg after {n_entry} rim touches"
            msg += "; start offset frozen (scripted r4)"
            s.decide(msg)
        s.summary(entry=None if c_entry is None else dict(c=np.round(c_entry, 4).tolist(), rim_touches=n_entry))
        s.summary(touch=dict(best=np.round(ts.best, 4).tolist(), best_h=None if not np.isfinite(ts.best_h) else round(ts.best_h, 4), touches=len(ts.seen)))
        s.summary(insert=dict(end=end, steps=len(self.trans), bottom=round(b, 4), c=np.round(c, 4).tolist(), contacts=n_contact, peak=round(peak, 1), tilt=self.phi, contact_map=self.contact_map,
                                 tilts_tried=sorted({e[7] for e in self.events if e[1] == "force"})))
        self.save("side_inserted", s.side.latest()[0])
        return dict(end=end, b=b, c=c)

    # ---------------------------------------------------------------- sim arm (operator 20:27): the sim policy unchanged
    def insert_sim(self, w, g, pl):
        """Descent of sim/drawer_insert (README "Lying cabinet") from the same carry and hover as the other arms. The jaws lie along the
        opening's y, so yaw_straight is the tool yaw at the hover and the policy's tool frame is the opening frame there (the sim draws the
        soft jaw axis along tool x or y with equal odds). Believed centre = measured opening + the frozen r4 offset, so the drawer starts on it."""
        s = self.s
        s.phase("insert_sim")
        root = str(config.ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from sim.drawer_insert.lying_policy import Descent, Policy
        ax, bx = V.axes(w["yaw_o"])
        centre = w["c_o"] + self.c0[0] * ax + self.c0[1] * bx
        rim_z, ys = self.tz + RIM, wrapd(w["yaw_o"] - 90.0)
        d_tool = -rot2(pl["off"], pl["dyaw"] + self.c0[2])    # drawer top centre minus tool point, base = opening frame after rotating
        est = dict(off=(float(d_tool @ ax), float(d_tool @ bx)), yaw=0.0, tilt=(0.0, 0.0), hold=float(g["h_g"] - w["face"]))
        yaw_of = lambda T: float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        time.sleep(0.3)
        fs = []
        for _ in range(5):
            fs.append(self.force())
            time.sleep(0.03)
        f0 = np.mean(fs, axis=0)
        T = self.T_now()
        ins = Descent(Policy(str(config.ROOT / "sim/drawer_insert/weights/policy_lying.npz")), centre, w["yaw_o"], rim_z, ys, est, T[:3, 3], yaw_of(T), f0)
        z_min = B_MIN + w["face"] - RIM                       # never command the drawer's bottom below the lowest height of the other arms
        self.sim_head = dict(C=centre, AX=w["yaw_o"], RIM_Z=rim_z, YS=ys, est=est, est_source="wrist look-down of the standing drawer before the grasp; tilt not measured",
                             f0=f0, start_tool=T[:3, 3], start_yaw=yaw_of(T), force_stop_n=F_SIM)
        b = float(T[2, 3]) - self.tz - g["h_g"]
        end, done, over, away, seated, stops = "steps", False, 0, 0, 0, 0
        while not done and end == "steps":
            f_raw = self.force()
            xyz, yaw = ins.next(f_raw)
            if ins.target[3] < z_min:
                ins.target[3] = z_min
                xyz, yaw = ins.to_base(ins.target)
            k, t0 = ins.k, time.time()
            status, res = self.try_step(f"sim{k:02d}", tool_down_T(xyz, yaw), limit=F_SIM)
            stopped = status == "force"
            peak = float(res.get("max_force_n", 0.0)) + float(np.linalg.norm(f_raw - f0))     # README fallback for max |f - f0|
            stop_n = None
            if stopped:
                m = re.search(r"contact force ([0-9.]+) N", str(res.get("reason", "")))
                stop_n = float(m.group(1)) if m else F_SIM
                peak = max(peak, stop_n)
                stops += 1
                over += stop_n > SIM_OVER_N
            T = self.T_now()
            f_o = np.asarray(ins.obs["f"], float)
            step_xy = np.asarray(ins.step[:2], float)
            if (stopped or f_o[2] > 15.0) and np.linalg.norm(f_o[:2]) > 4.0 and np.linalg.norm(step_xy) > 0.001 and step_xy @ f_o[:2] < 0:
                away += 1                                     # in contact and stepping against the sideways force
            b2 = float(T[2, 3]) - self.tz - g["h_g"]
            r = -0.02 + (0.0 if stopped else 30.0 * (b - b2)) - 0.05 * max(0.0, peak - 15.0) / 5.0 - (0.15 if stopped else 0.0) - (1.0 if status not in ("ok", "force") else 0.0)
            self.ret += r
            b = b2 if not stopped else min(b, b2)
            if status in ("ok", "force"):
                done = ins.done(peak, stopped, T[:3, 3], yaw_of(T))
            self.sim_log.append(dict(k=k, obs={q: (np.round(v, 5).tolist() if np.ndim(v) else round(float(v), 5)) for q, v in ins.obs.items()},
                                     step=np.round(ins.step, 5).tolist(), target=np.round(ins.target, 5).tolist(), tool=np.round(T[:3, 3], 5).tolist(), tool_yaw=round(yaw_of(T), 3),
                                     f_raw=np.round(f_raw, 3).tolist(), peak=round(peak, 2), stopped=bool(stopped), status=status, stop_n=stop_n, dt=round(time.time() - t0, 3), bottom=round(b, 4), r=round(r, 3)))
            s.log(f"sim {k}: step {np.round(ins.step * [1000, 1000, 1, 1000], 2).tolist()} (mm, mm, deg, mm) -> {status} bottom {b:.4f} f {np.round(f_o, 1).tolist()} peak {peak:.1f} N")
            if stopped and b > RIM - ENTERED and np.linalg.norm(ins.cmd[:2] - ins.cmd0[:2]) >= SIM_DRIFT_MAX:
                away += 1                                     # on the rim at the end of the policy's 15 mm range. Not at once: the real test went
                                                              # through +15 mm on its way in (attempt 67), so only repeated stops there mean it is lost
            z_dec = float(ins.obs["z"])                   # README: on the floor when z < 1 cm and f_z > 10 N for two decisions
            seated = seated + 1 if z_dec < 0.01 and f_o[2] > 10.0 else (0 if z_dec > 0.012 else seated)
            if status not in ("ok", "force"):
                end = status
            elif seated >= 2 or (stopped and b <= FLOOR + 0.012):   # the second is the seated rule of the other arms
                end = "seated"
            elif over >= SIM_OVER_MAX:
                end = "over_force"
            elif away >= SIM_AWAY_MAX:
                end = "pushing_away"
            elif stops >= SIM_STOPS_MAX:
                end = "too_many_stops"
            elif not s.dry and not HOLD[0] <= float(s.feedback()["gripper"]) <= HOLD[1]:
                end = "dropped"
        cy = wrapd(yaw_of(T) - pl["yaw_f"])
        xy = T[:2, 3] - w["c_o"] - rot2(pl["off"], pl["dyaw"] + cy)
        c = np.array([xy @ ax, xy @ bx, cy])
        self.sim_head.update(end=end, decisions=len(self.sim_log), stops=stops, over_40=over, steps_against_force=away, final_bottom=round(b, 4))
        s.summary(sim_arm={k_: v for k_, v in self.sim_head.items() if k_ in ("end", "decisions", "stops", "over_40", "steps_against_force", "final_bottom", "est", "force_stop_n")},
                  insert=dict(end=end, steps=len(self.sim_log), bottom=round(b, 4), c=np.round(c, 4).tolist(), contacts=stops, peak=round(self.sim_log[-1]["peak"], 1)))
        self.save("side_inserted", s.side.latest()[0])
        return dict(end=end, b=b, c=c)

    # ---------------------------------------------------------------- end of the episode
    def release(self, w, g, pl, ins):
        s = self.s
        s.phase("release")
        if ins["end"] != "dropped" and ins["b"] > RIM - 0.02:       # not in the opening: put it back on the table
            s.decide(f"drawer not in the opening (bottom {ins['b']:.3f}): put it back on the table")
            s.move_line("up_to_carry", self.pose(w, g, pl, ins["c"], RIM + LIFT_CLEAR), "slow")
            T = tool_down_T(np.r_[g["xy_g"], self.tz + g["h_g"] + RIM + LIFT_CLEAR], g["yaw_g"])
            q = self.q_now()
            s.move_joint("carry_back", self.ik(T, [q + np.array([0, 0, 0, 0, 0, sgn * pl["dyaw"]]) for sgn in (1, -1)]), CARRY)
            T_d = g["T_g"].copy(); T_d[2, 3] += 0.003
            T_p = T_d.copy(); T_p[2, 3] += 0.04
            s.move_line("down_back_pre", T_p, "normal")
            try:
                with self.low_force(F_GRASP):
                    s.move_line("down_back", T_d, "slow")
            except Abort as e:
                s.log(f"put-back descent stopped: {str(e)[:100]}")
        s.gripper(GRIP_PRE, "release")
        T = self.T_now(); T[2, 3] += 0.06
        s.move_line("rise_slow", T, "slow")
        T[2, 3] = max(T[2, 3], self.tz + 0.25)
        s.move_line("rise", T, "normal")

    def cabinet_outline_move(self, after):
        """Distance (m) between the cabinet's outline centres before and after (operator 18:56), or None when a scene has no clean
        cabinet blob (for example merged with the standing drawer)."""
        f = getattr(checks, "outline_xy", V.outline_xy)
        yaw, c0 = float(CAB["yaw_deg"]), np.asarray(CAB["centre"], float)
        def pick(blobs):
            cand = [b for b in blobs if b["n"] > 3000 and 0.08 <= b["height"] <= 0.12 and 0.20 <= max(b["extent"]) <= 0.34
                    and np.linalg.norm(np.asarray(f(b, yaw))[:2] - c0) < 0.06]
            return max(cand, key=lambda b: b["n"]) if cand else None
        try:
            b0, b1 = pick(self.before), pick(after)
            return None if b0 is None or b1 is None else float(np.linalg.norm(np.asarray(f(b1, yaw))[:2] - np.asarray(f(b0, yaw))[:2]))
        except Exception as e:
            self.s.log(f"outline check failed: {str(e)[:100]}")
            return None

    def finish(self):
        s = self.s
        s.phase("verify")
        s.home("home_verify")
        time.sleep(0.5)
        P = tt.cloud(s.side, n=7)
        after = tt.detect(P, table_z=tt.table_fit(P))
        self.save("side_after", s.side.latest()[0])
        # the check's cabinet comparison uses the blob centre, which a filled compartment moves by 3.3 cm (attempts 30, 32, nothing
        # moved): flushness from the same check without it, cabinet motion from the side image
        with_blob, _ = checks.drawer_lying(self.before, after, TASK)
        ok, rep = checks.drawer_lying([], after, TASK)
        shift = V.cabinet_shift_px(self.img_before, s.side.latest()[0])
        outline = self.cabinet_outline_move(after)
        # the outline's far edge is noisy (extent 0.278-0.318 m on a static scene), the image shift repeats to ~0.3 px: moved = both agree,
        # or the one that is available says so
        if shift is None:
            moved = outline is not None and outline > CAB_MOVE_MAX
        else:
            moved = shift > CAB_SHIFT_PX and (outline is None or outline > CAB_MOVE_MAX)
        if ok is True and moved:
            ok, rep = False, dict(rep, reason=f"the cabinet moved (side image {shift} px, outline {outline} m)")
        brief = lambda bl: [[*np.round(b["xy"], 3).tolist(), round(float(b["height"]), 3), *np.round(b["extent"], 3).tolist(), int(b["n"])]
                            for b in bl if b["n"] > 300]          # x, y, height, extent x, extent y, points
        s.summary(blobs_before=brief(self.before), blobs_after=brief(after))
        bonus = 10.0 if ok is True else 0.0
        R = bonus                                        # the report's reward (operator 20:21): rc.fixed_rewards for the whole episode
        self.terminal = dict(bonus=bonus, R=round(R, 3))
        if self.trans:
            rew = rc.fixed_rewards(len(self.trans), [i for i, e in enumerate(self.events) if e[1] == "force"], self.events[-1][6], ok is True)
            for tr, x in zip(self.trans, rew):
                tr["rew"] = float(x)
            self.ret = float(rew.sum())
        elif self.arm == "sim" and self.sim_log:
            rew = rc.fixed_rewards(len(self.sim_log), [i for i, e in enumerate(self.sim_log) if e["stopped"]], self.sim_log[-1]["peak"], ok is True)
            for e, x in zip(self.sim_log, rew):
                e["r"] = round(float(x), 3)
            self.ret = float(rew.sum())
        s.summary(verify=dict(check=ok, reason=rep.get("reason"), n=rep.get("n"), median_h=rep.get("median_h"), flush=rep.get("flush"),
                              above=rep.get("above"), deep=rep.get("deep"), cabinet_moved_m=rep.get("cabinet_moved_m"),
                              on_table=bool(rep.get("drawer_on_table")), terminal_reward=round(R, 3),
                              cabinet_shift_px=None if shift is None else round(shift, 2), cabinet_outline_move_m=None if outline is None else round(outline, 4),
                              orchestrator_check=with_blob))
        self.save_transitions(ok, rep)
        if ok is not True:
            s.summary(first_failed_check=f"geometric check {ok}: {rep.get('reason')}")
        s.claim("success" if ok is True else "fail")

    def save_transitions(self, ok=None, rep=None):
        if self._saved:
            return
        self._saved = True
        if self.arm == "sim":                            # another action layout: kept out of rl/transitions, the learner never sees it
            d = os.path.join(rc.rl_dir(), "sim_arm")
            os.makedirs(d, exist_ok=True)
            name = ("dry_" if self.s.dry else "") + (f"ep{self.ep:04d}" if self.ep >= 0 else f"test_{time.strftime('%Y%m%d_%H%M%S')}") + ".json"
            with open(os.path.join(d, name), "w") as f:
                json.dump(dict(self.sim_head, arm="sim", episode=self.ep, ret=self.ret, check=ok, terminal=self.terminal, reason=(rep or {}).get("reason"),
                               scripted_rev=SCRIPTED_REV, attempt=self.s.A, decisions_log=self.sim_log), f,
                          default=lambda o: np.round(o, 5).tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating) else str(o))
            self.s.summary(transitions_file=os.path.join("sim_arm", name), transitions=len(self.sim_log), **{"return": round(self.ret, 3)})
            return
        for i, tr in enumerate(self.trans):
            last = i == len(self.trans) - 1
            tr["next_obs"] = tr["obs"] if last else self.trans[i + 1]["obs"]
            tr["done"] = 1.0 if last else 0.0
        name = f"{self.mode}_ep{self.ep:04d}" if self.ep >= 0 else f"{self.mode}_test_{time.strftime('%Y%m%d_%H%M%S')}"
        if self.s.dry:
            name = "dry_" + name
        p = rc.save_episode(name, self.trans, dict(version=rc.PRIOR_VERSION, arm=self.arm, scripted_rev=SCRIPTED_REV, mode=self.mode, episode=self.ep, dry=self.s.dry, ret=self.ret, check=ok, terminal=self.terminal,
                                                   reason=(rep or {}).get("reason"), events=self.events, policy=self.actor.info,
                                                   attempt=self.s.A, t=time.time()))
        self.s.summary(transitions_file=os.path.relpath(p, rc.rl_dir()), transitions=len(self.trans), **{"return": round(self.ret, 3)})

    def run(self):
        s = self.s
        s.summary(first_failed_check=None, transitions=0, **{"return": 0.0}, policy=self.actor.info if self.arm != "scripted" else dict(source="scripted baseline"),
                  rl_mode=self.mode, episode=self.ep, arm=self.arm, scripted_rev=SCRIPTED_REV)
        w = self.perceive()
        g = self.grasp(w)
        if g is None:
            s.summary(first_failed_check="pull not held")
            self.finish()
            return
        pl = self.plan_place(w, g)
        s.decide(f"turn the drawer {pl['dyaw']:+.1f} deg, tool yaw {pl['yaw_f']:.1f}, opening {np.round(w['c_o'], 4).tolist()} yaw {w['yaw_o']:.1f}")
        self.carry(w, g, pl)
        ins = self.insert_sim(w, g, pl) if self.arm == "sim" else self.insert(w, g, pl)
        self.release(w, g, pl, ins)
        self.finish()


with RobotSession(program="execute") as sess:
    ep = Episode(sess)
    try:
        ep.run()
    finally:
        ep.save_transitions()
