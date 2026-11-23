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
DRAWER_L = 0.083                # back wall to front face (flush top 0.096 - compartment floor 0.013)
KNOB_OFF = 0.0168               # drawer centre to knob centre, along the knob axis (attempt 3, two wrist views)
TIP_CLOSED = 0.047              # closed fingertips below the tool frame
TIP_CLEAR = 0.006               # closed fingertips above the drawer's top face at the pinch
PRE = 0.55                      # pre-close (~38 mm) around the 15 mm knob
GRIP = 0.95
HOLD = (60.0, 96.0)             # reading when the knob is held
LIFT = 0.12
TOL = 0.02                      # orchestrator layout tolerance
GOOD = 0.012                    # correct above this
YAW_RANGE = 25.0                # deg, inside the allowed +/-30


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

    def contact_line(self, name, T, limit=22.0):
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
        held = HOLD[0] <= g <= HOLD[1]
        if h >= 0.20 and not held:
            if g > 3.0:
                s.gripper(0.0, "open: nothing is held at this height")
            return False
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
                 if 0.05 <= b["height"] <= 0.14 and 0.05 <= max(b["extent"]) <= 0.19 and b["n"] > 150]
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

    def locate(self, p):
        if p["blobs"]:
            b = p["blobs"][0]
            return dict(where="table", xy=np.array(b["xy"], float), h_top=float(b["height"]))
        filled = [k for k, v in enumerate(p["comps"]) if v is not None and v > 0.06]
        if filled:
            k = max(filled, key=lambda i: p["comps"][i])
            return dict(where=f"compartment_{k}", xy=COMP_0 + k * PITCH * A, h_top=float(np.clip(p["comps"][k], 0.085, 0.125)))
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
            return dict(knob=loc["xy"] + KNOB_OFF * ax, axis=ax, r_m=0.0, frame=[])
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
            ok2 = sim.check(seg2.q, s.kin, SLOW, gripper_frac=GRIP, table_z=self.tz)[0]
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
            h_g = loc["h_top"] + TIP_CLOSED + TIP_CLEAR
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
            if not HOLD[0] <= g <= HOLD[1]:
                s.decide(f"gripper reads {g:.1f}: the knob is not between the pads")
                s.gripper(PRE, "release")
                self.rise("up_after_miss", h_g + 0.09)
                continue
            with ForceLimit(s, 45.0):
                T = self.T_now()
                T[2, 3] += LIFT
                self.line("lift_drawer", T, "slow")
            g2 = float(s.feedback()["gripper"]) if not s.dry else g
            if not HOLD[0] <= g2 <= HOLD[1] + 1.0:
                s.decide(f"after the lift the gripper reads {g2:.1f}: the drawer was lost")
                s.gripper(0.0, "open")
                continue
            return dict(knob=v["knob"], axis=v["axis"], h_g=h_g, bottom=loc["h_top"] - DRAWER_L, yaw=yaw)
        s.require(False, f"could not grasp the knob after two tries ({loc['where']})")

    def yaw_delta(self, ax):
        """Rotation (deg) that brings the knob axis to a random yaw within +/-YAW_RANGE of the cabinet's -A direction
        (or the opposite direction, whichever is nearer)."""
        nominal = np.degrees(np.arctan2(-A[1], -A[0]))
        now = wrap(np.degrees(np.arctan2(ax[1], ax[0])) - nominal)
        goal = float(self.rng.uniform(-YAW_RANGE, YAW_RANGE))
        d = wrap(goal - now)
        return d if abs(d) <= 90 else wrap(d + 180.0)

    def place(self, grip, aim, delta):
        """Carry the held drawer so its centre is over aim, turned by delta deg, lower it onto the table, release."""
        s = self.s
        s.phase("place")
        T0 = self.T_now()
        R = rz(delta)
        off = KNOB_OFF * (R[:2, :2] @ grip["axis"]) if grip.get("axis") is not None else np.zeros(2)
        Tc = T0.copy()
        Tc[:3, :3] = R @ T0[:3, :3]
        Tc[:2, 3] = aim + off
        s.decide(f"carry: drawer centre to {np.round(aim, 3).tolist()}, turn {delta:+.1f} deg")
        try:
            self.line("carry", Tc, "normal")
        except (RuntimeError, ValueError) as e:
            s.log(f"straight carry not plannable ({str(e)[:80]}): joint move")
            s.move_joint("carry_joint", self.ik(Tc), "normal")
        h_place = grip["h_g"] - grip["bottom"] + 0.001
        T1 = Tc.copy()
        T1[2, 3] = self.tz + h_place + 0.05
        self.line("lower_fast", T1, "normal")
        T2 = Tc.copy()
        T2[2, 3] = self.tz + h_place
        touched = not self.contact_line("lower_to_table", T2, 20.0)
        s.gripper(PRE, "release the knob")
        s.decide(f"released {'on contact' if touched else 'at the planned height'}, tool "
                 f"{self.T_now()[2, 3] - self.tz:.3f} m above the table")
        self.rise("rise_after_release", h_place + 0.07)
        self.rise("rise_clear", 0.25, "normal")
        s.gripper(0.0, "open")

    def measure(self):
        """The drawer's position as the orchestrator sees it: side-camera blobs and the task's size prior."""
        blobs = [b for b in self.s.tabletop() if layout.matches(PRIOR, b)]
        if not blobs:
            return None, None
        b = min(blobs, key=lambda c: float(np.linalg.norm(np.subtract(c["xy"], self.target))))
        return np.array(b["xy"], float), float(b["height"])

    # ------------------------------------------------------------ main
    def run(self):
        s = self.s
        s.summary(measured=None, target=self.target.round(4).tolist())
        s.phase("clear")
        held = self.escape()
        grip = None
        if held:
            s.decide("the drawer is already in the gripper: carry it to the target")
            T = self.T_now()
            h_g = T[2, 3] - self.tz - LIFT                # unknown grasp: lower onto the table under the force stop
            grip = dict(knob=None, axis=None, h_g=max(h_g, DRAWER_L + TIP_CLOSED + TIP_CLEAR), bottom=0.0, yaw=None)
        else:
            self.go_home()
        aim = self.target.copy()
        err, m, hb = None, None, None
        for rnd in range(3):
            if grip is None:
                s.phase(f"scan_{rnd}")
                p = self.scan(str(rnd))
                loc = self.locate(p)
                s.require(loc is not None, f"drawer not found: no table blob, compartments {p['comps']}")
                s.decide(f"round {rnd}: drawer at {loc['where']} {np.round(loc['xy'], 3).tolist()}, top {loc['h_top']:.3f} m")
                grip = self.pick(loc, str(rnd))
            delta = self.yaw_delta(grip["axis"]) if rnd == 0 and grip.get("axis") is not None else 0.0
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
            if err <= GOOD:
                break
            aim = aim - np.clip(m - self.target, -0.06, 0.06)
        s.phase("verify")
        seen = self.look(m, hb, hb + 0.165, "verify") is not None
        self.go_home()
        m2, hb2 = self.measure()
        if m2 is not None:
            m, err = m2, float(np.linalg.norm(m2 - self.target))
        ok = bool(err is not None and err <= TOL and seen and m2 is not None)
        s.summary(measured={"drawer": np.round(m, 4).tolist()}, errors={"drawer": round(err, 4)}, pull_seen_from_above=seen,
                  verify=dict(tolerance=TOL, height=hb2))
        s.decide(f"final error {err * 100:.1f} cm, pull seen from above: {seen}")
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
