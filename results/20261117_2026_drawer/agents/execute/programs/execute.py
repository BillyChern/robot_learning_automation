"""drawer_rl episode: scripted grasp + carry to the slot front, learned insertion (held) and push (tilted closed
gripper), scripted retreat, verification with the orchestrator's own geometric check. See notes/design.md.

Env: RLPJ_RL_DIR (rl/), RLPJ_RL_MODE (train|eval), RLPJ_EPISODE (-1 = test run)."""
import os
import sys
import time

os.environ.setdefault("RLPJ_TASK", "drawer_rl")
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
import pullface as PF

TASK = config.load("drawer_rl")["task"]
CAB = TASK["fixtures"]["cabinet"]
FLUSH_TOL = TASK["fixtures"].get("flush_tol", 0.012)
N = np.array(CAB["front_normal"], float)             # into the cabinet
TT = np.array([-N[1], N[0]])                         # along the front
OFF = float(CAB["front_offset"])
N3, T3, Z3 = np.r_[N, 0.0], np.r_[TT, 0.0], np.array([0, 0, 1.0])

HOME_Q = np.array([0, 15, -130, 0, 55, 90.0])        # camera-clear pose (reset agent)
SEED_DOWN = np.array([0, 25, -110, 0, -45, 90.0])
READY_XYZ = (0.40, -0.10, 0.25)
TIP_OPEN = 0.027
HOLD_MAX = 97.0
SAFE_H = 0.20
WRIST_SIDE_BIAS = np.array([0.010, 0.058])   # wrist - side camera table positions (reset agent, attempts 150-163)
PLACE_GAP = 0.02             # m: back wall in front of the cabinet face when the drawer is set down for the table alignment
HAND_YAW = 7.0               # deg: tool yaw that makes the held drawer straight (data, see run())
EDGE_TARGET = 0.0045           # m: held-drawer edge-based centre offset of the reached insertions
SLOT_T = CAB["slot_t"]
T_C = 0.5 * (SLOT_T[0] + SLOT_T[1]) - 0.0045   # slot-relative (base moved 2026-11-20): the drawer centre target is T_C + EDGE_TARGET = window centre
BOTTOM_H = 0.018             # drawer bottom above the table while sliding in
BACK_GAP = 0.03              # drawer back in front of the front plane at the start of the insertion
INS_NMAX = OFF - 0.036       # vertical tool axis limit (keep-out: body cylinder 45 mm + 1 cm)
K_INS, K_PUSH = 20, 10
MIN_ADV = 0.004              # m: insertion advance floor unless the last step was hard (peak > F_SOFT)
F_SOFT = 35.0               # N: force penalty above this (sliding takes 22-28 N; user 01:50: was 20, penalised normal sliding)
F_STOP = 40.0               # N: only a value for the observation after a stop (no force cap: user, 22:30)
RELEASE = 0.75               # gripper after the insertion: ~21 mm, inner finger ~1.7 cm in front of the cabinet face
                             # (attempt 5: 0.40 put it under the rail above the slot, the lift caught it: 99.6 N, cabinet moved 6 cm)
L_FACE = 0.13                # drawer outer size for the wall geometry of the pull detection (real ~0.13)
MAX_FORCE = 3                # own force stops per insertion before the stage ends
L_MIN = 0.155                # drawer depth floor: the side camera sees only part of the rim (dry run 1: 0.093;
                             # reset agent: obb 0.154 x 0.14-0.15), a short estimate would put its back into the cabinet
PUSH_GRIP = 0.55            # pusher opening (ep 27: closed tips pushed the knob, drawer stopped 2.6 cm out): the two fingers
                            # straddle the knob (~3.8 cm gap) and push the face on both sides of it
PUSH_TILT, PUSH_H, PUSH_ZT = 35.0, 0.048, 0.027 + 0.020 * 0.55   # tilt, fingertip height (knob height), tip along tool z
RETREAT_F = 10.0             # N: a retreat step above this drags the drawer (ep 32: 53 / 42 N)
PUSH_DT = 0.0               # push on the drawer centre line (above the knob): every off-centre push (1.5-3 cm) jammed on contact (att 43,44,72, ep 4,5,8,25), the centred ones slid (att 7, 14, 20)
PUSH_TMAX = SLOT_T[1] - 0.015  # closed fingers (+-14 mm along t in the model) clear of the slot-side keep-out box (t > slot_t[1])
PUSH_NMAX = OFF + 0.012       # model tip; the flow test refuses steps beyond ~+0.012 (a static probe accepted +0.024: path differs)
PUSH_HI = 0.14
TIP_MARGIN_PUSH = 0.004


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def rot2(v, deg):
    a = np.radians(deg); c, s_ = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def nt(xy):
    xy = np.asarray(xy, float)
    return np.stack([xy[..., 0] * N[0] + xy[..., 1] * N[1], xy[..., 0] * TT[0] + xy[..., 1] * TT[1]], -1)


def xy_of(n, t):
    return n * N + t * TT


PUSH_FRAME = dict(yaw=0.0, n0=OFF, t0=0.0)   # push direction: N turned by yaw about (n0, t0) (set per episode)


def push_dirs():
    a = np.radians(PUSH_FRAME["yaw"]); c, s_ = np.cos(a), np.sin(a)
    u = np.array([c * N[0] - s_ * N[1], s_ * N[0] + c * N[1]])
    return u, np.array([-u[1], u[0]])


def push_T(n, t, h, tz):
    """Tilted closed gripper: tool z PUSH_TILT from vertical toward the push direction u (N turned by
    PUSH_FRAME yaw: the slot axis the insertion found), jaw axis across it; fingertip (PUSH_ZT along z) at
    (n, t) in the turned frame about (n0, t0), height h above the table."""
    u, v = push_dirs()
    th = np.radians(PUSH_TILT)
    z = np.r_[np.sin(th) * u, -np.cos(th)]
    x = np.r_[v, 0.0]
    T = np.eye(4); T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    xy = xy_of(PUSH_FRAME["n0"], PUSH_FRAME["t0"]) + (n - PUSH_FRAME["n0"]) * u + (t - PUSH_FRAME["t0"]) * v
    T[:3, 3] = np.r_[xy, tz + h] - PUSH_ZT * z
    return T


class Episode:
    def __init__(self, s):
        self.s = s
        self.tz = calib.table_z()
        self.mode = os.environ.get("RLPJ_RL_MODE", "train")
        self.ep = int(os.environ.get("RLPJ_EPISODE", "-1"))
        self.rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
        self.actor = rc.Actor()
        self.trans = []          # dicts obs, act, rew (next_obs/done filled at the end)
        self.events = []
        self.ret = 0.0
        self.orient_state = "unknown"
        self.t_line = T_C

    # ------------------------------------------------------------ motion helpers (reset agent's)
    def log(self, m):
        self.s.log(m)

    def q_now(self):
        return np.asarray(self.s.feedback()["q"], float)

    def T_now(self):
        return calib.pose_to_T(self.s.feedback()["pose"])

    def ik(self, T, seeds=None):
        best = None
        for sd in (seeds or [self.q_now(), SEED_DOWN]):
            try:
                q, _, _ = self.s.planner.ik_exact(T, sd)
            except Exception as e:
                self.log(f"IK from seed failed: {str(e)[:80]}"); continue
            d = float(np.max(np.abs(q - self.q_now())))
            if best is None or d < best[1]:
                best = (q, d)
        self.s.require(best is not None, f"no IK for tool at {np.round(T[:3, 3], 3).tolist()}")
        return best[0]

    def joint_to(self, name, T, speed="normal", seeds=None):
        self.s.move_joint(name, self.ik(T, seeds), speed)

    def line(self, name, T, speed="slow", tip_margin=0.008):
        self.s.move_line(name, T, speed, tip_margin=tip_margin)

    def to_ready(self):
        T = tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0)
        self.s.move_joint("ready", self.ik(T, [SEED_DOWN]), "normal")

    def go_home(self, name):
        self.to_ready()
        self.s.move_joint(name, HOME_Q, "normal")
        self.set_home()

    def set_home(self, q=None):
        s = self.s
        q = HOME_Q.copy() if q is None else np.asarray(q, float)
        if s.q_home is not None and np.abs(wrap180(np.asarray(s.q_home) - q)).max() < 1:
            return
        s.q_home = q.copy()
        if getattr(s, "wd", None) is not None:
            try:
                s.wd.stdin.write(("HOME " + " ".join(f"{v:.4f}" for v in q) + "\n").encode()); s.wd.stdin.flush()
            except Exception as e:
                s.log(f"watchdog home update failed: {e}")

    def lift_to(self, name, h, speed="normal"):
        T = self.T_now(); T[2, 3] = self.tz + h; self.line(name, T, speed)

    def escape(self):
        T = self.T_now()
        if T[2, 3] > self.tz + 0.15:
            return
        self.s.decide(f"start pose low ({T[2, 3] - self.tz:.3f} m): back off and up first")
        ds = (0.0,) if T[2, 2] < -0.9 else (0.08, 0.05, 0.03, 0.0)
        for d in ds:
            T2 = T.copy(); T2[:2, 3] -= d * N; T2[2, 3] = max(T[2, 3], self.tz + 0.10) + 0.06
            try:
                self.line("escape_up", T2, "slow", tip_margin=TIP_MARGIN_PUSH); return
            except Abort as e:
                self.log(f"escape back {d} refused: {str(e)[:100]}")
        self.s.require(False, "no escape path from the start pose")

    def snap(self, tag):
        """Diagnostic side + wrist images (<= 640 px) at a stage end (tuning 03:08: where does the drawer stop?)."""
        try:
            self.save(f"{tag}_side", self.s.side.latest()[0]); self.save(f"{tag}_wrist", self.s.wrist.latest()[0])
        except Exception as ex:
            self.s.log(f"snap {tag}: {ex}")

    def save(self, name, img, w=640):
        if img is not None:
            cv2.imwrite(os.path.join(self.s.A, f"{name}.jpg"), cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1]))))

    # ------------------------------------------------------------ perception
    def scene(self, tag):
        P = tt.cloud(self.s.side, n=5)
        tz = tt.table_fit(P)
        blobs = tt.detect(P, table_z=tz)
        self.save(f"side_{tag}", self.s.side.latest()[0])
        return P, tz, blobs

    def find_drawer(self, blobs, tz):
        cand = [b for b in blobs if nt(b["xy"])[0] < OFF - 0.03 and 0.03 <= b["height"] <= 0.12
                and 0.06 <= max(b["extent"]) <= 0.32 and b["n"] > 150]
        if not cand:
            return None
        b = max(cand, key=lambda b: b["n"])
        pts = b["points"]; h = pts[:, 2] - tz; h_top = float(np.percentile(h, 98))
        top = pts[h > h_top - 0.012, :2]
        (cx, cy), (w0, h0), a = cv2.minAreaRect(np.ascontiguousarray(top.astype(np.float32)))
        c = np.array([cx, cy]); u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]); v = np.array([-u[1], u[0]])
        faces = [(u, w0 / 2), (-u, w0 / 2), (v, h0 / 2), (-v, h0 / 2)]
        p, half = max(faces, key=lambda f: float(f[0] @ -N))     # outward normal closest to -N: the pull wall
        # the side camera sees the rim only partly (rect ~0.10-0.12 of ~0.13): its centre is biased toward the camera.
        # Along the pull wall, move the centre away from the camera by half the missing width (tuning 00:30: the grasp,
        # and so the drawer in the hand, sat ~1-2 cm toward -t; insertions jam at a fixed depth)
        along = h0 if abs(float(p @ u)) > 0.7 else w0
        wdir = np.array([-p[1], p[0]])
        cam = calib.T_base_side()[:2, 3]
        away = wdir if float(wdir @ (c - cam)) > 0 else -wdir
        shift_w = float(np.clip((L_FACE - along) / 2, 0.0, 0.02))
        c = c + away * shift_w
        return dict(xy=np.round(b["xy"], 4).tolist(), c=c, p=p, half=float(half), h_top=h_top, size=[round(w0, 3), round(h0, 3)], shift_w=round(shift_w, 4),
                    rect_ang=round(a, 1), front_dev_deg=round(wrapd(ang(p) - ang(-N)), 1), n=int(b["n"]))

    def measure_held(self, n_tool):
        """Side depth of the held drawer in front of the slot: t of its centre and yaw of its walls vs N."""
        P = tt.cloud(self.s.side, n=3)
        q = nt(P[:, :2]); h = P[:, 2] - self.tz
        m = (h > 0.025) & (h < 0.058) & (q[:, 0] > n_tool - 0.03) & (q[:, 0] < OFF - 0.005) & (np.abs(q[:, 1] - T_C) < 0.11)
        self.save("side_held", self.s.side.latest()[0])
        if m.sum() < 80:
            return dict(ok=False, n=int(m.sum()), t_err=0.0, yaw_err=0.0)
        Q = q[m]
        t_c = 0.5 * (np.percentile(Q[:, 1], 3) + np.percentile(Q[:, 1], 97))
        (_, _), (_, _), a = cv2.minAreaRect(np.ascontiguousarray(P[m, :2].astype(np.float32)))
        dev = wrapd(a - ang(N)); dev = (dev + 45.0) % 90.0 - 45.0
        return dict(ok=True, n=int(m.sum()), t_err=round(float(t_c - T_C), 4), yaw_err=round(float(dev), 2),
                    t_span=round(float(np.percentile(Q[:, 1], 97) - np.percentile(Q[:, 1], 3)), 3))

    def force(self):
        f = self.s.feedback().get("force")
        return np.zeros(3) if f is None else np.asarray(f, float)

    # ------------------------------------------------------------ learned steps
    def policy(self, obs):
        a, info = self.actor.act(obs, explore=(self.mode == "train"), rng=self.rng)
        return rc.anchor(a, obs), info

    def try_step(self, name, T, tip_margin):
        """Model-check a straight slow line to T; run it. Returns (status, result): ok | refused | force | abort."""
        s = self.s
        try:
            q0 = s.measured_q(s.last_ref())
            seg = s.planner.cart_line(name, q0, T, SLOW)
        except Exception as e:
            return "refused", dict(reason=f"plan: {str(e)[:100]}")
        ok, rep = sim.check(seg.q, s.kin, seg.speed, gripper_frac=s.grip_frac, tip_margin=tip_margin,
                            table_z=max(s.table_z, calib.table_z()), max_radius=0.80)
        if not ok:
            return "refused", dict(reason="model", keep_out=rep.get("keep_out_violations"), tip=round(rep["min_tip_above_table"], 3))
        try:
            r = s.run_segment(seg, tip_margin=tip_margin)
            return "ok", r
        except Abort as e:
            msg = str(e)
            return ("force" if "contact force" in msg else "abort"), dict(reason=msg[:160])

    def obs(self, stage, prog, dt, dz, dyaw, ma, mb, f, peak, k, K, hold):
        o = np.array([stage, prog, dt / 0.01, dz / 0.01, dyaw / 5.0, ma, mb, *((f) / 10.0), peak / 10.0, k / K, hold], float)
        return np.clip(o, -5, 5)

    def stage_insert(self, n0, z0, yaw2, meas):
        """Learned insertion while holding the drawer. Returns dict(end, n, t)."""
        s = self.s
        s.phase("insert_learned")
        TL = self.t_line
        n, t, z, dyaw = n0, TL, z0, 0.0
        f0 = self.force(); peak = 0.0
        # tuning 03:08: the held-drawer side-depth measurement is not trustworthy (t err a steady -2 cm, yaw swinging
        # +-12 deg while the wrist sees the wall within 2 deg of the jaws) and it confounded the data (the action sign
        # followed it): it stays in the summary but is no longer observed (obs 5, 6 = 0; learner zeroes old data too)
        ma, mb = 0.0, 0.0
        end = "steps"; n_force = 0
        for k in range(K_INS):
            f = self.force() - f0
            o = self.obs(0, (n - INS_NMAX) / 0.1, t - TL, z - z0, dyaw, ma, mb, np.array([f @ N3, f @ T3, f @ Z3]), peak, k, K_INS, 1)
            a, info = self.policy(o)
            adv, ddt, ddz, ddy = 0.0075 * (1 + a[0]), 0.004 * a[1], 0.001 * a[2], 2.0 * a[3]
            if adv < MIN_ADV and peak <= F_SOFT:
                adv = MIN_ADV
                a = a.copy(); a[0] = adv / 0.0075 - 1.0          # log the action that was executed
            tgt = (min(n + adv, INS_NMAX), float(np.clip(t + ddt, TL - 0.025, TL + 0.025)),
                   float(np.clip(z + ddz, z0 - 0.003, z0 + 0.002)), float(np.clip(dyaw + ddy, -10, 10)))
            status, res, scale = "refused", {}, 1.0
            for _ in range(3):
                nn = n + scale * (tgt[0] - n); tt_ = t + scale * (tgt[1] - t); zz = z + scale * (tgt[2] - z); yy = dyaw + scale * (tgt[3] - dyaw)
                T = tool_down_T(np.r_[xy_of(nn, tt_), zz], yaw2 + yy)
                status, res = self.try_step(f"ins{k:02d}", T, 0.008)
                if status != "refused":
                    break
                scale *= 0.5
            r = -0.02
            if status == "ok":
                r += 10.0 * (nn - n)
                peak = float(res.get("max_force_n", 0.0))
                r -= 0.1 * max(0.0, peak - F_SOFT) / 5.0
                n, t, z, dyaw = nn, tt_, zz, yy
            elif status == "refused":
                r -= 1.0
            elif status == "force":
                # own 40 N stop: the arm halted part-way; continue from the measured pose so the policy can correct
                # (t, z, yaw; advance 0) with the force in its observation; the stage ends after MAX_FORCE stops
                r -= 1.0
                peak = F_STOP
                n_force += 1
                self.unload(-N3, steps=3)
                if not s.dry:
                    # user 00:30: a jam means it was not straight: back out 1.5 cm, re-align, try again (no twisting)
                    T_b = self.T_now(); T_b[:2, 3] -= 0.015 * N
                    try:
                        self.line(f"back_out{n_force}", T_b, "slow")
                    except Abort as e:
                        s.log(f"back out: {str(e)[:100]}")
                    xyz = self.T_now()[:3, 3]; q2 = nt(xyz[:2]); n, t, z = float(q2[0]), float(q2[1]), float(xyz[2])
                    # no lateral re-alignment while the drawer is partly in the slot (attempt 84: the side view then
                    # does not follow the tool, the re-aligns walked the tool 2.5 cm sideways against the slot):
                    # back to the aligned line only
                    t = self.t_line
                    T_l = tool_down_T(np.r_[xy_of(n, t), z], yaw2 + dyaw)
                    try:
                        self.line(f"back_to_line{n_force}", T_l, "slow")
                    except Abort as e:
                        s.log(f"back to line: {str(e)[:100]}")
            else:
                r -= 2.0
                peak = F_STOP
            g = float(self.s.feedback()["gripper"])
            dropped = (not s.dry) and g >= HOLD_MAX
            if dropped:
                r -= 2.0
            self.trans.append(dict(obs=o, act=a, rew=r))
            self.ret += r
            self.events.append(["ins", k, status, np.round(a, 2).tolist(), round(r, 3), round(n - OFF, 4), round(t - TL, 4), round(z - z0, 4), round(dyaw, 1), round(peak, 1)])
            self.log(f"insert step {k}: a {np.round(a, 2).tolist()} (mu {info['mu']}) -> {status} {res.get('reason', '')} n-OFF {n - OFF:+.4f} dt {t - TL:+.4f} dz {z - z0:+.4f} dyaw {dyaw:+.1f} peak {peak:.1f} N r {r:+.3f}")
            if dropped:
                end = "dropped"; break
            if status == "force" and n_force < MAX_FORCE:
                continue
            if status != "ok":
                end = status; break
            if n >= INS_NMAX - 0.003:      # the last mm are often refused (model shrinks the step)
                end = "reached"; break
        self.snap("insert_end")
        s.summary(insert=dict(end=end, steps=len(self.trans), n_minus_off=round(n - OFF, 4), dt=round(t - TL, 4), t_line=round(TL, 4),
                              dz=round(z - z0, 4), dyaw=round(dyaw, 1)))
        return dict(end=end, n=n, t=t, z=z, yaw=yaw2 + dyaw)

    def stage_push(self, face_n, t0, ins_ok):
        s = self.s
        s.phase("push_learned")
        n = max(min(face_n - 0.025, PUSH_NMAX - 0.03), OFF - 0.085); t, h = t0, PUSH_H
        T_pre = push_T(n, t, h, self.tz)
        T_hi = T_pre.copy(); T_hi[2, 3] += 0.06
        s.gripper(PUSH_GRIP, "fingers apart: they straddle the knob")
        q = self.ik(T_hi, [np.array([0, 40, -120, 0, j5, j6], float) for j5 in (20.0, 40.0, 60.0) for j6 in (-90.0, 90.0)])
        s.move_joint("above_push", q, "normal")
        with self.low_force(20):
            self.line("to_push_pre", T_pre, "normal", tip_margin=TIP_MARGIN_PUSH)
        f0 = self.force(); peak = 0.0
        ma = float(np.clip((OFF - face_n) / 0.05, -3, 3)); mb = 1.0 if ins_ok else 0.0
        end = "steps"; k0 = len(self.trans); n_force = 0; jam_steps = 0
        for k in range(K_PUSH):
            f = self.force() - f0
            o = self.obs(1, (n - PUSH_NMAX) / 0.1, t - t0, h - PUSH_H, 0.0, ma, mb, np.array([f @ N3, f @ T3, f @ Z3]), peak, k, K_PUSH, 0)
            a, info = self.policy(o)
            adv, ddt, ddz = 0.0075 * (1 + a[0]), 0.003 * a[1], 0.003 * a[2]
            tgt = (min(n + adv, PUSH_NMAX), float(np.clip(t + ddt, t0 - 0.015, min(t0 + 0.006, PUSH_TMAX + 0.002))), float(np.clip(h + ddz, 0.043, 0.054)))
            status, res, scale = "refused", {}, 1.0
            for _ in range(3):
                nn = n + scale * (tgt[0] - n); tt_ = t + scale * (tgt[1] - t); hh = h + scale * (tgt[2] - h)
                status, res = self.try_step(f"push{k:02d}", push_T(nn, tt_, hh, self.tz), TIP_MARGIN_PUSH)
                if status != "refused":
                    break
                scale *= 0.5
            r = -0.02
            if status == "ok":
                peak = float(res.get("max_force_n", 0.0))
                r -= 0.1 * max(0.0, peak - F_SOFT) / 5.0
                # operator 01:55 jam rule: a hard step (> 35 N) whose tip did not really advance (< 3 mm, measured)
                # twice in a row ends the push: no repeated full-force steps into a jam
                if not s.dry:
                    Tm = self.T_now(); tip = Tm[:3, 3] + PUSH_ZT * Tm[:3, 2]
                    adv_real = float((tip[:2] - xy_of(PUSH_FRAME["n0"], PUSH_FRAME["t0"])) @ push_dirs()[0]) + PUSH_FRAME["n0"] - n
                    jam_steps = jam_steps + 1 if (peak > 35.0 and adv_real < 0.003) else 0
                n, t, h = nn, tt_, hh
            elif status == "force" and n_force < MAX_FORCE and not s.dry:
                # 00:05: a held drawer needs ~45-50 N to break free (ep 10 broke free at 47 N, then slid at 4-14 N):
                # unload, re-read the tip from the measured pose and try again (up to MAX_FORCE stops)
                r -= 1.0; peak = F_STOP; n_force += 1
                self.unload(-np.r_[push_dirs()[0], 0.0], steps=3)
                Tm = self.T_now(); tip = Tm[:3, 3] + PUSH_ZT * Tm[:3, 2]
                u, v = push_dirs(); rel = tip[:2] - xy_of(PUSH_FRAME["n0"], PUSH_FRAME["t0"])
                n, t, h = PUSH_FRAME["n0"] + float(rel @ u), PUSH_FRAME["t0"] + float(rel @ v), float(tip[2] - self.tz)
                status = "force_retry"
            else:
                r -= 2.0 if status in ("force", "abort") else 1.0
                peak = 25.0 if status in ("force", "abort") else peak
            self.trans.append(dict(obs=o, act=a, rew=r))
            self.ret += r
            self.events.append(["push", k, status, np.round(a, 2).tolist(), round(r, 3), round(n - OFF, 4), round(t - t0, 4), round(h, 4), round(peak, 1)])
            self.log(f"push step {k}: a {np.round(a, 2).tolist()} (mu {info['mu']}) -> {status} {res.get('reason', '')} tip n-OFF {n - OFF:+.4f} dt {t - t0:+.4f} h {h:.3f} peak {peak:.1f} N r {r:+.3f}")
            if status not in ("ok", "force_retry"):
                end = status; break
            if jam_steps >= 2:
                s.decide("push: two hard steps without progress: jammed, stop pushing")
                end = "jam"; break
            if n >= PUSH_NMAX - 0.0005:
                end = "reached"; break
        self.snap("push_end")
        try:                                   # diagnostic: the slot right after the push (arm still in front of it)
            P2 = tt.cloud(s.side, n=3); s.summary(slot_after_push=checks.slot_profile(P2, tt.table_fit(P2), CAB, FLUSH_TOL))
        except Exception as ex:
            s.log(f"slot after push: {ex}")
        s.summary(push=dict(end=end, steps=len(self.trans) - k0, tip_n_minus_off=round(n - OFF, 4), dt=round(t - t0, 4), h=round(h, 4)))
        if end in ("force", "abort") and not s.dry:
            self.unload(-np.r_[push_dirs()[0], 0.0])
            xyz = self.T_now()[:3, 3] + PUSH_ZT * self.T_now()[:3, 2]; q2 = nt(xyz[:2])
            n, t, h = float(q2[0]), float(q2[1]), float(xyz[2] - self.tz)
        self.push_retreat(n, t, h)
        return end

    def chain_ok(self, poses, tip_margin=TIP_MARGIN_PUSH):
        """Model-check a chain of straight normal-speed lines from the current joints; True when all pass."""
        s = self.s
        q = s.measured_q(s.last_ref())
        for i, T in enumerate(poses):
            try:
                seg = s.planner.cart_line(f"chk{i}", q, T, SLOW)
            except Exception:
                return False
            ok, _ = sim.check(seg.q, s.kin, seg.speed, gripper_frac=s.grip_frac, tip_margin=tip_margin, table_z=max(s.table_z, calib.table_z()))
            if not ok:
                return False
            q = seg.q[-1]
        return True

    def unload(self, d3, steps=5, dist=0.0015):
        """After a force stop the arm is still pressing: back off in tiny slow lines along d3 (unit, base frame) so
        each segment releases only part of the preload (the force stop watches the change within a segment:
        attempt 7's 6 cm retreat tripped at 51 N while unloading)."""
        for i in range(steps):
            T = self.T_now(); T[:3, 3] += dist * np.asarray(d3, float)
            try:
                self.line(f"unload{i}", T, "slow", tip_margin=TIP_MARGIN_PUSH)
            except Abort as e:
                self.log(f"unload {i}: {str(e)[:100]}")

    def reactive_retreat(self, n, t, h, step=0.003, max_steps=50):
        """Leave the drawer front in small steps without dragging it: per step try up, back (-u), down-back; undo a
        step whose peak force change exceeds RETREAT_F. Done when 4 cm in front of the plane or 5 cm higher."""
        s = self.s
        u = push_dirs()[0]
        h0 = h
        for i in range(max_steps):
            # attempt 112: stopping once 5 cm up left the tool at n-OFF -0.011, where no clear-out chain passed the
            # model: keep stepping (up while allowed and < 6 cm, else back) until 4 cm in front of the plane
            if n <= OFF - 0.04:
                s.summary(retreat=dict(steps=i, n=round(n - OFF, 4), h=round(h, 4)))
                self._ret = (n, h)
                return True
            moved = False
            for dn, dh in ((0.0, step), (-step, 0.0), (-step, -0.002)):
                if h + dh < 0.035 or h + dh > h0 + 0.06:
                    continue
                T = push_T(n + dn, t, h + dh, self.tz)
                with self.low_force(20):
                    st, r = self.try_step(f"retreat{i}", T, TIP_MARGIN_PUSH)
                if st == "ok" and float(r.get("max_force_n", 0.0)) <= RETREAT_F:
                    n, h = n + dn, h + dh; moved = True
                    break
                if st in ("ok", "force"):         # undo the dragging step
                    try:
                        self.line(f"retreat_undo{i}", push_T(n, t, h, self.tz), "slow", tip_margin=TIP_MARGIN_PUSH)
                    except Abort as e:
                        s.log(f"retreat undo: {str(e)[:80]}")
            if not moved:
                break
        s.summary(retreat=dict(steps=i, n=round(n - OFF, 4), h=round(h, 4), stuck=True))
        self._ret = (n, h)
        return n <= OFF - 0.04

    def push_retreat(self, n, t, h):
        """Leave the drawer front: back off horizontally to OFF-0.04 (if deeper), then up, then READY; the first
        candidate chain that passes the model check is executed (tools/offline_test.py grid)."""
        # ep 28: the push reached the limit, then the 5 cm retreat pulled 27 N and the drawer ended 1.7 cm out: the
        # fingers straddling the knob caught it. Back off 4 mm, open wider (0.35: ~5.5 cm gap, still inside the slot
        # window in the model) and only then retreat.
        s = self.s
        if s.grip_frac > 0.16 and s.grip_frac < 0.9:
            Tq = push_T(n - 0.004, t, h, self.tz)
            if self.chain_ok([Tq]):
                self.line("push_clear_knob", Tq, "slow", tip_margin=TIP_MARGIN_PUSH)
                n = n - 0.004
            s.gripper(0.15, "open wide (~7 cm): clear the pull tab and knob before the retreat")
        # operator 18:10: retreat UP first (clear of the knob and label plate), then back: back off only to just in
        # front of the face (model), lift 5 cm, then back to OFF-0.04
        # ep 32: the drawer ended 0.65 cm out after the push, but the retreat pulled 53 N (back) and 42 N (up): the
        # fingers hook the pull tab / label plate. Reactive retreat: 3 mm steps, up first, then back, then down-back;
        # a step that rises above RETREAT_F N is undone and the next direction tried, until clear.
        if not s.dry and self.reactive_retreat(n, t, h):
            n, h = self._ret                            # 4 cm in front of the plane: up, model-checked
            Tc2 = push_T(n, t, h, self.tz); Tc2[2, 3] += 0.05
            if self.chain_ok([Tc2]):
                self.line("retreat_clear_up", Tc2, "normal", tip_margin=TIP_MARGIN_PUSH)
            return                                      # clear of the drawer: READY/home plan from here
        # operator 22:35: straight up before any move along -N where the model allows it (no drag on the knob)
        Th = push_T(n, t, h, self.tz); Th[2, 3] += 0.05
        Thb = Th.copy(); Thb[:2, 3] -= max(0.0, n - (OFF - 0.04)) * push_dirs()[0]
        if self.chain_ok([Th, Thb]):
            s.decide(f"push retreat: straight up at n-OFF {n - OFF:+.3f}, then back")
            self.line("push_ret_up0", Th, "slow", tip_margin=TIP_MARGIN_PUSH)
            self.line("push_ret_back0", Thb, "normal", tip_margin=TIP_MARGIN_PUSH)
            return
        n_up = min(n, OFF - 0.006)
        Tu0 = push_T(n_up, t, h, self.tz); Tu1 = Tu0.copy(); Tu1[2, 3] += 0.05
        Tu2 = Tu1.copy(); Tu2[:2, 3] -= max(0.0, n_up - (OFF - 0.04)) * push_dirs()[0]
        if self.chain_ok([Tu0, Tu1, Tu2]):
            s.decide(f"push retreat: up first at n-OFF {n_up - OFF:+.3f}, then back")
            self.line("push_ret_front", Tu0, "slow", tip_margin=TIP_MARGIN_PUSH)
            self.line("push_ret_up", Tu1, "slow", tip_margin=TIP_MARGIN_PUSH)
            self.line("push_ret_back", Tu2, "normal", tip_margin=TIP_MARGIN_PUSH)
            return
        nb = min(n - 0.01, OFF - 0.04)
        Tb = push_T(nb, t, h, self.tz)
        cands = []
        for up in (0.05, 0.03):
            Tu = Tb.copy(); Tu[2, 3] += up; cands.append(("up", [Tb, Tu]))
        Ta = Tb.copy(); Ta[:3, 3] -= 0.06 * Tb[:3, 2]; cands.append(("axis", [Tb, Ta]))
        for name, chain in cands:
            if self.chain_ok(chain):
                self.s.decide(f"push retreat: back to n-OFF {nb - OFF:+.3f}, then {name}")
                for attempt in range(2):
                    try:
                        self.line("push_back", chain[0], "slow", tip_margin=TIP_MARGIN_PUSH)
                        break
                    except Abort as e:
                        # attempt 14: the retreat caught something (50.5 N): unload, retry once from where it stopped
                        self.log(f"push retreat stopped: {str(e)[:120]}")
                        self.s.require(attempt == 0 and "contact force" in str(e), f"push retreat failed: {str(e)[:120]}")
                        self.unload(-np.r_[push_dirs()[0], 0.0], steps=3)
                self.line("push_up", chain[1], "normal", tip_margin=TIP_MARGIN_PUSH)
                return
        self.s.require(False, "no model-checked retreat from the push pose")

    # ------------------------------------------------------------ episode
    def grasp(self, d, refine=True):
        s = self.s
        c, p, half, h_top = d["c"], d["p"], d["half"], d["h_top"]
        e = c + p * (half - 0.006)
        d0 = -p
        yaw = ang(p); yaw = yaw - 180 if yaw > 90 else yaw + 180 if yaw < -90 else yaw
        tip_h = max(h_top - 0.014, 0.018)
        Tg = tool_down_T(np.r_[e, self.tz + tip_h + TIP_OPEN], yaw)
        T_pre = Tg.copy(); T_pre[2, 3] += 0.05
        T_hi = Tg.copy(); T_hi[2, 3] = self.tz + SAFE_H
        s.decide(f"front wall at {np.round(e, 3).tolist()} (face {d['front_dev_deg']:+.0f} deg from facing the robot), top {h_top:.3f}, depth {2 * half:.3f}")
        s.gripper(0.4, "open")
        self.to_ready()
        self.joint_to("above_front", T_hi)
        self.line("down_to_front_pre", T_pre, "normal")
        time.sleep(0.4)
        img = s.wrist.latest()[0]
        self.save("wrist_front_pre", img)
        sh = self.refine_wall(img, e, d0, h_top) if refine else None
        # straight drawer in the hand (user 00:30): turn the jaws to the wall's angle seen by the wrist camera
        ang_w = self.wall_angle(img, e, d0, h_top)
        if ang_w is not None and 0.5 <= abs(ang_w) <= 15.0:
            yaw = yaw + ang_w; d0 = rot2(d0, ang_w)
        # grasp candidates along d0 (eps 29-30: the side-camera front edge is now ~2-3.5 cm out; the wrist step says
        # +0.024..+0.037 with strength ~100; successful grasps of eps 24-28 used +0.003..+0.025): the refined shift
        # first, then the unrefined point, then +-1.5 cm around the better guess, until the wall is held
        cands = ([sh] if sh is not None else []) + [0.0, 0.015, 0.03, -0.012]
        seen, tried = [], []
        e0 = e.copy(); g = 100.0
        for k, shv in enumerate(cands):
            if any(abs(shv - x) < 0.006 for x in seen) or len(tried) >= 3:
                continue
            seen.append(shv)
            e = e0 + shv * d0
            Tg = tool_down_T(np.r_[e, self.tz + tip_h + TIP_OPEN], yaw); T_pre = Tg.copy(); T_pre[2, 3] += 0.05
            self.line(f"front_pre{k}", T_pre, "slow")
            try:
                with self.low_force(20):
                    self.line(f"down_to_front{k}", Tg, "slow")
            except Abort as e:                           # touched the wall top: a miss, back up and try the next point
                s.log(f"grasp approach {k} stopped: {str(e)[:100]}")
                tried.append([round(shv, 3), "stop"])
                self.line(f"up_after_stop{k}", T_pre, "slow")
                continue
            g = s.gripper(1.0, "grasp the front wall")
            tried.append([round(shv, 3), round(g, 1)])
            if g < HOLD_MAX or s.dry:
                sh = shv
                break
            s.gripper(0.4, "open, missed")
            self.line(f"up_empty{k}", T_pre, "slow")
        s.summary(grasp_reading=round(g, 1), wall_shift=sh, grasp_tries=tried)
        if g >= HOLD_MAX and not s.dry:
            self.lift_to("lift_empty", SAFE_H)
            return None
        self.line("lift_slow", T_pre, "slow")
        self.lift_to("lift", SAFE_H, "normal")
        s.summary(grasp_wall_angle=ang_w)
        # along-wall offset of the tool from the drawer centre (w = d0 turned +90 deg; maps to +t once d0 is on N)
        w_dir = np.array([-d0[1], d0[0]])
        fp_off = float((e - np.asarray(c, float)) @ w_dir) if d.get("wrist_fp") else None
        return dict(yaw=yaw, d0=d0, drop=tip_h + TIP_OPEN, L_back=max(2 * half - 0.006, L_MIN), e=e.copy(), Tg=Tg.copy(),
                    wrist_fp=bool(d.get("wrist_fp")), fp_off=fp_off)

    def wrist_footprint(self, d):
        """Operator 01:50: the side camera's drawer outline is ~5 cm off (along the wall too). Look straight down from
        0.45 m above the side estimate + WRIST_SIDE_BIAS, project wood pixels onto the rim plane, take the drawer
        blob's OBB (re-centre up to 3 looks while it touches the image border), rebuild the front wall from it.
        Returns an updated drawer dict, or d unchanged."""
        s = self.s
        if s.dry:
            return d
        from rlpj.perception.wrist import project_base_to_wrist
        h_top = float(np.clip(d["h_top"], 0.05, 0.085))
        xy0 = np.asarray(d["c"], float) + WRIST_SIDE_BIAS
        for look in range(3):
            T = tool_down_T(np.r_[xy0, self.tz + 0.45], 0.0)
            try:
                self.to_ready()
                self.joint_to(f"look_down{look}", T)
            except (Abort, RuntimeError) as e:
                s.log(f"look-down refused: {str(e)[:80]}"); return d
            time.sleep(0.6)
            img = s.wrist.latest()[0]
            if img is None:
                return d
            self.save(f"wrist_down{look}", img)
            Tm = calib.pose_to_T(s.feedback()["pose"])
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 80) & (hsv[..., 2] > 25)
            g = np.arange(-0.18, 0.1801, 0.003)
            Xg, Yg = np.meshgrid(xy0[0] + g, xy0[1] + g, indexing="ij")
            G = np.stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, self.tz + h_top)], 1)
            uv = project_base_to_wrist(G, Tm)
            H_, W_ = img.shape[:2]
            ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H_)
            edge = np.zeros(Xg.size, bool)
            edge[ok] = (uv[ok, 0] < 12) | (uv[ok, 0] > W_ - 13) | (uv[ok, 1] < 12) | (uv[ok, 1] > H_ - 13)
            occ = np.zeros(Xg.size, np.uint8)
            occ[ok] = mask[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
            occ[nt(G[:, :2])[:, 0] > OFF - 0.015] = 0             # not the cabinet
            occ = cv2.morphologyEx(occ.reshape(Xg.shape), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
            best, bd = None, 9.0
            for kk in range(1, n):
                if st[kk, cv2.CC_STAT_AREA] < 100:
                    continue
                cc = np.argwhere(lab == kk)
                cxy = np.array([Xg[cc[:, 0], cc[:, 1]].mean(), Yg[cc[:, 0], cc[:, 1]].mean()])
                if np.linalg.norm(cxy - xy0) < bd:
                    best, bd = kk, float(np.linalg.norm(cxy - xy0))
            if best is None:
                s.log("wrist footprint: no wood seen"); return d
            cells = np.argwhere(lab == best)
            pts = np.stack([Xg[cells[:, 0], cells[:, 1]], Yg[cells[:, 0], cells[:, 1]]], 1).astype(np.float32)
            border = bool(edge.reshape(Xg.shape)[lab == best].any())
            (cx, cy), (w0, h0), a = cv2.minAreaRect(np.ascontiguousarray(pts))
            c = np.array([cx, cy])
            plausible = 0.09 <= max(w0, h0) <= 0.17 and 0.05 <= min(w0, h0) <= 0.15
            s.summary(**{f"wrist_footprint{look}": dict(c=np.round(c, 4).tolist(), size=[round(w0, 3), round(h0, 3)], ang=round(a, 1), border=border)})
            if plausible and not border:
                u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]); v = np.array([-u[1], u[0]])
                faces = [(u, w0 / 2), (-u, w0 / 2), (v, h0 / 2), (-v, h0 / 2)]
                p, half = max(faces, key=lambda f: float(f[0] @ -N))
                s.decide(f"wrist footprint: centre {np.round(c, 3).tolist()} size {w0:.3f}x{h0:.3f} (side camera {np.round(d['c'], 3).tolist()})")
                return dict(d, c=c, p=p, half=float(half), shift_w=0.0, front_dev_deg=round(wrapd(ang(p) - ang(-N)), 1), wrist_fp=True)
            xy0 = c                                                # cut by the border: re-centre on it
        return d

    def place_in_front(self, d, held, it):
        """Put the held drawer down on the table right in front of the slot: axis on N, centre on the slot centre t,
        back wall PLACE_GAP in front of the face; release, lift. Returns the target centre."""
        s = self.s
        L = float(np.clip(2 * d["half"], 0.10, 0.16))
        c_tgt = xy_of(OFF - PLACE_GAP - L / 2, T_C + EDGE_TARGET)
        yaw_p = wrapd(held["yaw"] + wrapd(ang(N) - ang(held["d0"])))
        off = float(held.get("fp_off") or 0.0)
        xy = c_tgt - N * (L / 2 - 0.006) + off * TT
        Tp = tool_down_T(np.r_[xy, self.tz + held["drop"] + 0.008], yaw_p)
        Th = Tp.copy(); Th[2, 3] = self.tz + SAFE_H
        Tq = Tp.copy(); Tq[2, 3] += 0.04
        self.joint_to(f"above_place{it}", Th)
        self.line(f"down_place_pre{it}", Tq, "normal")
        with self.low_force(25):
            self.line(f"place{it}", Tp, "slow", tip_margin=0.0)
        s.gripper(0.4, "release on the table in front of the slot")
        self.line(f"up_place{it}", Tq, "slow", tip_margin=0.0)
        self.lift_to(f"clear_place{it}", SAFE_H)
        return c_tgt, L

    def table_align(self, d, held):
        """User 12:55 strategy: align the drawer ON THE TABLE in front of the slot (pick-and-place, re-measured with a
        wrist look-down after each place) until |t err| <= 5 mm and |yaw| <= 2 deg, then grasp the pull wall square and
        insert straight from there with a ~2 cm lift. Returns (held, n0, z0, yaw2) for the insertion, or None."""
        s = self.s
        s.phase("table_align")
        log = []
        for it in range(3):
            c_tgt, L = self.place_in_front(d, held, it)
            d2 = self.wrist_footprint(dict(d, c=c_tgt - WRIST_SIDE_BIAS))
            if not d2.get("wrist_fp") and not s.dry:
                s.decide("table align: no footprint after placing: grasp from the side-camera estimate")
                P, tz, b2 = self.scene(f"align{it}")
                d2 = self.find_drawer(b2, tz) or d
            q = nt(np.asarray(d2["c"], float))
            t_err = float(q[1] - (T_C + EDGE_TARGET))
            yaw_err = float(wrapd(ang(-np.asarray(d2["p"], float)) - ang(N)))
            n_err = float(q[0] - (OFF - PLACE_GAP - L / 2))
            log.append([it, round(t_err, 4), round(yaw_err, 1), round(n_err, 4)])
            s.summary(table_align=log)
            s.decide(f"table align {it}: t err {t_err * 1000:+.1f} mm, yaw err {yaw_err:+.1f} deg, n err {n_err * 1000:+.1f} mm")
            held = self.grasp(d2)
            if held is None:
                held = self.grasp(d2, refine=False)
            s.require(held is not None, "pull wall not grasped after the table alignment")
            d = d2
            if s.dry or (abs(t_err) <= 0.005 and abs(yaw_err) <= 2.0):
                break
        s.summary(align_before_insert=dict(t_err=log[-1][1], yaw_err=log[-1][2], n_err=log[-1][3], iters=len(log)))
        # straight insertion from here: the tool is over the pull wall, lift only ~2 cm (slot floor lip ~1.5 cm)
        Tn = self.T_now() if not s.dry else held["Tg"].copy()
        yaw2 = wrapd(held["yaw"] + wrapd(ang(N) - ang(held["d0"])))
        q2 = nt(Tn[:2, 3])
        z0 = self.tz + held["drop"] + BOTTOM_H
        T0 = tool_down_T(np.r_[xy_of(q2[0], q2[1]), z0], yaw2)
        T0q = T0.copy(); T0q[2, 3] += 0.04
        self.line("over_insert_start", T0q, "normal")
        with self.low_force(25):
            self.line("down_to_insert_start", T0, "slow")
        self.t_line = float(q2[1])
        return held, float(q2[0]), z0, yaw2

    def ext_insert(self, n0, z0, yaw2, policy):
        """Step-policy insertion (operator 12:40, for the sim policy): obs = tool pose in the slot frame, force change,
        previous action, in-hand offset/yaw estimate, step; policy.act(obs) -> (dn m, dt m, dyaw deg); one short
        model-checked slow line per step under the 65 N cap; stops on the end depth, a jam (> 35 N with < 3 mm twice)
        or the step limit. Returns the same dict as stage_insert."""
        s = self.s
        s.phase("insert_ext")
        n, t, dyaw, z = n0, self.t_line, 0.0, z0
        f0 = self.force(); prev = np.zeros(3); jam = 0; end = "steps"; ev = []
        for k in range(30):
            f = self.force() - f0
            obs = dict(n=n - OFF, t=t - (T_C + EDGE_TARGET), yaw=dyaw, force=np.array([f @ N3, f @ T3, f @ Z3]),
                       prev=prev.copy(), inhand_t=0.0, inhand_yaw=0.0, step=k)
            a = np.asarray(policy.act(obs), float)
            dn, dt, dy = float(np.clip(a[0], -0.005, 0.012)), float(np.clip(a[1], -0.003, 0.003)), float(np.clip(a[2], -2, 2))
            nn = min(n + dn, INS_NMAX); tt_ = t + dt; yy = float(np.clip(dyaw + dy, -10, 10))
            st, res = self.try_step(f"ext{k:02d}", tool_down_T(np.r_[xy_of(nn, tt_), z], yaw2 + yy), 0.008)
            peak = float(res.get("max_force_n", 0.0)) if st == "ok" else 99.0
            if st == "ok":
                adv = nn - n
                if not s.dry:
                    adv = float(nt(self.T_now()[:2, 3])[0]) - n
                    nn = n + adv
                jam = jam + 1 if (peak > 35.0 and adv < 0.003) else 0
                n, t, dyaw = nn, tt_, yy
            prev = np.array([dn, dt, dy])
            ev.append([k, st, round(n - OFF, 4), round(peak, 1)])
            if st != "ok":
                end = st; break
            if jam >= 2:
                end = "jam"; break
            if n >= INS_NMAX - 0.003:
                end = "reached"; break
        s.summary(insert=dict(end=end, steps=len(ev), n_minus_off=round(n - OFF, 4), policy="ext"), ext_events=ev[-8:])
        self.snap("insert_end")
        return dict(end=end, n=n, t=t, z=z, yaw=yaw2 + dyaw)

    def sim_insert(self, n0, z0, yaw_sq, off_est, yaw_est):
        """Insertion by the sim-trained PPO policy (sim/drawer_insert, README 'Calling the policy on the robot'):
        Insertion bookkeeping in the base frame, one short model-checked slow line per decision at the start height,
        force stop lowered to 40 N for these steps, f_peak ~ max_force_n + |f_seg_start - f0|; stops when the policy
        has the tool at the vertical gripper's keep-out limit (INS_NMAX; release + scripted push follow), on 'done'
        (48 decisions) or on a tracking abort."""
        s = self.s
        s.phase("insert_sim")
        sys.path.insert(0, str(config.ROOT))
        from sim.drawer_insert.policy import Policy, Insertion
        pol = Policy(str(config.ROOT / "sim" / "drawer_insert" / "weights" / "policy.npz"))
        Tn = self.T_now() if not s.dry else tool_down_T(np.r_[xy_of(n0, self.t_line), z0], yaw_sq - yaw_est)
        f0 = self.force()
        ins = Insertion(pol, normal=N, offset=OFF, t_centre=0.5 * (SLOT_T[0] + SLOT_T[1]), yaw_straight=yaw_sq,
                        off_est=off_est, yaw_est=yaw_est, tool_xy=Tn[:2, 3], tool_yaw=ang(Tn[:2, 0]), f0=f0)
        end, ev, n_lim = "steps", [], INS_NMAX - OFF
        Tm = Tn
        for k in range(48):
            xy, yaw = ins.next(self.force())
            q = nt(np.asarray(xy, float))
            if q[0] > INS_NMAX:                           # keep-out limit of the vertical gripper
                xy = xy_of(INS_NMAX, q[1])
            f_start = self.force()
            with self.low_force(40):
                st, res = self.try_step(f"sim{k:02d}", tool_down_T(np.r_[xy, z0], yaw), 0.008)
            if st == "abort":
                end = "abort"; ev.append([k, st]); break
            stopped = st in ("force", "refused")
            peak = (float(res.get("max_force_n", 40.0 if st == "force" else 0.0)) + float(np.linalg.norm(f_start - f0)))
            Tm = self.T_now() if not s.dry else tool_down_T(np.r_[xy, z0], yaw)
            done = ins.done(peak, stopped, Tm[:2, 3], ang(Tm[:2, 0]))
            n_now = float(nt(Tm[:2, 3])[0])
            ev.append([k, st, round(n_now - OFF, 4), round(float(ins.cmd[1]), 4), round(float(ins.cmd[2]), 1), round(peak, 1)])
            if n_now >= INS_NMAX - 0.003 and peak < 40.0:
                end = "reached"; break
            if done:
                break
        Tm = self.T_now() if not s.dry else (Tm if ev else Tn)
        qn = nt(Tm[:2, 3])
        s.summary(insert=dict(end=end, steps=len(ev), n_minus_off=round(float(qn[0]) - OFF, 4), policy="sim_ppo"),
                  sim_events=ev[-10:])
        self.snap("insert_end")
        return dict(end=end, n=float(qn[0]), t=float(qn[1]), z=z0, yaw=float(ang(Tm[:2, 0])))

    def low_force(self, f):
        """Context: lower the session's force stop to f N for the moves inside (operator 00:50: final approaches toward
        the drawer, knob or table stop at ~20 N instead of 65 N), restored after."""
        import contextlib
        ex = getattr(self.s, "executor", None)

        @contextlib.contextmanager
        def cm():
            old = getattr(ex, "force_limit", None) if ex is not None else None
            try:
                if ex is not None and getattr(ex, "force_cap", None):
                    ex.force_limit = min(float(f), float(ex.force_cap))
                yield
            finally:
                if ex is not None and getattr(ex, "force_cap", None):
                    ex.force_limit = old if old is not None else ex.force_cap
        return cm()

    def put_down_if_held(self):
        """On any exception: if the gripper still holds the wall, lower to just above the table at the current xy (or
        stay if lower already) and open (operator 00:45: no exception may end an episode with the drawer in the hand)."""
        s = self.s
        try:
            if s.dry or float(s.feedback()["gripper"]) >= HOLD_MAX or float(s.feedback()["gripper"]) < 60:
                return
            T = self.T_now()
            if T[2, 3] > self.tz + 0.13:
                Td = T.copy(); Td[2, 3] = self.tz + 0.12
                try:
                    self.line("put_down", Td, "slow")
                except Exception as e:
                    s.log(f"put down: {str(e)[:100]}")
            s.gripper(0.4, "exception: let go of the drawer")
            Tu = self.T_now(); Tu[2, 3] += 0.05
            try:
                self.line("up_after_letgo", Tu, "slow")
            except Exception as e:
                s.log(f"up after let go: {str(e)[:100]}")
        except Exception as e:
            s.log(f"put_down_if_held failed: {e}")

    def turn_safely(self, T):
        """Rotate to T: straight line, else the rotation in 3 line steps, else a model-checked joint move; False when
        none works (ep 33: 'joint jump' in the cancel_yaw line raised and ended the episode with the drawer held)."""
        s = self.s
        try:
            self.line("cancel_yaw", T, "slow"); return True
        except (RuntimeError, Abort) as e:
            s.log(f"cancel_yaw line: {str(e)[:100]}")
        T0 = self.T_now()
        try:
            from rlpj.robot.plan import slerp_R
            for k in (1, 2, 3):
                Tk = T.copy(); Tk[:3, :3] = slerp_R(T0[:3, :3], T[:3, :3], k / 3.0)
                Tk[:3, 3] = T0[:3, 3] + (T[:3, 3] - T0[:3, 3]) * k / 3.0
                self.line(f"cancel_yaw_{k}", Tk, "slow")
            return True
        except (RuntimeError, Abort) as e:
            s.log(f"cancel_yaw split: {str(e)[:100]}")
        try:
            self.joint_to("cancel_yaw_joint", T, "slow"); return True
        except (RuntimeError, Abort) as e:
            s.log(f"cancel_yaw joint: {str(e)[:100]}")
        return False

    def align(self, n_at, z, yaw, meas=None, tries=3):
        """Put the held drawer's centre line on the slot centre (T_C + EDGE_TARGET, the slot window centre) from the
        side image (its camera-side edge), move, re-measure until within 3 mm (user 00:30: straight, then push)."""
        s = self.s
        for i in range(tries):
            if meas is None:
                meas = self.measure_held(n_at)
            if not (meas.get("ok") and meas.get("t_span")):
                break
            if meas["t_span"] < 0.07:
                # attempts 84-86: t_span 0.048-0.058 (drawer partly hidden by the arm), the edge estimate moved the line
                # 0.9-1.5 cm and every insertion jammed; with t_span >= 0.08 (ep 25, 27) it reached its end. Too little
                # of the drawer seen: keep the tool (grasp centre) on the slot centre instead.
                if i == 0:
                    self.t_line = T_C + EDGE_TARGET
                    self.line("align_t_nominal", tool_down_T(np.r_[xy_of(n_at, self.t_line), z], yaw), "slow")
                    s.log(f"align: t_span {meas['t_span']:.3f} too small to trust, tool on the slot centre")
                break
            # drawer centre (absolute t) = T_C + t_err_meas - t_span/2 + 0.065; t_err is measured against T_C
            c_abs = T_C + meas["t_err"] - meas["t_span"] / 2 + 0.065
            err = (T_C + EDGE_TARGET) - c_abs
            s.log(f"align {i}: drawer centre t {c_abs:+.4f}, target {T_C + EDGE_TARGET:+.4f}, error {err * 1000:+.1f} mm")
            self.align_log = getattr(self, "align_log", []) + [round(err, 4)]
            if abs(err) <= 0.003:
                break
            if n_at > OFF - 0.13:                 # the drawer's back is already in the slot: small lateral steps only
                err = float(np.clip(err, -0.004, 0.004))
            self.t_line = float(np.clip(self.t_line + err, T_C - 0.025, T_C + 0.025))
            self.line(f"align_t{i}", tool_down_T(np.r_[xy_of(n_at, self.t_line), z], yaw), "slow")
            meas = None
        s.summary(align=getattr(self, "align_log", []), t_line=round(self.t_line, 4))

    def wall_angle(self, img, e, d_in, h_top):
        """Wall angle (deg, CCW) from the wrist image at the pre-grasp pose (wallfit.wall_fit), or None."""
        if img is None or self.s.dry:
            return None
        try:
            import wallfit
            r = wallfit.wall_fit(img, calib.pose_to_T(self.s.feedback()["pose"]), e, d_in, self.tz + h_top)
            self.log(f"wall fit: {r}")
            return None if r is None else r.get("angle_deg")
        except Exception as ex:
            self.log(f"wall fit failed: {ex}")
            return None

    def refine_wall(self, img, e, d_in, h_top, half=0.05, max_shift=0.04):    # candidates are tried in grasp()
        """Reset agent's wrist refinement: strongest bright->dark step along d_in at the wall-top height."""
        s = self.s
        if img is None or s.dry:
            return None
        try:
            from rlpj.perception.wrist import project_base_to_wrist
            T = calib.pose_to_T(s.feedback()["pose"])
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
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
                return None
            prof = np.where(np.isnan(prof), np.nanmedian(prof), prof)
            grad = np.correlate(prof, [1, 1, 0, -1, -1], "same"); grad[:3] = grad[-3:] = 0
            k = int(np.argmax(grad)); shift = ss[k] + 0.006
            s.log(f"refine_wall: step {grad[k]:.0f}, shift {shift:+.3f}")
            if grad[k] < 40 or abs(shift) > max_shift:
                return None
            return float(shift)
        except Exception as ex:
            s.log(f"refine_wall failed: {ex}")
            return None

    def finish(self, before, extra_terminal=0.0):
        """Home, verify with the orchestrator's geometric check, terminal reward, save transitions."""
        s = self.s
        s.phase("verify")
        self.go_home("home_verify")
        time.sleep(0.5)
        P, tz, after = self.scene("after")
        ok, rep = checks.drawer_rl(before, after, TASK)
        slot = rep.get("slot", {})
        d_after = self.find_drawer(after, tz)
        # terminal bonus must agree with the real verdict (geometry AND vision judge; backwards = failure, user 01:50):
        # geometry ok -> ask the orchestrator's vision judge; judged not successful -> -5 (never pay a backwards drawer).
        # learner.py relabels this bonus from the orchestrator's verdict file when there is one.
        bonus, judge = 0.0, None
        if ok is True:
            judge = self.judge()
            bonus = -5.0 if (judge or {}).get("success") is False else 10.0
        R = extra_terminal + bonus + 2.0 * slot.get("flush", 0.0) - 2.0 * slot.get("front", 0.0) - 2.0 * slot.get("behind", 0.0) - (1.0 if d_after else 0.0)
        self.terminal = dict(bonus=bonus, R=round(R, 3), judge=judge)
        if self.trans:
            self.trans[-1]["rew"] += R
        self.ret += R if self.trans else 0.0
        self.save_transitions(ok, rep)
        s.summary(verify=dict(check=ok, judge=judge, bonus=bonus, reason=rep.get("reason"), slot=slot, cabinet_moved_m=rep.get("cabinet_moved_m"),
                              drawer_on_table=bool(d_after), terminal_reward=round(R, 3)),
                  transitions=len(self.trans), **{"return": round(self.ret, 3)})
        if ok is not True:
            s.summary(first_failed_check=f"geometric check {ok}: {rep.get('reason')}")
        s.claim("success" if ok is True and bonus > 0 else "fail")

    def judge(self):
        """The orchestrator's vision judge on before/after side images and the after wrist image (task success text
        includes the orientation). Returns its dict or None."""
        s = self.s
        try:
            from rlpj.success import vlm_judge
            imgs = {"before_side": os.path.join(s.A, "side_before.jpg"), "after_side": os.path.join(s.A, "side_after.jpg")}
            w = s.wrist.latest()[0]
            if w is not None:
                self.save("wrist_after", w); imgs["after_wrist"] = os.path.join(s.A, "wrist_after.jpg")
            r = vlm_judge.judge({k: v for k, v in imgs.items() if os.path.exists(v)}, TASK, timeout=150)
            out = {k: r.get(k) for k in ("success", "confidence", "reason")}
            s.log(f"vision judge: {out}")
            return out
        except Exception as ex:
            s.log(f"vision judge failed: {ex}")
            return None

    def save_transitions(self, ok=None, rep=None):
        if getattr(self, "_saved", False):
            return
        self._saved = True
        for i, tr in enumerate(self.trans):
            last = i == len(self.trans) - 1
            tr["next_obs"] = tr["obs"] if last else self.trans[i + 1]["obs"]
            tr["done"] = 1.0 if last else 0.0
        name = f"{self.mode}_ep{self.ep:04d}" if self.ep >= 0 else f"{self.mode}_test_{time.strftime('%Y%m%d_%H%M%S')}"
        if self.s.dry:
            name = "dry_" + name
        p = rc.save_episode(name, self.trans, dict(mode=self.mode, episode=self.ep, dry=self.s.dry, ret=self.ret, check=ok, terminal=getattr(self, "terminal", None),
                                                   reason=(rep or {}).get("reason"), events=self.events, policy=self.actor.info,
                                                   attempt=self.s.A, t=time.time()))
        self.s.summary(transitions_file=os.path.relpath(p, rc.rl_dir()), transitions=len(self.trans), **{"return": round(self.ret, 3)})

    def release(self, ins):
        """Open the gripper on the front wall and make sure it let go before lifting (attempt 16: commanded 0.75,
        reading stayed 86.9 with the drawer wedged; the lift hit 58 N). Widest opening that keeps the inner finger
        >= 12 mm in front of the yaml front plane (attempt 5: 0.40 at the end put it under the rail); between tries
        the tool shifts 2 mm to free a wedged wall; if it still holds, end without lifting."""
        s = self.s
        n_axis = float(nt(self.T_now()[:2, 3])[0]) if not s.dry else ins["n"]
        room = OFF - 0.012 - n_axis - 0.008                     # inner fingertip half-opening allowed (m)
        frac = float(np.clip(1 - 2 * room / 0.085, 0.40, RELEASE))
        self.release_frac = frac
        shifts = [np.zeros(3), np.r_[0.002 * N, 0.0], np.r_[0.002 * TT, 0.0], np.r_[-0.004 * TT, 0.0]]
        for i, sh in enumerate(shifts):
            if i:
                T = self.T_now(); T[:3, 3] += sh
                try:
                    self.line(f"free{i}", T, "slow")
                except Abort as e:
                    s.log(f"free {i}: {str(e)[:100]}")
            g = s.gripper(frac, f"release the front wall (opening {frac:.2f}, inner finger room {room * 1000:.0f} mm)")
            if s.dry or g < 100 * frac + 4:
                s.summary(release=dict(frac=round(frac, 2), reading=round(g, 1), tries=i + 1))
                return
        s.summary(release=dict(frac=round(frac, 2), reading=round(g, 1), tries=len(shifts), stuck=True))
        self.set_home(self.q_now())          # the session must not lift on the way out either: stay here
        s.require(False, f"gripper does not let go of the wall (reading {g:.1f} after opening to {frac:.2f}): not lifting")

    def pull_check(self, d, img):
        """The reset guarantees the pull faces the robot (orchestrator check, 00:20). Only if my side-image detector
        positively sees the pull on another wall: no insertion, the episode ends as a failure (operator)."""
        s = self.s
        try:
            c = d["c"] + d["p"] * d["half"] - d["p"] * L_FACE / 2
            r = PF.detect_pull(img, PF.uv_of(np.r_[c, self.tz + 0.04]))
            lr = PF.visible_lr(PF.faces_of(c, -d["p"], L_FACE, L_FACE))
            face = (lr[0] if r.get("side") == "left" else lr[1]) if r.get("found") else None
            cv2.imwrite(os.path.join(s.A, "orient_table.jpg"), PF.crop_around(img, np.r_[c, self.tz + 0.04]))
        except Exception as ex:
            s.log(f"pull check failed: {ex}"); return True
        s.summary(pull_check=dict(face=face, lr=lr, **{k: r.get(k) for k in ("found", "area", "side")}))
        if face is not None and face != "A":
            s.decide(f"pull seen on wall {face}, not on the robot-facing wall: no insertion")
            s.summary(first_failed_check=f"pull on wall {face}, not facing the robot", verify=dict(check=None, pull_face=face))
            self.save_transitions()
            s.claim("fail")
            return False
        return True

    def orient(self, d, img_table, held):
        """Pull wall (grey metal label frame) must face out of the slot (user, 23:50). Score the walls in the side
        image on the table (A = grasped, robot-facing wall) and, when that is not conclusive, with the held drawer
        turned 180 (then 90, 270) deg at SAFE_H. If the pull is not A: put the drawer down turned so the pull faces
        the robot, re-grasp the robot-facing wall. Returns the held dict for the carry."""
        s = self.s
        s.phase("orient")
        e, yaw, d0 = held["e"], held["yaw"], held["d0"]
        c_pick = e + d0 * (L_FACE / 2 - 0.006)
        views, votes = [], {}

        def look(img, ch, dn, zb, tag, exclude):
            try:
                uvc = PF.uv_of(np.r_[ch, zb + 0.04])
                r = PF.detect_pull(img, uvc)
                lr = PF.visible_lr(PF.faces_of(ch, dn, L_FACE, L_FACE))
                face = (lr[0] if r.get("side") == "left" else lr[1]) if r.get("found") else None
                if face in exclude:
                    face = None
                if face:
                    votes[face] = votes.get(face, 0) + 1
                views.append(dict(tag=tag, lr=lr, face=face, **{k: r.get(k) for k in ("found", "area", "side", "u", "corner_u")}))
                cv2.imwrite(os.path.join(s.A, f"orient_{tag}.jpg"), PF.crop_around(img, np.r_[ch, zb + 0.04]))
            except Exception as ex:
                s.log(f"orient view {tag} failed: {ex}")

        look(img_table, c_pick, d0, self.tz, "table", ())
        turned = 0
        for rot in (180, 90, 270):
            if votes or s.dry:
                break
            self.joint_to(f"turn{rot}", tool_down_T(np.r_[e, self.tz + SAFE_H], yaw + rot))
            turned = rot
            time.sleep(0.6)
            dn = rot2(d0, rot)
            look(s.side.latest()[0], e + dn * (L_FACE / 2 - 0.006), dn, self.tz + SAFE_H - held["drop"], f"rot{rot}", ("A",))
        pull = max(votes, key=votes.get) if votes else None
        best = votes
        info = dict(pull=pull, best=best, views=views, turned=turned)
        s.summary(orient=info)
        s.decide(f"orientation: pull wall {pull} (A = grasped; best grey {best})")
        if pull in (None, "A"):
            self.orient_state = "A" if pull == "A" else "unknown"
            if turned:
                self.joint_to("turn_back", tool_down_T(np.r_[e, self.tz + SAFE_H], yaw))
            return held
        # put the drawer down turned so that the pull faces the robot (-N), centre where it was
        n_p = PF.faces_of(c_pick, d0, L_FACE, L_FACE)[pull][1]
        phi = wrapd(ang(-N) - ang(n_p))
        dnew = rot2(d0, phi)
        xy = c_pick - dnew * (L_FACE / 2 - 0.006)
        Tp = tool_down_T(np.r_[xy, held["Tg"][2, 3] + 0.008], yaw + phi)
        Thi = Tp.copy(); Thi[2, 3] = self.tz + SAFE_H
        Tpre = Tp.copy(); Tpre[2, 3] += 0.05
        s.decide(f"turn the drawer by {phi:+.0f} deg so the pull ({pull}) faces the robot; re-grasp")
        self.joint_to("above_turned_place", Thi)
        self.line("down_turned_pre", Tpre, "normal")
        self.line("place_turned", Tp, "slow", tip_margin=0.0)
        s.gripper(0.4, "release the turned drawer")
        self.line("up_turned", Tpre, "slow", tip_margin=0.0)
        self.lift_to("clear_turned", SAFE_H)
        self.go_home("home_regrasp")
        P, tz, b2 = self.scene("turned")
        d2 = self.find_drawer(b2, tz)
        s.require(d2 is not None, "drawer not seen after turning it")
        img2 = s.side.latest()[0]
        held2 = self.grasp(d2)
        s.require(held2 is not None, "pull wall not grasped after turning")
        try:
            c2 = held2["e"] + held2["d0"] * (L_FACE / 2 - 0.006)
            sc2 = PF.score_faces(img2, PF.faces_of(c2, held2["d0"], L_FACE, L_FACE), self.tz)
            cv2.imwrite(os.path.join(s.A, "orient_after_turn.jpg"), PF.crop_around(img2, np.r_[c2, self.tz + 0.04]))
            s.summary(orient=dict(info, after_turn=sc2))
        except Exception as ex:
            s.log(f"orient after-turn scoring failed: {ex}")
        self.orient_state = "turned"
        return held2

    def run(self):
        s = self.s
        s.summary(rl_mode=self.mode, episode=self.ep, policy=self.actor.info, transitions=0, **{"return": 0.0})
        s.phase("perceive")
        if np.abs(wrap180(self.q_now() - HOME_Q)).max() > 3:
            g0 = float(s.feedback()["gripper"])
            if 3.0 < g0 < 97.0 and self.T_now()[2, 3] < self.tz + 0.15:
                s.gripper(0.4, "start: let go first")
            self.escape()
            self.go_home("home_start")
        self.set_home()
        P, tz, before = self.scene("before")
        d = self.find_drawer(before, tz)
        prof0 = checks.slot_profile(P, tz, CAB, FLUSH_TOL)
        s.summary(objects=dict(drawer=None if d is None else dict(xy=d["xy"], rect_c=np.round(d["c"], 4).tolist(), size=d["size"],
                                                                  h_top=round(d["h_top"], 3), front_dev_deg=d["front_dev_deg"], shift_w=d.get("shift_w"))),
                  slot_before=prof0)
        s.require(d is not None, "no drawer on the table")
        s.view.set_scene(objects=[dict(xy=list(map(float, d["c"])), z_top=self.tz + d["h_top"], r=d["half"], label="drawer")])
        P_img = s.side.latest()[0]
        try:                                        # lighting (04:30 the room lights went off: brightness 122 -> 76)
            s.summary(side_brightness=round(float(cv2.cvtColor(P_img, cv2.COLOR_BGR2HSV)[..., 2].mean()), 1))
        except Exception:
            pass
        if not self.pull_check(d, P_img):
            return
        s.phase("grasp")
        d = self.wrist_footprint(d)
        held = self.grasp(d)
        if held is None:
            P, tz, b2 = self.scene("retry")
            d = self.find_drawer(b2, tz)
            s.require(d is not None, "drawer not seen after a missed grasp")
            held = self.grasp(d, refine=False)           # after a refined miss: the unrefined perception point
        s.require(held is not None, "front wall not grasped after two tries")
        strategy = os.environ.get("RLPJ_STRATEGY") or ("table_align" if (self.ep >= 0 and (self.ep // 2) % 2 == 0) else "carry")
        ins_arm = os.environ.get("RLPJ_INSERT_ARM") or ("sim" if (self.ep < 0 or self.ep % 2 == 0) else "own")
        if not (config.ROOT / "sim" / "drawer_insert" / "weights" / "policy.npz").exists():
            ins_arm = "own"
        self.ins_arm = ins_arm
        s.summary(insert_arm=ins_arm)
        if strategy == "table_align" and not held.get("wrist_fp") and not s.dry:
            strategy = "carry"                          # the table alignment needs the wrist footprint
        s.summary(strategy=strategy)
        if strategy == "table_align":
            held, n0, z0, yaw2 = self.table_align(d, held)
            meas = dict(ok=False, n=0, t_err=0.0, yaw_err=0.0)
            s.summary(held=dict(n0_minus_off=round(n0 - OFF, 4), z0_above_table=round(z0 - self.tz, 4), yaw2=round(yaw2, 1), t_line=round(self.t_line, 4)))
            ext = self.ext_policy()
            if self.ins_arm == "sim":
                ins = self.sim_insert(n0, z0, yaw2, -float(held.get("fp_off") or 0.0), 0.0)
            elif ext is not None:
                ins = self.ext_insert(n0, z0, yaw2, ext)
            else:
                ins = self.stage_insert(n0, z0, yaw2, meas)
            self.after_insert(ins, yaw2, before)
            return
        s.phase("carry")
        # straight drawer (user 00:30): the drawer sits ~7 deg turned in the jaws (insertions got through in 5/12
        # episodes with the tool at +5..+10 deg, 1/14 otherwise, and jam again at 0, ep 24): the tool is turned by
        # HAND_YAW for the whole insertion so the DRAWER's axis is on the slot normal; no yaw change during it
        # HAND_YAW was inferred from side-camera grasps (their yaw was off); with the wrist footprint the jaws sit on the
        # wall's true direction, so the drawer is straight in the hand (operator 04:50: no side-camera in-hand estimates)
        yaw2 = wrapd(held["yaw"] + wrapd(ang(N) - ang(held["d0"]))) + (0.0 if held.get("wrist_fp") else HAND_YAW)
        z0 = self.tz + held["drop"] + BOTTOM_H
        n0 = min(OFF - BACK_GAP - held["L_back"], INS_NMAX - 0.05)
        T0 = tool_down_T(np.r_[xy_of(n0, T_C), z0], yaw2)
        T0h = T0.copy(); T0h[2, 3] = self.tz + SAFE_H
        T0b = T0h.copy(); T0b[:2, 3] -= 0.07 * N
        self.joint_to("above_slot_front_back", T0b)
        self.line("forward_above_slot", T0h, "normal")
        T0m = T0.copy(); T0m[2, 3] += 0.04
        self.line("down_in_front_of_slot", T0m, "normal")
        with self.low_force(25):
            self.line("down_to_insert_start", T0, "slow")
        g = float(s.feedback()["gripper"])
        s.require(g < HOLD_MAX or s.dry, f"drawer lost while carrying (gripper {g:.1f})")
        meas = self.measure_held(n0)
        # scripted lateral alignment (00:40): the drawer's -t edge (the side facing the camera, well measured) puts
        # its centre at t_err - t_span/2 + 0.065 (drawer ~0.13 wide); insertions reached their end in 11/21 episodes
        # while that edge estimate was -0.001..+0.011 (attempts 2-46), and in 0/16 since it moved to -0.005..-0.016
        # (attempts 48-71). Shift the insertion line so the estimate lands at EDGE_TARGET.
        self.t_line = T_C
        # operator 00:22: ep 32 went in with the measured in-hand yaw +7.9 deg uncorrected and wedged skewed: cancel the
        # measured yaw (enough points only, at most 8 deg) by turning the hand before the insertion
        turned = False
        if not held.get("wrist_fp") and meas.get("ok") and meas.get("n", 0) >= 600 and abs(meas.get("yaw_err", 0.0)) > 1.0:
            dy = float(np.clip(meas["yaw_err"], -8.0, 8.0))
            yaw2 = yaw2 - dy
            T_c = tool_down_T(np.r_[xy_of(n0, T_C), z0], yaw2)
            turned = self.turn_safely(T_c)
            if turned:
                s.summary(yaw_cancel=round(-dy, 1))
            else:
                yaw2 = yaw2 + dy                         # not turned: keep the old yaw
                s.summary(yaw_cancel=None, yaw_cancel_skipped=round(-dy, 1))
        if held.get("wrist_fp") and held.get("fp_off") is not None:
            # operator 04:50: put the DRAWER centre on the slot centre from the wrist footprint (the side camera reads
            # the held drawer 5-7 cm off): tool t = slot centre + (tool - drawer centre along the wall)
            self.t_line = T_C + EDGE_TARGET + float(np.clip(held["fp_off"], -0.03, 0.03))
            self.line("align_t_fp", tool_down_T(np.r_[xy_of(n0, self.t_line), z0], yaw2), "slow")
            s.summary(t_line=round(self.t_line, 4), fp_off=round(held["fp_off"], 4))
        else:
            self.align(n0, z0, yaw2, None if turned else meas)   # re-measure after turning
        s.summary(held=dict(n0_minus_off=round(n0 - OFF, 4), z0_above_table=round(z0 - self.tz, 4), yaw2=round(yaw2, 1), meas=meas, t_line=round(self.t_line, 4)))
        s.decide(f"held drawer: t err {meas['t_err']:+.3f} yaw err {meas['yaw_err']:+.1f} ({meas['n']} pts); insertion from n-OFF {n0 - OFF:+.3f}")
        ext = self.ext_policy() if os.environ.get("RLPJ_EXT_ALL") else None
        if self.ins_arm == "sim":
            yaw_sq = wrapd(held["yaw"] + wrapd(ang(N) - ang(held["d0"])))
            off_est = -float(held.get("fp_off") or 0.0) if held.get("wrist_fp") else 0.0
            yaw_est = 0.0 if held.get("wrist_fp") else -HAND_YAW
            ins = self.sim_insert(n0, z0, yaw_sq, off_est, yaw_est)
        elif ext is not None:
            ins = self.ext_insert(n0, z0, yaw2, ext)
        else:
            ins = self.stage_insert(n0, z0, yaw2, meas)
        self.after_insert(ins, yaw2, before)

    def ext_policy(self):
        """The sim worker's step policy (sim/drawer_insert/policy.py, numpy Policy(path).act(obs)) when RLPJ_EXT_POLICY
        names it, 'scripted' for a straight-push test policy, else None (my learned insertion)."""
        name = os.environ.get("RLPJ_EXT_POLICY")
        if not name:
            return None
        if name == "scripted":
            class Straight:
                def act(self, obs):
                    return (0.010, 0.0, 0.0)
            return Straight()
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("ext_policy", name if name.endswith(".py") else os.path.join(name, "policy.py"))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            return m.Policy(os.path.dirname(name) if name.endswith(".py") else name)
        except Exception as e:
            self.s.log(f"external policy not loaded ({e}): own insertion")
            return None

    def after_insert(self, ins, yaw2, before):
        s = self.s
        s.phase("release")
        self.release(ins)
        # back off along -N first (the outer finger moves away from the wall, the open inner finger has ~18 mm to the
        # wall's inner face), then up from the measured pose (attempt 17: the lift from a load-deflected pose came
        # within 6 mm of the cabinet_front box and was refused), then further back, lift
        T_b = self.T_now() if not s.dry else tool_down_T(np.r_[xy_of(ins["n"], ins["t"]), ins["z"]], ins["yaw"])
        gap = 0.085 * (1 - getattr(self, "release_frac", 0.62)) / 2 - 0.005     # inner finger to the wall's inner face
        T_b[:2, 3] -= float(np.clip(gap - 0.003, 0.006, 0.012)) * N
        self.line("back_off_wall", T_b, "slow")
        T_up = T_b.copy(); T_up[2, 3] += 0.06
        self.line("up_from_front", T_up, "slow")
        T_bk = T_up.copy(); T_bk[:2, 3] -= 0.03 * N
        self.line("back_from_front", T_bk, "normal")
        self.lift_to("lift_clear", SAFE_H, "normal")
        face_n = ins["n"] - 0.008          # outer face of the front wall in front of the tool axis
        if ins["end"] == "dropped":
            s.decide("insertion ended with the drawer dropped: no push")
            self.finish(before); return
        # ep 26: the insertion stopped on force at n-OFF -0.056 and the drawer ended nearly flush in the side view
        # (flush 0.43, median_d 0.0) but the push was skipped: a jammed insertion still gets the push when reachable
        if face_n - 0.012 < OFF - 0.088:          # pusher reach (tilt 40: pre-pose checked to n-OFF -0.08 offline)
            s.decide(f"drawer front at n-OFF {face_n - OFF:+.3f}: too far out for the tilted push (j3 limit), no push")
            self.finish(before); return
        self.to_ready()
        # push beside the pull (knob ~4 cm high, label frame above it, both centred): 3 cm toward +t at 5 cm height
        # (attempt 20 pushed at 7.4 cm: the tip went over the wall top and the retreat dragged the drawer 2 cm out)
        # push along the slot axis the insertion found (tuning 03:08: insertions only got through with the tool
        # turned +5.6..+10 deg; pushes straight along the yaml normal then jammed after 1-2 cm, 4/4)
        # the drawer centre is no longer under the tool once the insertion line was shifted sideways: it sits where the
        # alignment put it (T_C + EDGE_TARGET) plus the learned dt (attempt 72: pushing 2.4 cm off-centre jammed after 1.5 cm)
        # once the drawer's back is in the slot, the slot holds it: push on the slot centre line, straight along the
        # normal (the sim arm moves the tool sideways/turns it while searching; following the tool pushed the fingers
        # into the slot-side keep-out offline)
        c_t = T_C + EDGE_TARGET
        PUSH_FRAME.update(yaw=0.0, n0=face_n, t0=float(c_t))
        s.summary(push_frame=dict(PUSH_FRAME, drawer_t=round(c_t, 4)))
        self.stage_push(face_n, min(c_t + PUSH_DT, PUSH_TMAX), ins["end"] == "reached")
        self.finish(before)


def main():
    with RobotSession(program="execute") as s:
        ep = Episode(s)
        try:
            ep.run()
        except BaseException:
            ep.put_down_if_held()
            raise
        finally:
            try:
                ep.save_transitions()     # also when aborted (no terminal reward then)
            except Exception as e:
                s.log(f"saving transitions failed: {e}")


if __name__ == "__main__":
    main()
