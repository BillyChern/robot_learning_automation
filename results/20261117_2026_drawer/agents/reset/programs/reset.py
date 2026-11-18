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
PIN_T = -0.088            # pin t (wrist triangulation, attempts 9-10)
PIN_H = 0.052             # pin height above the table
PIN_OUT = 0.010           # pad centre in front of the front plane
PIN_VARIANTS = [(0.002, 0.029), (0.000, 0.034)]   # (out, h): pin triangulated from 3 wrist views (attempts 9-10): axis h ~0.028, out -0.011..+0.001
PIN_TILT = 30.0           # deg, gripper pointing down and into the cabinet
PIN_PHI = 25.0            # deg roll about the normal
PAD_Z = 0.043             # pad centre along the tool z (closed on the pin)
L_DEFAULT = 0.13          # drawer depth until measured
HOLD_MAX = 97.0           # gripper reading below this = holding something
TOL = 0.05
GOOD = 0.02               # correct when the error is above this
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

    def escape(self):
        """Start away from home and low (a stopped run): back off along -N and up, keeping the orientation."""
        T = self.T_now()
        if T[2, 3] > self.tz + 0.15:
            return
        self.s.decide(f"start pose low ({T[2, 3] - self.tz:.3f} m above the table): back off and up first")
        T2 = T.copy(); T2[:2, 3] -= 0.04 * N; T2[2, 3] = self.tz + 0.16
        self.line("escape_up", T2, "slow")

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
        cv2.imwrite(os.path.join(self.s.A, f"{name}.jpg"), cv2.resize(img, (w, h)))

    # ------------------------------------------------------------ perception
    def perceive(self, tag):
        s = self.s
        P = tt.cloud(s.side, n=5)
        tz = tt.table_fit(P); self.tz_fit = tz
        blobs = tt.detect(P, table_z=tz)
        cand = []
        for b in blobs:
            q = nt(b["xy"]); ext = max(b["extent"])
            if q[0] < OFF - 0.03 and 0.03 <= b["height"] <= 0.12 and 0.06 <= ext <= 0.32 and b["n"] > 150:
                c, (L, W), yaw = obb(b["points"][:, :2])
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
                out = dict(t=float(np.median(q[sel, 1])), h_top=float(np.percentile(h[front & (q[:, 0] < OFF - 0.01)], 98)),
                           h_bot=float(np.percentile(h[sel], 3)), n=int(sel.sum()))
        info = dict(tag=tag, table_z=round(tz, 4), blobs=len(blobs), drawer_blobs=[{k: v for k, v in c.items() if k != "pts"} for c in cand],
                    protr=round(protr, 4), front_pts=int(front.sum()), face_pts=int(face.sum()), inside_pts=int(inside.sum()),
                    out=None if out is None else {k: round(v, 4) for k, v in out.items()})
        self.log(f"perceive {tag}: {info}")
        self.save(f"side_{tag}", s.side.latest()[0])
        return dict(info, cand=cand)

    def state_of(self, p):
        if p["cand"]:
            return "table"
        if p["out"] is not None and p["protr"] >= PROTR_GRASP:
            return "out"
        if p["face_pts"] > 40 and p["face_pts"] > 2 * p["inside_pts"]:
            return "slot"
        return "missing"

    # ------------------------------------------------------------ pin pull
    def pin_T(self, out_m):
        """Tilted grasp pose: pad centre at the pin, jaws along the front (t), pointing down and PIN_TILT into the
        cabinet, then rolled PIN_PHI about the normal (keeps j5 away from the wrist singularity)."""
        th = np.radians(PIN_TILT)
        z = np.r_[np.sin(th) * N, -np.cos(th)]
        x = np.r_[TT, 0.0]
        Rm = rot_axis(np.r_[N, 0.0], PIN_PHI) @ np.stack([x, np.cross(z, x), z], 1)
        pad = np.r_[xy_of(OFF - out_m, self.pin_t), self.tz + PIN_H]
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = pad - PAD_Z * Rm[:, 2]
        return T

    def pull_pin(self, k=0):
        """Grasp the pin (variant k of the pin position guess), pull PULL along -N, release, retreat."""
        global PIN_OUT, PIN_H
        s = self.s
        PIN_OUT, PIN_H = PIN_VARIANTS[k % len(PIN_VARIANTS)]
        self.pin_t = PIN_T
        Tg = self.pin_T(PIN_OUT)
        z = Tg[:3, 2]
        T_pre = Tg.copy(); T_pre[:3, 3] -= 0.06 * z
        T_hi = T_pre.copy(); T_hi[2, 3] = self.tz + PIN_HI; T_hi[:2, 3] -= 0.04 * N
        s.decide(f"pin try {k}: pad {PIN_OUT:.3f} m in front of the face, {PIN_H:.3f} m above the table, t {self.pin_t:.3f}")
        s.gripper(0.0, "open for the pin")
        if np.abs(wrap180(self.q_now() - self.ik(tool_down_T(np.r_[READY_XYZ[:2], self.tz + READY_XYZ[2]], 0.0), [SEED_DOWN]))).max() > 3:
            self.to_ready()
        # tilted pose: seed with j4 = 0 and j5 > 0 (no wrist flip)
        q = self.ik(T_hi, [np.array([0, 40, -120, 0, 20, j6], float) for j6 in (-90.0, 90.0)])
        s.require(wrap180(q[4]) > 15, f"pin pose IK gave a wrist near the singularity or flipped {np.round(q, 0).tolist()}")
        s.move_joint("above_pin", q, "normal")
        self.line("down_to_pin_pre", T_pre, "normal")
        self.save(f"wrist_pin_pre{k}", s.wrist.latest()[0])
        self.line("approach_pin", Tg, "slow")
        self.save(f"wrist_pin_at{k}", s.wrist.latest()[0])
        g = s.gripper(1.0, "grasp the pin")
        s.summary(**{f"pin_grip{k}": round(g, 1)})
        held = g < HOLD_MAX or s.dry
        if not held:
            s.decide(f"pin grasp reading {g:.1f}: nothing held")
            s.gripper(0.0, "open, missed pin")
            d = np.zeros(3)
        else:
            d = -np.r_[PULL * N, 0.0]
            T_out = Tg.copy(); T_out[:3, 3] += d
            self.line("pull_drawer", T_out, "slow")
            s.gripper(0.0, "release the pin")
        T_b = T_pre.copy(); T_b[:3, 3] += d
        self.line("back_from_pin", T_b, "slow")
        T_h = T_hi.copy(); T_h[:3, 3] += d
        self.line("up_from_pin", T_h, "normal")
        self.to_ready()
        return held

    # ------------------------------------------------------------ front-wall grasp and extraction
    def extract(self, p):
        s = self.s
        o = p["out"]
        wall_n = OFF - p["protr"] + 0.006
        h_top = float(np.clip(o["h_top"], 0.05, 0.11))
        tip_h = max(h_top - 0.012, 0.03)
        xy = xy_of(wall_n, o["t"])
        yaw = ang(N)
        Tg = tool_down_T(np.r_[xy, self.tz + tip_h + TIP_OPEN], yaw)
        T_hi = Tg.copy(); T_hi[2, 3] = self.tz + SAFE_H
        T_pre = Tg.copy(); T_pre[2, 3] += 0.05
        s.decide(f"front wall at n-OFF={wall_n - OFF:.3f} t={o['t']:.3f} top {h_top:.3f}: top-down grasp, jaws along the normal")
        s.gripper(0.45, "open to straddle the front wall")
        self.joint_to("above_front_wall", T_hi)
        self.line("down_to_front_pre", T_pre, "normal")
        self.save("wrist_front_pre", s.wrist.latest()[0])
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
        drop = (Tg[2, 3] - self.tz) - max(o["h_bot"], 0.0)
        return dict(yaw_pick=yaw, d0=N.copy(), L=L_DEFAULT, drop=drop)

    # ------------------------------------------------------------ table pick
    def pick_table(self, c):
        s = self.s
        ctr = np.array(c["obb_c"]); u = np.array([np.cos(np.radians(c["yaw"])), np.sin(np.radians(c["yaw"]))])
        L = c["L"]
        ends = [ctr + u * (L / 2 - 0.007), ctr - u * (L / 2 - 0.007)]
        e = min(ends, key=lambda v: np.hypot(*v))
        d0 = unit(ctr - e)
        h_top = float(np.clip(c["h"], 0.03, 0.11))
        tip_h = max(h_top - 0.014, 0.018)
        yaw = ang(d0)
        yaw = yaw - 180 if yaw > 90 else yaw + 180 if yaw < -90 else yaw
        Tg = tool_down_T(np.r_[e, self.tz + tip_h + TIP_OPEN], yaw)
        T_hi = Tg.copy(); T_hi[2, 3] = self.tz + SAFE_H
        T_pre = Tg.copy(); T_pre[2, 3] += 0.05
        s.decide(f"table drawer obb {np.round(ctr, 3).tolist()} L {L:.3f} W {c['W']:.3f} yaw {c['yaw']:.0f}; grasp end wall at {np.round(e, 3).tolist()} top {h_top:.3f}")
        s.gripper(0.4, "open to straddle the end wall")
        self.joint_to("above_drawer_wall", T_hi)
        self.line("down_to_wall_pre", T_pre, "normal")
        self.save("wrist_wall_pre", s.wrist.latest()[0])
        self.line("down_to_wall", Tg, "slow")
        g = s.gripper(1.0, "grasp the end wall")
        s.summary(table_grip=round(g, 1))
        if g >= HOLD_MAX and not s.dry:
            s.decide(f"end wall grasp reading {g:.1f}: nothing held")
            s.gripper(0.4, "open, missed")
            self.line("up_from_wall", T_pre, "slow")
            self.lift_to("lift_empty", SAFE_H, "normal")
            return None
        self.line("lift_drawer_slow", T_pre, "slow")
        self.lift_to("lift_drawer", SAFE_H, "normal")
        return dict(yaw_pick=yaw, d0=d0, L=max(L, 0.08), drop=tip_h + TIP_OPEN)

    # ------------------------------------------------------------ place
    def place(self, held, target, delta=None):
        s = self.s
        if delta is None:
            delta = float(self.rng.uniform(-25, 25))
        a = rot2(N, delta)
        yaw = wrapd(held["yaw_pick"] + wrapd(ang(a) - ang(held["d0"])))
        g = np.asarray(target) - a * (held["L"] / 2 - 0.007)
        h_tool = held["drop"] + 0.008
        Tp = tool_down_T(np.r_[g, self.tz + h_tool], yaw)
        T_hi = Tp.copy(); T_hi[2, 3] = self.tz + SAFE_H
        T_pre = Tp.copy(); T_pre[2, 3] += 0.05
        s.decide(f"place: drawer axis {ang(a):.0f} deg (normal {ang(N):.0f} + {delta:.0f}), grasp point {np.round(g, 3).tolist()}, tool {h_tool:.3f} above table")
        self.joint_to("above_target", T_hi)
        self.line("down_to_place_pre", T_pre, "normal")
        self.line("place_down", Tp, "slow", tip_margin=0.0)
        s.gripper(0.4, "release the drawer")
        self.line("up_from_drawer", T_pre, "slow", tip_margin=0.0)
        self.lift_to("clear_up", SAFE_H, "normal")
        return delta

    # ------------------------------------------------------------ main
    def run(self):
        s = self.s
        tgt = np.asarray(self.targets["drawer"], float)
        s.phase("perceive")
        if np.abs(wrap180(self.q_now() - HOME_Q)).max() > 3:
            self.escape()
            self.go_home("home_start")
        p = self.perceive("start")
        st = self.state_of(p)
        s.summary(start_state=st, start_perception={k: v for k, v in p.items() if k != "cand"})
        s.decide(f"start state: {st}")
        s.require(st != "missing", "drawer not found (no drawer blob on the table and the bottom slot looks empty)")
        held = None
        self.to_ready()
        if st == "slot":
            s.phase("pull_out")
            for k in range(2):
                ok = self.pull_pin(k)
                self.go_home(f"home_after_pull{k}")
                p = self.perceive(f"after_pull{k}")
                st = self.state_of(p)
                if st != "slot":
                    break
                s.decide(f"drawer still in the slot after pull {k} (protrusion {p['protr']:.3f})")
            s.require(st in ("out", "table") or s.dry, f"drawer not pulled out after two pin grasps (protrusion {p['protr']:.3f})")
            if s.dry and st == "slot":
                p = dict(p, out=dict(t=self.pin_t, h_top=0.088, h_bot=0.03, n=0), protr=PULL - 0.002); st = "out"
        delta = None
        if st == "out":
            s.phase("extract")
            for k in range(2):
                held = self.extract(p)
                if held:
                    break
                p = self.perceive(f"retry_front{k}")
                s.require(self.state_of(p) == "out", f"drawer state changed to {self.state_of(p)} after a missed front grasp")
            s.require(held is not None, "front wall not grasped after two tries")
            s.phase("place")
            delta = self.place(held, tgt)
        # table: pick and place; then up to two corrections
        tries = 0
        for it in range(4):
            s.phase(f"check{it}")
            self.go_home("home_for_check")
            p = self.perceive(f"check{it}")
            st = self.state_of(p)
            if st != "table":
                s.require(s.dry and held is not None, f"drawer not seen on the table (state {st})")
                break
            c = p["cand"][0]
            err = float(np.linalg.norm(np.array(c["xy"]) - tgt))
            s.summary(**{f"err_check{it}": round(err, 4)})
            if err <= GOOD or (held is not None and it >= 3):
                break
            s.require(tries < 3, f"drawer still {err:.3f} m from the target after {tries} picks")
            s.phase(f"pick{it}")
            held = self.pick_table(c)
            tries += 1
            if held is None:
                continue
            s.phase(f"place{it}")
            self.place(held, tgt, delta)
        # final measurement (arm at home)
        s.phase("verify")
        self.go_home("home_final")
        p = self.perceive("final")
        st = self.state_of(p)
        s.require(st == "table" or s.dry, f"final: drawer not seen on the table (state {st})")
        if st == "table":
            c = p["cand"][0]
            err = float(np.linalg.norm(np.array(c["xy"]) - tgt))
            yaw_axis = c["yaw"]
            dev = wrapd(yaw_axis - ang(N)); dev = dev - 180 if dev > 90 else dev + 180 if dev < -90 else dev
            ok = err <= TOL and c["h"] <= 0.10
            s.summary(measured={"drawer": c["xy"]}, errors={"drawer": round(err, 4)}, target={"drawer": tgt.tolist()},
                      verify=dict(obb_center=c["obb_c"], L=c["L"], W=c["W"], height=c["h"], yaw_vs_normal_deg=round(dev, 1), tol=TOL))
            s.claim("success" if ok else "fail")
        else:
            s.summary(measured={"drawer": None}, note="dry run: drawer not moved")
            s.claim("fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()
    targets = json.loads(a.targets)
    with RobotSession(program="reset") as s:
        s.summary(targets=targets, measured=None)
        Reset(s, targets).run()


if __name__ == "__main__":
    main()
