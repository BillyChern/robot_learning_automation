"""Scene setup (not part of the reset contract): put the drawer from the table back into the bottom slot, flush, so
reset.py can be tested from the done state without an operator.
1. find the pull wall (side depth: the pull tab sticks out of the rim rectangle), grasp it top-down (wrist-refined),
2. carry, align with the slot (pull facing the robot), slide in until it sticks out ~INSERT_OUT, release, lift,
3. pinch the pull tab (the grasp reset.py uses, shifted out by the measured protrusion) and push it flush."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj.perception import tabletop as tt
from rlpj.robot.plan import tool_down_T
from rlpj.robot.session import RobotSession, Abort
import reset as R
from scene import N, TT, OFF, nt, xy_of

HINT = None          # --pull_hint dx,dy: base-frame direction the pull face points to (one-off setup)
T_C = -0.087          # drawer centre line along the front (pull triangulated at -0.088)
INSERT_OUT = 0.065    # front wall this far in front of the yaml front plane after the slide (as after a pull)
BOTTOM_H = 0.021      # drawer bottom height while sliding in (slot floor ~0.015)


def pull_side(s, r):
    """(rim centre, unit axis toward the pull wall, half-length along it, rim top height) or raises Abort."""
    P = tt.cloud(s.side, n=7)
    tz = tt.table_fit(P)
    blobs = tt.detect(P, table_z=tz)
    cand = [b for b in blobs if nt(b["xy"])[0] < OFF - 0.03 and 0.03 <= b["height"] <= 0.12 and b["n"] > 150]
    s.require(cand, "insert: no drawer on the table")
    b = max(cand, key=lambda b: b["n"])
    pts = b["points"]; h = pts[:, 2] - tz; h_top = float(np.percentile(h, 98))
    top = pts[h > h_top - 0.012, :2]
    (cx, cy), (w0, h0), ang = cv2.minAreaRect(np.ascontiguousarray(top.astype(np.float32)))
    c = np.array([cx, cy]); u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); v = np.array([-u[1], u[0]])
    # the pull wall carries a metal label plate + tab: of the faces the side camera sees, it is the least saturated
    # one around its centre at plate height (wood is saturated orange-brown)
    from rlpj.robot import calib
    img = s.side.latest()[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    K, dist = calib.side_intrinsics(); Tbs = calib.T_base_side(); Tsb = np.linalg.inv(Tbs); cam = Tbs[:3, 3]
    sides = []
    for vec, half, ohalf in ((u, w0 / 2, h0 / 2), (-u, w0 / 2, h0 / 2), (v, h0 / 2, w0 / 2), (-v, h0 / 2, w0 / 2)):
        perp = np.array([-vec[1], vec[0]])
        mid = c + vec * (half + 0.004)
        vis = float(np.r_[vec, 0] @ (cam - np.r_[mid, tz + 0.035]))
        P = np.array([np.r_[mid + a * perp, tz + hh] for a in np.linspace(-0.025, 0.025, 11) for hh in np.linspace(0.02, 0.05, 7)])
        uv = calib.project(calib.apply(Tsb, P), K, dist).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < hsv.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hsv.shape[0])
        sat = float(np.median(hsv[uv[ok, 1], uv[ok, 0], 1])) if ok.sum() > 10 else 999.0
        sides.append(dict(sat=sat, vis=vis, vec=vec, half=half))
    seen = [d for d in sides if d["vis"] > 0.05]
    seen.sort(key=lambda d: d["sat"])
    s.summary(insert_rim=dict(c=np.round(c, 4).tolist(), size=[round(w0, 3), round(h0, 3)], ang=round(ang, 1), h_top=round(h_top, 3),
                              faces=[[round(d["sat"], 0), round(d["vis"], 3)] for d in sides]))
    s.log(f"insert: rim {np.round(c, 3).tolist()} {w0:.3f}x{h0:.3f} ang {ang:.0f}, face saturation/visibility {[[round(d['sat']), round(d['vis'], 2)] for d in sides]}")
    if HINT is not None:
        seen.sort(key=lambda d: -float(d["vec"] @ HINT))
        s.decide(f"insert: pull face chosen by the hint {HINT.round(2).tolist()} (seen in attempt 23's side image): {seen[0]['vec'].round(2).tolist()}")
    else:
        s.require(len(seen) >= 2 and seen[0]["sat"] < 0.75 * seen[1]["sat"], "insert: pull face not clear in the side image")
    for i, d in enumerate(sides):
        P = np.r_[c + d["vec"] * (d["half"] + 0.004), tz + 0.035]
        u0 = calib.project(calib.apply(Tsb, P[None]), K, dist)[0].astype(int)
        cv2.circle(img, tuple(u0), 8, (0, 255, 0) if d is seen[0] else (0, 0, 255), 2)
    cv2.imwrite(os.path.join(s.A, "insert_faces.jpg"), cv2.resize(img, (640, 360)))
    sides = [(0, seen[0]["vec"], seen[0]["half"] + 0.005)]
    return c, sides[0][1], sides[0][2], h_top


def main():
    global HINT
    if "--pull_hint" in sys.argv:
        HINT = R.unit(np.array([float(v) for v in sys.argv[sys.argv.index("--pull_hint") + 1].split(",")]))
    with RobotSession(program="reset") as s:
        r = R.Reset(s, {})
        s.phase("perceive")
        if np.abs(R.wrap180(r.q_now() - R.HOME_Q)).max() > 3:
            r.escape(); r.go_home("home_start")
        c, p, half, h_top = pull_side(s, r)
        D = 2 * half - 0.01                      # drawer depth estimate (rim box, minus bloat)
        e = c + p * (half - 0.006)               # front wall centre
        d0 = -p
        yaw = R.ang(p); yaw = yaw - 180 if yaw > 90 else yaw + 180 if yaw < -90 else yaw
        tip_h = max(h_top - 0.014, 0.018)
        Tg = tool_down_T(np.r_[e, r.tz + tip_h + R.TIP_OPEN], yaw)
        T_pre = Tg.copy(); T_pre[2, 3] += 0.05
        T_hi = Tg.copy(); T_hi[2, 3] = r.tz + R.SAFE_H
        s.decide(f"insert: pull wall at {np.round(e, 3).tolist()}, depth est {D:.3f}")
        s.phase("grasp_front")
        s.gripper(0.4, "open")
        r.to_ready()
        r.joint_to("above_front", T_hi)
        r.line("down_to_front_pre", T_pre, "normal")
        time.sleep(0.4)
        sh = r.refine_wall(s.wrist.latest()[0], e, d0, h_top)
        if sh is not None:
            e = e + sh * d0; Tg[:2, 3] = e; T_pre[:2, 3] = e
            r.line("front_correct", T_pre, "slow")
        r.line("down_to_front", Tg, "slow")
        g = s.gripper(1.0, "grasp the front wall")
        s.require(g < R.HOLD_MAX or s.dry, f"insert: front wall not held ({g:.1f})")
        r.line("lift_slow", T_pre, "slow")
        r.lift_to("lift", R.SAFE_H, "normal")
        drop = tip_h + R.TIP_OPEN                # tool height above the drawer bottom
        s.phase("slide_in")
        yaw2 = R.wrapd(yaw + R.wrapd(R.ang(N) - R.ang(d0)))    # drawer axis (front -> back) along +N
        h_tool = r.tz + drop + BOTTOM_H
        n_start = OFF - 0.03 - D                 # back of the drawer 3 cm in front of the front plane
        n_end = OFF - INSERT_OUT
        P0 = np.r_[xy_of(n_start, T_C), h_tool]; P1 = np.r_[xy_of(n_end, T_C), h_tool]
        T0 = tool_down_T(P0, yaw2); T0h = T0.copy(); T0h[2, 3] = r.tz + R.SAFE_H
        r.joint_to("above_slot_front", T0h)
        r.line("down_in_front_of_slot", T0, "slow")
        r.line("slide_in", tool_down_T(P1, yaw2), "slow")
        s.gripper(0.4, "release the front wall")
        T1u = tool_down_T(P1, yaw2); T1u[2, 3] += 0.06
        r.line("up_from_front", T1u, "slow")
        r.go_home("home_mid")
        p2 = r.perceive("after_slide")
        s.require(p2["out"] is not None and p2["protr"] > 0.03, f"insert: drawer not seen in the slot after the slide ({p2['protr']:.3f})")
        s.phase("push_flush")
        shift = float(p2["protr"])
        R.PIN_OUT, R.PIN_H = R.PIN_BASE
        r.pin_t = R.PIN_T
        r.pin_dy, r.pin_dz = 0.002, 0.0015       # the offset that held the tab in attempt 17
        Tg_done = r.pin_T(R.PIN_OUT)
        Tg_out = Tg_done.copy(); Tg_out[:3, 3] -= np.r_[shift * N, 0.0]
        z = Tg_done[:3, 2]
        Tpre_out = Tg_out.copy(); Tpre_out[:3, 3] -= 0.05 * z
        Thi_out = Tpre_out.copy(); Thi_out[2, 3] = r.tz + R.PIN_HI
        s.decide(f"insert: drawer sticks out {shift:.3f} m; pinch the tab there and push it flush")
        r.to_ready()
        s.gripper(R.PRECLOSE, "pre-close")
        q = r.ik(Thi_out, [np.array([0, 40, -120, 0, j5, j6], float) for j5 in (20.0, 40.0, 60.0) for j6 in (-90.0, 90.0)])
        s.move_joint("above_tab_out", q, "normal")
        r.line("to_tab_pre", Tpre_out, "normal", tip_margin=R.PIN_MARGIN)
        r.line("approach_tab", Tg_out, "slow", tip_margin=R.PIN_MARGIN)
        g = s.gripper(R.PINCH, "pinch the tab")
        s.summary(insert_tab_grip=round(g, 1))
        s.require(g < R.PINCH * 100 - 2.5 or s.dry, f"insert: tab not held ({g:.1f})")
        s.gripper(min(g / 100 + 0.008, 0.90), "keep squeezing")
        r.line("push_flush", Tg_done, "slow", tip_margin=R.PIN_MARGIN)
        s.gripper(R.PRECLOSE, "release the tab")
        Tpre_done = Tg_done.copy(); Tpre_done[:3, 3] -= 0.05 * z
        r.line("back_from_tab", Tpre_done, "slow", tip_margin=R.PIN_MARGIN)
        Thi_done = Tpre_done.copy(); Thi_done[2, 3] = r.tz + R.PIN_HI; Thi_done[:2, 3] -= 0.04 * N
        r.line("up_from_tab", Thi_done, "normal")
        r.go_home("home_end")
        p3 = r.perceive("final")
        st = r.state_of(p3)
        s.summary(insert_final=dict(state=st, protr=p3["protr"]))
        s.claim("success" if st == "slot" else "fail")


if __name__ == "__main__":
    main()
