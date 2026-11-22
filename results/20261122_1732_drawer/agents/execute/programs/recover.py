"""Recovery after attempt 5: free the arm from the cabinet front (narrow the gripper, back off along -N, up),
go home, then measure where the cabinet front is now (side depth) vs the task fixture."""
import os
import sys

os.environ.setdefault("RLPJ_TASK", "drawer_ng")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from rlpj.perception import tabletop as tt
from rlpj.robot.kin import wrap180
from rlpj.robot.session import Abort, RobotSession
from rlpj.success import checks
import execute as X


def front_fit(P, tz):
    """Cabinet front face from side depth: points 0.10..0.25 m above the table near the fixture front (|d| < 0.12),
    the nearest surface per 1 cm t-bin; returns offset change (m along N), angle change (deg), t range."""
    q = X.nt(P[:, :2]); h = P[:, 2] - tz
    m = (h > 0.10) & (h < 0.25) & (np.abs(q[:, 0] - X.OFF) < 0.12) & (q[:, 1] > -0.20) & (q[:, 1] < 0.04)
    Q = q[m]
    if len(Q) < 50:
        return dict(n=int(len(Q)))
    pts = []
    for t0 in np.arange(-0.20, 0.04, 0.01):
        s = (Q[:, 1] >= t0) & (Q[:, 1] < t0 + 0.01)
        if s.sum() >= 10:
            pts.append([float(np.percentile(Q[s, 0], 5)), t0 + 0.005, int(s.sum())])
    pts = np.array(pts)
    if len(pts) < 3:
        return dict(n=int(len(Q)), bins=pts.tolist())
    # the face: bins within 3 cm of the nearest-surface median (drop the cabinet sides)
    med = np.median(pts[:, 0]); f = pts[np.abs(pts[:, 0] - med) < 0.03]
    A = np.c_[np.ones(len(f)), f[:, 1]]
    (c0, c1), *_ = np.linalg.lstsq(A, f[:, 0], rcond=None)
    return dict(n=int(len(Q)), d_offset_at_slot=round(float(c0 + c1 * X.T_C - X.OFF), 4), d_angle_deg=round(float(np.degrees(np.arctan(c1))), 2),
                t_range=[round(float(f[:, 1].min()), 3), round(float(f[:, 1].max()), 3)], bins=np.round(pts, 3).tolist())


def main():
    with RobotSession(program="execute") as s:
        ep = X.Episode(s)
        s.phase("free_arm")
        T = ep.T_now()
        g = float(s.feedback()["gripper"])
        s.summary(start_tool=np.round(T[:3, 3], 4).tolist(), start_n_minus_off=round(float(X.nt(T[:2, 3])[0] - X.OFF), 4), start_gripper=round(g, 1))
        if np.abs(wrap180(ep.q_now() - X.HOME_Q)).max() > 3:
            if T[2, 3] < ep.tz + 0.16 and T[2, 2] > -0.95:
                # tilted pusher left at the drawer front (attempt 7): unload, then the push retreat, READY, home
                ep.unload(-X.N3, steps=6)
                T = ep.T_now(); tip = T[:3, 3] + X.PUSH_ZT * T[:3, 2]; q2 = X.nt(tip[:2])
                ep.push_retreat(float(q2[0]), float(q2[1]), float(tip[2] - ep.tz))
            elif T[2, 3] < ep.tz + 0.16:
                ep.unload(-X.N3, steps=3)
                s.gripper(X.RELEASE, "narrow the fingers (inner finger away from the cabinet)")
                T = ep.T_now(); Tb = T.copy(); Tb[:2, 3] -= 0.025 * X.N
                ep.line("back_off", Tb, "slow", tip_margin=0.004)
                Tu = Tb.copy(); Tu[2, 3] = ep.tz + 0.16
                ep.line("up_slow", Tu, "slow", tip_margin=0.004)
                ep.lift_to("up", X.SAFE_H, "normal")
            ep.go_home("home")
        ep.set_home()
        s.phase("measure")
        P, tz, blobs = ep.scene("cabinet")
        ff = front_fit(P, tz)
        prof = checks.slot_profile(P, tz, X.CAB, X.FLUSH_TOL)
        d = ep.find_drawer(blobs, tz)
        cab = [b for b in blobs if 0.22 <= b["height"] <= 0.32]
        s.summary(front_fit=ff, slot=prof, cabinet_blobs=[dict(xy=b["xy"], h=b["height"], ext=b["extent"], n=b["n"]) for b in cab],
                  drawer=None if d is None else dict(xy=d["xy"], size=d["size"]), objects={}, verify=dict(front_fit=ff),
                  transitions=0, **{"return": 0.0})
        s.log(f"front fit {ff}; slot {prof}")
        s.claim("success")


if __name__ == "__main__":
    main()
