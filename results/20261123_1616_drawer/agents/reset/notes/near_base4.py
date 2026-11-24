"""Offline: joint margins along the knob descent next to the robot base, tool vertical vs tilted away from the base."""
import os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_reset as T
from rlpj.robot.kin import LIMITS_DEG, wrap180
from rlpj.robot.plan import Planner, SLOW
R = T.R; KIN = T.KIN; pl = Planner(KIN, None, log=lambda m: None); tz = R.calib.table_z()
print("limits", LIMITS_DEG.tolist())
knob = np.array([0.2031, 0.0032]); ax = np.array([-0.989, 0.15])
face = R.DRAWER_L
for yaw in (-98.6, 81.4):
    for tilt in (0, 10, 20, 30):
        Tg = R.tool_down_T(np.zeros(3), yaw)
        u = np.r_[knob / np.linalg.norm(knob), 0.0]              # radial direction from the base axis
        k = np.cross([0, 0, 1.0], u); k /= np.linalg.norm(k)       # tilt axis: horizontal, square to the radial direction
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); th = np.radians(tilt)
        Rt = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
        Tg[:3, :3] = Rt @ Tg[:3, :3]
        tip = np.r_[knob, tz + face + R.TIP_CLEAR]
        Tg[:3, 3] = tip - R.TIP_CLOSED * Tg[:3, 2]
        Ta = Tg.copy(); Ta[:3, 3] -= 0.09 * Tg[:3, 2]
        try:
            qa = pl.ik_exact(Ta, np.array([0, -7, -134, 0, -53, yaw]))[0]
            seg = pl.cart_line("d", qa, Tg, SLOW)
        except Exception as e:
            print(yaw, tilt, "IK", str(e)[:50]); continue
        q = seg.q.copy(); q[:, [1, 2, 4]] = wrap180(q[:, [1, 2, 4]])
        lo = (q - LIMITS_DEG[:, 0]).min(0); hi = (LIMITS_DEG[:, 1] - q).min(0)
        m = np.minimum(lo, hi)
        print(f"yaw {yaw:6.1f} tilt {tilt:2d}: flange radius {np.hypot(*(Tg[:2, 3] - 0.12 * Tg[:2, 2])):.3f} joint margins (deg) {np.round(m[[1, 2, 4]], 1).tolist()} q_end {np.round(q[-1], 0).tolist()}")
