"""Offline: every IK branch for a vertical knob pinch next to the robot base, with joint-limit margins and checks."""
import os, sys, itertools
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_reset as T
from rlpj.robot import sim
from rlpj.robot.kin import LIMITS_DEG, wrap180
from rlpj.robot.plan import Planner, SLOW
R = T.R; KIN = T.KIN; pl = Planner(KIN, None, log=lambda m: None); tz = R.calib.table_z()
knob = np.array([0.192, 0.0282]); ax = np.array([-0.83, 0.558])
h_g = R.DRAWER_L + R.TIP_CLOSED + R.TIP_CLEAR
a = np.degrees(np.arctan2(ax[1], ax[0]))
sols = []
for yaw in (a + 90, a - 90):
    Tg = R.tool_down_T(np.r_[knob, tz + h_g], yaw); Ta = Tg.copy(); Ta[2, 3] += 0.09
    for j1, j2, j3, j5 in itertools.product((0, 90, 180, -90), (-80, -30, 0, 30, 80), (-140, -90, 90, 140), (-90, -45, 45, 90)):
        try:
            q = pl.ik_exact(Ta, np.array([j1, j2, j3, 0, j5, yaw], float))[0]
        except Exception:
            continue
        if any(np.abs(wrap180(q - s[1])).max() < 2 for s in sols):
            continue
        try:
            seg = pl.cart_line("d", q, Tg, SLOW)
        except Exception:
            continue
        qq = seg.q.copy(); qq[:, [1, 2, 4]] = wrap180(qq[:, [1, 2, 4]])
        m = np.minimum(qq - LIMITS_DEG[:, 0], LIMITS_DEG[:, 1] - qq).min(0)[[1, 2, 4]]
        ok, rep = sim.check(seg.q, KIN, SLOW, gripper_frac=R.PRE, table_z=tz)
        sols.append((yaw, q, m, ok, rep.get("min_link_z"), rep.get("near_base")))
for yaw, q, m, ok, lz, nb in sols:
    print(f"yaw {R.wrap(yaw):6.1f} q {np.round(wrap180(q), 0).tolist()} margins j2,j3,j5 {np.round(m, 1).tolist()} check {ok} link_z {lz and round(lz, 3)} near_base {nb}")
