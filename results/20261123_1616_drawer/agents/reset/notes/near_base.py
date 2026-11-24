"""Offline: why the pinch at the knob near the robot base (attempts 94, 95) is refused, and which poses would pass."""
import os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_reset as T                      # fake session, keep-out boxes, real reset module as T.R
R, KIN = T.R, T.KIN
from rlpj.robot import sim, calib
from rlpj.robot.plan import Planner, NORMAL, SLOW
pl = Planner(KIN, None, log=lambda m: None)
tz = calib.table_z()
knob = np.array([0.2005, 0.0061]); ax = np.array([-0.988, 0.152])
h_g = R.DRAWER_L + R.TIP_CLOSED + R.TIP_CLEAR
a = np.degrees(np.arctan2(ax[1], ax[0]))
q_look = np.array([341.2, 359.07, 215.52, 358.88, 323.3, 2.94]) - np.array([360, 360, 360, 360, 360, 0]) * [1, 1, 1, 1, 1, 0]
for yaw in (a + 90, a - 90, a + 60, a - 60, a + 120, a - 120):
    Ta = R.tool_down_T(np.r_[knob, tz + h_g + 0.09], yaw); Tg = R.tool_down_T(np.r_[knob, tz + h_g], yaw)
    for seed in (q_look, R.SEED_DOWN, np.array([-20, 40, -120, 0, -60, 0.0])):
        try:
            qa = pl.ik_exact(Ta, seed)[0]
            seg = pl.cart_line("d", qa, Tg, SLOW)
        except Exception as e:
            print(f"yaw {T.R.wrap(yaw):6.1f} seed {seed[:2]}: {str(e)[:60]}"); continue
        ok, rep = sim.check(seg.q, KIN, SLOW, gripper_frac=R.PRE, table_z=tz)
        keys = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items() if k in ("joint_limit_violations", "keep_out_violations", "near_base", "min_tip_above_table", "min_body_above_table", "min_link_z", "max_radius", "keep_in_violations")}
        print(f"yaw {T.R.wrap(yaw):6.1f} q_above {np.round(qa, 0).tolist()} ok {ok} {keys}")
        break
