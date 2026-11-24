"""Offline: the program's grasp_poses from the measured look pose of attempt 97 (knob next to the robot base)."""
import os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_reset as T
R = T.R
w = T.World("table", [0.2197, 0.0006], [-0.989, 0.15])
s = T.Sess(w); s.log = print
s.q = np.array([13.26, 4.61, 221.39, 358.13, 322.54, 215.56])
r = T.Sim(s, np.array([0.276, -0.024]))
Ta, Tg, yaw = r.grasp_poses(np.array([0.2031, 0.0032]), np.array([-0.989, 0.15]), R.DRAWER_L + R.TIP_CLOSED + R.TIP_CLEAR)
print("chosen yaw", round(yaw, 1), "q", np.round(r.grasp_q, 1).tolist())
s.move_joint("above", r.grasp_q); s.move_line("descend", Tg); print("descent executes in the fake session")
