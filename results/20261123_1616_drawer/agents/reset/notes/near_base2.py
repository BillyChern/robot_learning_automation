"""Offline: run the program's own grasp_poses from the measured look pose of attempt 95, with the check report."""
import os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_reset as T
from rlpj.robot import sim
R = T.R
orig = sim.check
def check(q, kin, speed, **kw):
    ok, rep = orig(q, kin, speed, **kw)
    if not ok:
        print("  refused:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items() if k != "ok"})
    return ok, rep
sim.check = check
w = T.World("table", [0.217, 0.0036], [-0.988, 0.152])
s = T.Sess(w); s.log = print
s.q = np.array([341.2, 359.07, 215.52, 358.88, 323.3, 2.94])
r = T.Sim(s, np.array([0.276, -0.024]))
print("tool now", np.round(R.calib.pose_to_T(s.feedback()["pose"])[:3, 3], 3).tolist())
try:
    print(r.grasp_poses(np.array([0.2009, 0.0062]), np.array([-0.989, 0.147]), R.DRAWER_L + R.TIP_CLOSED + R.TIP_CLEAR)[2])
except Exception as e:
    print("ABORT", e)
