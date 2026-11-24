"""Offline: is the righting move reachable for a drawer lying open side up at ~(0.285, -0.064), any yaw?"""
import os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_reset as T
from rlpj.robot import sim
from rlpj.robot.plan import Planner, SLOW, NORMAL
R = T.R; KIN = T.KIN; pl = Planner(KIN, None, log=lambda m: None); tz = R.calib.table_z()
c = np.array([0.285, -0.064])
def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
for ang in range(0, 360, 30):
    nf = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    G = c + nf * 0.039
    yaw = ang
    Tg = R.tool_down_T(np.r_[G, tz + 0.104], yaw); Tup = Tg.copy(); Tup[2, 3] = tz + 0.22
    tipz = 0.036
    P = Tup[:3, 3] + tipz * Tup[:3, 2]
    Rr = rot(np.cross(np.r_[nf, 0], [0, 0, 1.0]), 90.0) @ Tup[:3, :3]
    Tr = np.eye(4); Tr[:3, :3] = Rr; Tr[:3, 3] = P - tipz * Rr[:, 2]
    Tdown = Tr.copy(); Tdown[2, 3] = tz + 0.0885 - 0.005 - 0.006 - (P[2] - Tr[2, 3]) + 0.0
    out = []
    q = None
    for nm, A_, B_ in (("descend", None, Tg), ("lift", Tg, Tup), ("turn", Tup, Tr), ("lower", Tr, Tdown)):
        try:
            if q is None:
                q = pl.ik_exact(B_, R.SEED_DOWN)[0]; out.append(f"{nm}:ik"); continue
            seg = pl.cart_line(nm, q, B_, SLOW)
            ok, rep = sim.check(seg.q, KIN, SLOW, gripper_frac=0.9, table_z=tz)
            out.append(f"{nm}:{'ok' if ok else 'NO ' + str({k: v for k, v in rep.items() if k in ('joint_limit_violations','near_base','min_tip_above_table','min_body_above_table','min_link_z','keep_in_violations','keep_out_violations') and v not in (0, None)})[:90]}")
            q = seg.q[-1]
        except Exception as e:
            out.append(f"{nm}:ERR {str(e)[:50]}"); break
    print(f"front normal {ang:3d} deg: " + " | ".join(out))
