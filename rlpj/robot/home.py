"""Return-to-default-pose planning (shared by the main program and the watchdog): if the tool is low, first a
slow straight lift, then a normal-speed joint move to the default joints. Every segment is checked in the model."""
import numpy as np
from . import calib, sim
from .plan import NORMAL, SLOW


def plan_return(planner, q_now, q_home, grip_frac=0.0, lift_to=0.20):
    segs = []
    T = planner.kin.fk(q_now)
    if T[2, 3] < calib.table_z() + lift_to - 0.01:
        Tg = T.copy(); Tg[2, 3] = calib.table_z() + lift_to
        segs.append(planner.cart_line("return_lift", q_now, Tg, SLOW, exact_goal=False))
        q_now = segs[-1].q[-1]
    segs.append(planner.joint_move("return_home", q_now, np.asarray(q_home, float), NORMAL))
    reps = []
    for s in segs:
        ok, rep = sim.check(s.q, planner.kin, s.speed, gripper_frac=grip_frac, tip_margin=-0.01 if s.name == "return_lift" else 0.008)
        rep["segment"] = s.name
        reps.append(rep)
        if not ok:
            return segs, reps, False
    return segs, reps, True
