"""Probe (no insertion): which wall carries the pull? Score the drawer walls in the side image on the table, then
grasp the robot-facing wall, lift and turn the drawer by 0/90/180/270 deg in view of the side camera, score again,
put it back where it was, home. Saves crops (<= 440 px) of every view."""
import os
import sys
import time

os.environ.setdefault("RLPJ_TASK", "drawer_ng")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np

from rlpj.robot.plan import tool_down_T
from rlpj.robot.session import RobotSession
import execute as X
import pullface as PF

L_DEF = 0.13


def rot2(v, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def main():
    with RobotSession(program="execute") as s:
        ep = X.Episode(s)
        s.phase("perceive")
        if np.abs(X.wrap180(ep.q_now() - X.HOME_Q)).max() > 3:
            ep.go_home("home_start")
        ep.set_home()
        P, tz, blobs = ep.scene("before")
        d = ep.find_drawer(blobs, tz)
        s.require(d is not None, "no drawer on the table")
        img = s.side.latest()[0]
        eA = d["c"] + d["p"] * d["half"]
        d0 = -d["p"]
        c = eA + d0 * L_DEF / 2
        sc0 = PF.score_faces(img, PF.faces_of(c, d0, L_DEF, L_DEF), tz)
        cv2.imwrite(os.path.join(s.A, "crop_table.jpg"), PF.crop_around(img, np.r_[c, tz + 0.04]))
        views = [dict(view="table", scores=sc0)]
        s.log(f"table view scores {sc0}")
        s.phase("grasp")
        held = ep.grasp(d)
        s.require(held is not None, "front wall not grasped")
        e = held["e"]; yaw = held["yaw"]
        for rot in (0, 90, 180, 270):
            T = tool_down_T(np.r_[e, ep.tz + X.SAFE_H], yaw + rot)
            ep.joint_to(f"turn{rot}", T)
            time.sleep(0.6)
            img = s.side.latest()[0]
            dn = rot2(held["d0"], rot)
            ch = e + dn * (L_DEF / 2 - 0.006)
            zb = ep.tz + X.SAFE_H - held["drop"]
            sc = PF.score_faces(img, PF.faces_of(ch, dn, L_DEF, L_DEF), zb)
            cv2.imwrite(os.path.join(s.A, f"crop_rot{rot}.jpg"), PF.crop_around(img, np.r_[ch, zb + 0.04]))
            views.append(dict(view=f"rot{rot}", scores=sc))
            s.log(f"rot {rot}: {sc}")
            s.summary(views=views)
        pull, best = PF.decide([v["scores"] for v in views])
        s.summary(pull=pull, best_grey=best)
        s.decide(f"pull wall: {pull} (best grey per wall {best})")
        s.phase("put_back")
        T = tool_down_T(np.r_[e, ep.tz + X.SAFE_H], yaw)
        ep.joint_to("turn_back", T)
        Tp = held["Tg"].copy(); Tp[2, 3] += 0.008
        Tpre = held["Tg"].copy(); Tpre[2, 3] += 0.05
        ep.line("down_pre", Tpre, "normal")
        ep.line("place", Tp, "slow", tip_margin=0.0)
        s.gripper(0.4, "release")
        ep.line("up", Tpre, "slow", tip_margin=0.0)
        ep.lift_to("clear", X.SAFE_H)
        ep.go_home("home_end")
        s.summary(objects=dict(drawer=dict(xy=d["xy"])), verify=dict(pull=pull, best=best), transitions=0, **{"return": 0.0})
        s.claim("success" if pull else "fail")


if __name__ == "__main__":
    main()
