"""Perception-only survey (no motion): blobs, oriented boxes, the profile in front of the bottom slot, images."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj import config
from rlpj.robot import calib
from rlpj.robot.session import RobotSession
from rlpj.perception import tabletop as tt
import scene

with RobotSession(program="reset") as s:
    A = s.A
    s.phase("survey")
    s.log(f"task in session config: {config.load().get('task', {}).get('name')}  keep_out={bool(config.load().get('task', {}).get('keep_out'))}")
    fb = s.feedback()
    s.log(f"tool pose {np.round(fb['pose'], 4).tolist()} q {np.round(fb['q'], 1).tolist()} gripper {fb['gripper']:.1f}")
    P = tt.cloud(s.side, n=7)
    tz = tt.table_fit(P)
    blobs = tt.detect(P)
    out = []
    for b in blobs:
        c, (L, W), yaw = scene.obb(b["points"][:, :2])
        q = scene.nt(b["xy"])
        out.append(dict(xy=np.round(b["xy"], 3).tolist(), h=b["height"], ext=b["extent"], n=b["n"], obb_c=np.round(c, 3).tolist(),
                        obb_LW=[round(L, 3), round(W, 3)], yaw=round(yaw, 1), n_front=round(float(q[0] - scene.OFF), 3), t=round(float(q[1]), 3)))
        s.log(f"blob {out[-1]}")
    prof = scene.front_profile(P, tz)
    s.log(f"table_z fit {tz:.4f} (calib {calib.table_z():.4f}); front profile: {prof}")
    s.summary(blobs=out, table_z_fit=round(tz, 4), front_profile=prof, tool_pose=np.round(fb["pose"], 4).tolist())
    # images: side with the cabinet front, slot and blobs drawn
    col = s.side.latest()[0].copy()
    K, dist = calib.side_intrinsics(); Tsb = np.linalg.inv(calib.T_base_side())
    def px(Pb):
        return calib.project(calib.apply(Tsb, np.atleast_2d(Pb)), K, dist).astype(int)
    for t0, t1 in [(-0.135, -0.026), tuple(scene.SLOT_T)]:
        for h in (scene.SLOT_H[0], scene.SLOT_H[1]):
            a, b2 = px(np.r_[scene.xy_of(scene.OFF, t0), tz + h]), px(np.r_[scene.xy_of(scene.OFF, t1), tz + h])
            cv2.line(col, tuple(a[0]), tuple(b2[0]), (0, 0, 255), 2)
    for i, b in enumerate(blobs):
        u = px(np.r_[b["xy"], b["z_top"]])[0]
        cv2.circle(col, tuple(u), 6, (0, 255, 0), 2); cv2.putText(col, str(i), tuple(u + 8), 0, 0.8, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(A, "side.jpg"), cv2.resize(col, (640, 360)))
    u = px(np.r_[scene.xy_of(scene.OFF - 0.05, -0.08), tz + 0.05])[0]
    x0, y0 = int(np.clip(u[0] - 320, 0, 640)), int(np.clip(u[1] - 180, 0, 360))
    cv2.imwrite(os.path.join(A, "side_slot_crop.jpg"), s.side.latest()[0][y0:y0 + 360, x0:x0 + 640])
    w = s.wrist.latest()[0]
    cv2.imwrite(os.path.join(A, "wrist.jpg"), cv2.resize(w, (640, 360)))
    s.claim("fail")
