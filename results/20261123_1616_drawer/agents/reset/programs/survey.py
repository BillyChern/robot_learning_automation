"""Perception-only survey of the lying-cabinet scene (no motion): blobs in the base and cabinet frames, the height
profile of each compartment, side and wrist images, and a downsampled cloud for offline analysis."""
import os

import cv2
import numpy as np

from rlpj import config
from rlpj.perception import tabletop as tt
from rlpj.robot import calib
from rlpj.robot.session import RobotSession

FX = config.load("drawer_lying")["task"]["fixtures"]["cabinet"]
YAW = np.radians(FX["yaw_deg"])
A_AX = np.array([np.cos(YAW), np.sin(YAW)])          # along the row of compartments, away from the robot
B_AX = np.array([-A_AX[1], A_AX[0]])
PITCH = 0.0878                                       # compartment spacing along A_AX (divider centres in the task yaml)
T0 = np.array(FX["target_centre"], float)


def cab(xy):
    d = np.asarray(xy, float) - np.asarray(FX["centre"], float)
    return np.stack([d @ A_AX, d @ B_AX], -1)


with RobotSession(program="reset") as s:
    s.phase("survey")
    fb = s.feedback()
    s.log(f"tool pose {np.round(fb['pose'], 4).tolist()} q {np.round(fb['q'], 1).tolist()} gripper {fb['gripper']:.1f}")
    P = tt.cloud(s.side, n=7)
    tz = tt.table_fit(P)
    blobs = tt.detect(P, table_z=tz)
    out = []
    for b in blobs:
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.ascontiguousarray(b["points"][:, :2].astype(np.float32)))
        out.append(dict(xy=np.round(b["xy"], 3).tolist(), cab=np.round(cab(b["xy"]), 3).tolist(), h=b["height"],
                        ext=b["extent"], n=b["n"], box=[round(max(w, h), 3), round(min(w, h), 3)], ang=round(ang, 1)))
        s.log(f"blob {out[-1]}")
    comps = []
    hh = P[:, 2] - tz
    for k in range(3):
        d = P[:, :2] - (T0 + k * PITCH * A_AX)
        m = (np.abs(d @ A_AX) < FX["compartment"][0] / 2 - 0.01) & (np.abs(d @ B_AX) < FX["compartment"][1] / 2 - 0.01)
        v = hh[m]
        comps.append(dict(k=k, n=int(m.sum()), p10=round(float(np.percentile(v, 10)), 4) if len(v) else None,
                          med=round(float(np.median(v)), 4) if len(v) else None,
                          p90=round(float(np.percentile(v, 90)), 4) if len(v) else None))
        s.log(f"compartment {comps[-1]}")
    keep = (hh > -0.005) & (hh < 0.25)
    idx = np.flatnonzero(keep)[:: max(1, int(keep.sum() / 40000))]
    np.save(os.path.join(s.A, "cloud.npy"), P[idx].astype(np.float32))
    s.summary(blobs=out, compartments=comps, table_z_fit=round(tz, 4), gripper=round(float(fb["gripper"]), 1),
              tool_pose=np.round(fb["pose"], 4).tolist())
    side = s.side.latest()[0]
    cv2.imwrite(os.path.join(s.A, "side.jpg"), cv2.resize(side, (640, 360)))
    wrist = s.wrist.latest()[0]
    cv2.imwrite(os.path.join(s.A, "wrist.jpg"), cv2.resize(wrist, (640, 360)))
    s.claim("fail")
