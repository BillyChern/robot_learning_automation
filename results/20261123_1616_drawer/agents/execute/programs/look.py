"""Read-only scene probe for drawer_lying: drawer footprint/height from side depth, target compartment profile (the
same window as rlpj.success.checks.drawer_lying), downscaled side and wrist images. Moves nothing."""
import os
import sys

os.environ.setdefault("RLPJ_TASK", "drawer_lying")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from rlpj import config
from rlpj.perception import tabletop as tt
from rlpj.robot.session import RobotSession

TASK = config.load("drawer_lying")["task"]
CAB = TASK["fixtures"]["cabinet"]


def rect(xy):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.ascontiguousarray(np.asarray(xy, np.float32)))
    return dict(c=[round(cx, 4), round(cy, 4)], size=[round(w, 4), round(h, 4)], ang=round(a, 1))


def compartment(P, tz):
    y = np.radians(CAB["yaw_deg"]); a = np.array([np.cos(y), np.sin(y)]); b = np.array([-a[1], a[0]])
    d = P[:, :2] - np.asarray(CAB["target_centre"], float)
    s, t, h = d @ a, d @ b, P[:, 2] - tz
    hl, hw = CAB["compartment"][0] / 2 - 0.01, CAB["compartment"][1] / 2 - 0.01
    m = (np.abs(s) < hl) & (np.abs(t) < hw) & (h > 0.003)
    if m.sum() == 0:
        return dict(n=0)
    hh = h[m]
    return dict(n=int(m.sum()), pct_h=np.round(np.percentile(hh, [5, 25, 50, 75, 95]), 4).tolist(),
                flush=round(float(np.mean(np.abs(hh - CAB["rim_h"]) < 0.012)), 3))


with RobotSession(program="execute") as s:
    s.phase("look")
    P = tt.cloud(s.side, n=9)
    tz = tt.table_fit(P)
    blobs = tt.detect(P, table_z=tz)
    out = []
    for b in blobs:
        Q = np.asarray(b["points"], float)
        h = Q[:, 2] - tz
        top = Q[h > float(b["height"]) - 0.012, :2]
        out.append(dict(xy=np.round(b["xy"], 4).tolist(), height=b["height"], extent=b["extent"], n=b["n"],
                        top_rect=rect(top) if len(top) >= 5 else None, top_n=int(len(top))))
    c, _, _ = s.side.latest()
    # drawer: top-face height (mode of the upper points), the pull (points above it), a sharp crop of the side image
    dr = [b for b in blobs if 0.05 < b["height"] < 0.14 and 0.06 < max(b["extent"]) < 0.17 and b["n"] > 500]
    pull = None
    if dr:
        Q = np.asarray(dr[0]["points"], float); h = Q[:, 2] - tz
        H, e = np.histogram(h[h > 0.05], bins=np.arange(0.05, 0.125, 0.002))
        k = int(np.argmax(H)); face = float(np.median(h[(h > e[k] - 0.002) & (h < e[k + 1] + 0.002)]))
        F = Q[np.abs(h - face) < 0.004, :2]; U = Q[h > face + 0.004]
        pull = dict(face_h=round(face, 4), hist_from_0p05_2mm=H.tolist(), face_rect=rect(F), n_above=int(len(U)))
        if len(U) >= 5:
            pull.update(above_rect=rect(U[:, :2]), above_h=np.round(np.percentile(U[:, 2] - tz, [50, 95]), 4).tolist())
        from rlpj.robot import calib
        K, dist = calib.side_intrinsics()
        Tcb = np.linalg.inv(calib.T_base_side())
        uv = calib.project(calib.apply(Tcb, np.array([[*F.mean(0), tz + face]])), np.asarray(K, float), np.asarray(dist, float))[0]
        u, v = int(uv[0]), int(uv[1]); r = 110
        crop = c[max(v - r, 0):v + r, max(u - r, 0):u + r]
        cv2.imwrite(os.path.join(s.A, "drawer_crop.jpg"), cv2.resize(crop, (440, 440), interpolation=cv2.INTER_CUBIC))
        pull["crop_center_uv"] = [u, v]
    w, _ = s.wrist.latest()
    cv2.imwrite(os.path.join(s.A, "side.jpg"), cv2.resize(c, (640, 360)))
    cv2.imwrite(os.path.join(s.A, "wrist.jpg"), cv2.resize(w, (640, 360)))
    fb = s.feedback()
    s.summary(table_z=round(tz, 4), pull=pull, blobs=out[:3], compartment=compartment(P, tz),
              tool_pose=np.round(fb["pose"], 4).tolist(), q=np.round(fb["q"], 2).tolist(),
              gripper=round(float(fb["gripper"]), 1),
              force=None if fb.get("force") is None else np.round(fb["force"], 2).tolist(),
              claimed_outcome="fail", step="look", first_failed_check="probe only", objects=out[:3], verify={})
    s.claim("fail")
