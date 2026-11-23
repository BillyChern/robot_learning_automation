"""Wrist look-downs at the standing drawer and at the target compartment, from free space (tool >= 0.20 m above the
table, gripper open, nothing touched). Saves <= 640 px crops, the tool poses and the projected nominal outlines, to
build the wrist perception offline."""
import json
import os
import sys
import time

os.environ.setdefault("RLPJ_TASK", "drawer_lying")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from rlpj import config
from rlpj.perception import tabletop as tt
from rlpj.perception.wrist import project_base_to_wrist
from rlpj.robot import calib
from rlpj.robot.plan import tool_down_T
from rlpj.robot.session import RobotSession

CAB = config.load("drawer_lying")["task"]["fixtures"]["cabinet"]
SEED_DOWN = np.array([0, 25, -110, 0, -45, 90.0])
READY_XYZ = (0.40, -0.10, 0.25)


def look_T(target_xyz, tool_h, tz, yaw=0.0):
    """Tool pointing down at height tool_h above the table, shifted so the camera's optical axis hits target_xyz."""
    Ttc = calib.T_tool_cam()
    T = tool_down_T(np.array([target_xyz[0], target_xyz[1], tz + tool_h]), yaw)
    Tc = T @ Ttc
    o, zc = Tc[:3, 3], Tc[:3, 2]
    hit = o + (target_xyz[2] - o[2]) / zc[2] * zc
    T[:2, 3] += np.asarray(target_xyz[:2]) - hit[:2]
    return T


def outline(c, yaw_deg, size, z):
    y = np.radians(yaw_deg)
    a, b = np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])
    return np.array([np.r_[np.asarray(c) + sa * size[0] / 2 * a + sb * size[1] / 2 * b, z]
                     for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def crop(img, uv, size=640):
    H, W = img.shape[:2]
    u, v = (W / 2, H / 2) if uv is None or not np.all(np.isfinite(uv)) else uv
    x0 = int(np.clip(u - size / 2, 0, max(W - size, 0))); y0 = int(np.clip(v - size / 2, 0, max(H - size, 0)))
    return img[y0:y0 + size, x0:x0 + size].copy(), x0, y0


with RobotSession(program="execute") as s:
    s.phase("perceive")
    P = tt.cloud(s.side, n=7)
    tz = tt.table_fit(P)
    blobs = tt.detect(P, table_z=tz)
    dr = [b for b in blobs if 0.05 < b["height"] < 0.14 and 0.06 < max(b["extent"]) < 0.17 and b["n"] > 500]
    s.require(bool(dr), "no drawer-sized blob on the table")
    Q = np.asarray(dr[0]["points"], float)
    h = Q[:, 2] - tz
    H, e = np.histogram(h[h > 0.05], bins=np.arange(0.05, 0.125, 0.002))
    k = int(np.argmax(H))
    face = float(np.median(h[(h > e[k] - 0.002) & (h < e[k + 1] + 0.002)]))
    F = Q[np.abs(h - face) < 0.004, :2]
    (cx, cy), (w0, h0), ang = cv2.minAreaRect(np.ascontiguousarray(F.astype(np.float32)))
    drawer = dict(c=[round(cx, 4), round(cy, 4)], size=[round(w0, 4), round(h0, 4)], ang=round(ang, 1), face_h=round(face, 4))
    s.summary(objects=dict(drawer=drawer), table_z=round(tz, 4))
    s.decide(f"drawer top face {drawer}")

    def ik(T, seeds):
        best = None
        q_now = np.asarray(s.feedback()["q"], float)
        for sd in seeds:
            try:
                q, _, _ = s.planner.ik_exact(T, sd)
            except Exception as ex:
                s.log(f"IK failed from a seed: {str(ex)[:80]}")
                continue
            d = float(np.max(np.abs(q - q_now)))
            if best is None or d < best[1]:
                best = (q, d)
        s.require(best is not None, f"no IK for {np.round(T[:3, 3], 3).tolist()}")
        return best[0]

    q_ready = ik(tool_down_T(np.r_[READY_XYZ[:2], tz + READY_XYZ[2]], 0.0), [SEED_DOWN])
    s.phase("look")
    s.move_joint("ready", q_ready, "normal")
    c_d, zd = np.array([cx, cy]), tz + face
    c_o, zo = np.asarray(CAB["target_centre"], float), tz + CAB["rim_h"]
    views = [("drawer_high", c_d, zd, 0.30, outline(c_d, ang, (w0, h0), zd)),
             ("drawer_low", c_d, zd, 0.20, outline(c_d, ang, (w0, h0), zd)),
             ("opening_high", c_o, zo, 0.30, outline(c_o, CAB["yaw_deg"], CAB["compartment"], zo)),
             ("opening_low", c_o, zo, 0.20, outline(c_o, CAB["yaw_deg"], CAB["compartment"], zo))]
    rec = {}
    for name, c, z, th, corners in views:
        T = look_T(np.r_[c, z], th, tz)
        s.move_joint(f"look_{name}", ik(T, [np.asarray(s.feedback()["q"], float), SEED_DOWN]), "normal")
        time.sleep(0.8)
        img, _ = s.wrist.latest()
        Tm = calib.pose_to_T(s.feedback()["pose"])
        uv = project_base_to_wrist(corners, Tm)
        uc = project_base_to_wrist(np.r_[c, z][None], Tm)[0]
        im, x0, y0 = crop(img, uc)
        cv2.imwrite(os.path.join(s.A, f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ov = im.copy()
        if np.all(np.isfinite(uv)):
            cv2.polylines(ov, [np.int32(uv - [x0, y0])], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(s.A, f"{name}_ov.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec[name] = dict(T=np.round(Tm, 5).tolist(), crop_xy=[x0, y0], corners_uv=np.round(uv, 1).tolist(),
                         corners_base=np.round(corners, 4).tolist())
    with open(os.path.join(s.A, "views.json"), "w") as f:
        json.dump(rec, f)
    s.phase("back")
    s.move_joint("ready_back", q_ready, "normal")
    s.home()
    s.summary(step="done", first_failed_check=None, verify=dict(views=list(rec)))
    s.claim("fail")   # a probe: nothing to claim
