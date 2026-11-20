"""Scene geometry helpers for the drawer reset (pure functions, no robot)."""
import cv2
import numpy as np

from rlpj import config

TASK = config.load("drawer_rl")["task"]
CAB = TASK["fixtures"]["cabinet"]
N = np.array(CAB["front_normal"], float)            # into the cabinet
TT = np.array([-N[1], N[0]])                        # along the front (t = -n_y x + n_x y)
OFF = float(CAB["front_offset"])
SLOT_T = CAB["slot_t"]
SLOT_H = CAB["slot_h"]
SLOT_C = (SLOT_T[0] + SLOT_T[1]) / 2               # slot window centre along the front (-0.0825 before 2026-11-20)
DT = SLOT_C - (-0.0825)                             # shift of all along-front (t) numbers measured before 2026-11-20


def _cab_t():
    """t-span of the cabinet body from the task keep-out box named 'cabinet' (yaml; fallback slot-relative)."""
    for b in TASK.get("keep_out") or []:
        if b.get("name") == "cabinet":
            c = np.asarray(b["center"], float)[:2]
            tc = float(c[0] * TT[0] + c[1] * TT[1])
            return (tc - float(b["half"][1]), tc + float(b["half"][1]))
    return (-0.155 + DT, -0.005 + DT)


CAB_T = _cab_t()                                    # (-0.155, -0.005) before 2026-11-20


def nt(xy):
    """base xy -> (n, t) with n = distance along the normal (front plane at n = OFF)."""
    xy = np.asarray(xy, float)
    return np.stack([xy[..., 0] * N[0] + xy[..., 1] * N[1], xy[..., 0] * TT[0] + xy[..., 1] * TT[1]], -1)


def xy_of(n, t):
    return n * N + t * TT


def obb(xy):
    """Oriented box of 2-D points: centre, (long, short) sizes, yaw (deg) of the long axis in [-90, 90)."""
    xy = np.ascontiguousarray(np.asarray(xy, np.float32).reshape(-1, 2))
    (cx, cy), (w, h), a = cv2.minAreaRect(xy)
    if w < h:
        w, h, a = h, w, a + 90.0
    a = (a + 90.0) % 180.0 - 90.0
    return np.array([cx, cy]), (float(w), float(h)), float(a)


def front_profile(P, table_z, dn=0.08):
    """Points in front of the bottom slot (base frame): per 1 cm height band the max protrusion (m) in front of the
    front plane and the t-range there."""
    q = nt(P[:, :2]); h = P[:, 2] - table_z
    m = (q[:, 0] > OFF - dn) & (q[:, 0] < OFF + 0.02) & (q[:, 1] > SLOT_T[0] - 0.03) & (q[:, 1] < SLOT_T[1] + 0.03) & (h > 0.004) & (h < 0.13)
    out = []
    for z0 in np.arange(0.0, 0.13, 0.01):
        s = m & (h >= z0) & (h < z0 + 0.01)
        if s.sum() < 5:
            continue
        out.append(dict(h=round(float(z0), 2), n=int(s.sum()), protr=round(float(OFF - np.percentile(q[s, 0], 3)), 4),
                        t=[round(float(np.percentile(q[s, 1], 3)), 3), round(float(np.percentile(q[s, 1], 97)), 3)]))
    return out
