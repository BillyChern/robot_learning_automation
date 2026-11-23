"""Offline: the table's camera-side edge from wrist images (the side camera's view ends at the image border before the edge)."""
import json, os
import cv2
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
he = json.load(open(R + "/inputs/calibration/handeye_kinova.json")); Ttc = np.array(he["T_tool_cam"])
f = json.load(open(R + "/inputs/calibration/intrinsics_kinova_1280x720.json"))["factory"]
K = np.array([[f["fx"], 0, f["cx"]], [0, f["fy"], f["cy"]], [0, 0, 1.0]]); D = np.array(f["coeffs"], float)
TZ = -0.0156
def pose_T(p):
    a, b, c = np.radians(p[3:6])
    Rx = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    Rz = np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    T = np.eye(4); T[:3,:3] = Rz @ Ry @ Rx; T[:3,3] = p[:3]; return T
def hit(T, u, v):
    n = cv2.undistortPoints(np.array([[[2.0*u, 2.0*v]]]), K, D)[0,0]; Tc = T @ Ttc; d = Tc[:3,:3] @ np.r_[n,1.0]
    return (Tc[:3,3] + (TZ - Tc[2,3]) / d[2] * d)[:2]
pts = {}
for att, mode in (("attempt_8", "col"), ("attempt_9", "row")):
    A = f"{R}/attempts/{att}"; S = json.load(open(A + "/summary.json"))["stop"]; T = pose_T(np.array(S["tool_meas"], float))
    im = cv2.imread(A + "/stop_wrist.jpg"); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    tab = cv2.morphologyEx(((hsv[...,1] < 40) & (hsv[...,2] > 130)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    out = []
    if mode == "col":                     # edge crosses the image horizontally: first table row per column, away from the drawer
        for u in list(range(10, 225, 15)) + list(range(415, 630, 15)):
            r = np.flatnonzero(tab[:, u]); 
            if len(r) and 3 < r[0] < 200: out.append(hit(T, u, r[0]))
    else:                                 # edge runs vertically: first table column per row, above the drawer
        for v in range(100, 265, 12):
            c = np.flatnonzero(tab[v, :]); 
            if len(c) and 3 < c[0] < 500: out.append(hit(T, c[0], v))
    pts[att] = np.array(out); print(att, "tool", np.round(T[:3,3],3).tolist(), "n", len(out), "x", pts[att][:,0].min().round(3) if len(out) else None, pts[att][:,0].max().round(3) if len(out) else None)
for k, P in list(pts.items()) + [("both", np.vstack(list(pts.values())))]:
    c = np.polyfit(P[:,0], P[:,1], 1); r = P[:,1] - np.polyval(c, P[:,0])
    print(f"{k}: y = {c[1]:+.4f} {c[0]:+.4f} x, rms {r.std()*1000:.1f} mm, max {np.abs(r).max()*1000:.1f} mm; at x .29 {np.polyval(c,.29):+.3f}, .40 {np.polyval(c,.40):+.3f}, .50 {np.polyval(c,.50):+.3f}, .60 {np.polyval(c,.60):+.3f}")
print("operator line: at .29", round(-0.13-.29/6,3), ".40", round(-0.13-.40/6,3), ".50", round(-0.13-.50/6,3), ".60", round(-0.13-.60/6,3))
c8 = np.polyfit(pts["attempt_8"][:,0], pts["attempt_8"][:,1], 1)
d = (pts["attempt_9"][:,1] - np.polyval(c8, pts["attempt_9"][:,0])) / np.hypot(c8[0], 1)
print(f"attempt 9 edge points vs the attempt 8 line: median {np.median(d)*1000:.1f} mm, spread {d.min()*1000:.0f}..{d.max()*1000:.0f} mm")
