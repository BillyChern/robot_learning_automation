"""Offline: what is the thin blob beside the drawer at its new spot (attempt 14 cloud)?"""
import json, os
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = json.load(open(R + "/inputs/calibration/side_pose_935322070708.json"))
Tbs = np.array(sp.get("T_base_side", sp.get("T_base_cam")), float); cam = Tbs[:3, 3]
print("side camera at", np.round(cam, 3).tolist())
P = np.load(R + "/attempts/attempt_14/cloud.npy"); tz = -0.0153; h = P[:, 2] - tz
m = (P[:, 0] > 0.20) & (P[:, 0] < 0.37) & (P[:, 1] > -0.10) & (P[:, 1] < 0.13) & (h > 0.008) & (h < 0.16)
Q, hq = P[m], h[m]
print("max height (mm) per 1 cm cell: rows y (high to low), cols x 0.20..0.36; '.' none")
xe = np.arange(0.20, 0.371, 0.01); ye = np.arange(-0.10, 0.131, 0.01)
print("        " + " ".join(f"{x*100:3.0f}" for x in xe[:-1]))
for j in range(len(ye) - 2, -1, -1):
    row = []
    for i in range(len(xe) - 1):
        s = (Q[:, 0] >= xe[i]) & (Q[:, 0] < xe[i+1]) & (Q[:, 1] >= ye[j]) & (Q[:, 1] < ye[j+1])
        row.append(f"{hq[s].max()*1000:3.0f}" if s.sum() else "  .")
    print(f"{ye[j]:+.2f}  " + " ".join(row))
s = (Q[:, 0] > 0.23) & (Q[:, 0] < 0.262) & (Q[:, 1] > 0.0) & (Q[:, 1] < 0.12)
S = Q[s]; print("streak n", len(S), "x", np.round(np.percentile(S[:,0],[2,50,98]),3).tolist(), "y", np.round(np.percentile(S[:,1],[2,50,98]),3).tolist(), "h", np.round(np.percentile(hq[s],[2,50,98]),3).tolist())
c = np.polyfit(S[:,1], S[:,0], 1); print("streak line x = %.3f %+.3f y" % (c[1], c[0]))
d = S[np.argmin(S[:,1])] - cam; print("direction from camera to its near end (xy, unit)", np.round(d[:2] / np.linalg.norm(d[:2]), 3).tolist())
