"""Offline: the table's camera-side edge from the side cloud, and the drawer outline's margin to it over the target range."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = np.load(os.path.join(ROOT, "attempts", "attempt_1", "cloud.npy"))
T = P[np.abs(P[:, 2] + 0.0156) < 0.005]
xs, ys = [], []
for x in np.arange(0.15, 0.56, 0.025):
    m = np.abs(T[:, 0] - x) < 0.0125
    if m.sum() > 20:
        v = np.sort(T[m, 1])
        xs.append(x)
        ys.append(v[min(2, len(v) - 1)])          # third-lowest point: the edge without single outliers
c = np.polyfit(xs, ys, 1)
res = np.array(ys) - np.polyval(c, xs)
print(f"measured edge: y = {c[1]:+.4f} {c[0]:+.4f} x, residual max {np.abs(res).max() * 1000:.1f} mm, n {len(xs)}")
for x in (0.25, 0.29, 0.30, 0.40, 0.53):
    print(f"  x {x:.2f}: measured {np.polyval(c, x):+.3f}  operator {-0.18 - (x - 0.3) / 6:+.3f}")


def margin(xy, delta, line, h=(0.040, 0.058)):
    """Distance from the drawer outline to the edge y = b + m x (drawer turned delta deg from the cabinet's axis)."""
    m_, b_ = line
    n = np.array([-m_, 1.0]) / np.hypot(m_, 1.0)
    a = np.radians(21.2 + delta)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    return float(n @ (np.asarray(xy) - np.array([0.0, b_])) - h[0] * abs(n @ u) - h[1] * abs(n @ v))


lines = {"measured": (c[0], c[1]), "operator": (-1 / 6, -0.18 + 0.05)}
print("outline margin (cm) to the edge, by turn from the cabinet axis")
for name, ln in lines.items():
    print(name)
    for xy in ((0.28, -0.03), (0.29, -0.02), (0.30, -0.01)):
        print(f"  target {xy}: " + "  ".join(f"{d:+d}:{margin(xy, d, ln) * 100:.1f}" for d in (-30, -15, 0, 15, 30, 45, 60)))
