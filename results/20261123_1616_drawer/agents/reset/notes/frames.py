"""Offline: wrist panels of an attempt's dashboard at given log times, as one montage in notes/debug/."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
attempt, out, times = sys.argv[1], sys.argv[2], sys.argv[3:]
A = os.path.join(ROOT, "attempts", attempt)
L = open(os.path.join(A, "log.txt")).read().splitlines()


def sec(s):
    h, m, x = map(int, s.split(":"))
    return 3600 * h + 60 * m + x


t0, t1 = sec(L[0][1:9]), sec(L[-1][1:9])
c = cv2.VideoCapture(os.path.join(A, "dashboard.mp4"))
n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
fps = n / max(t1 - t0, 1)
panels = []
for v in times:
    c.set(cv2.CAP_PROP_POS_FRAMES, min(int((sec(v) - t0) * fps), n - 1))
    ok, f = c.read()
    if ok:
        panels.append(cv2.putText(cv2.resize(f[:, f.shape[1] // 2:], (320, 240)), v, (6, 18), 0, 0.5, (0, 255, 255), 1))
while len(panels) % 2:
    panels.append(np.zeros((240, 320, 3), np.uint8))
rows = [np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]
os.makedirs(os.path.join(ROOT, "notes", "debug"), exist_ok=True)
cv2.imwrite(os.path.join(ROOT, "notes", "debug", out), np.vstack(rows))
print("wrote", out, len(panels), "panels")
