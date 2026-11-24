"""Offline: knob detection on the attempt 109 looks (drawer next to the cabinet) with a 5 cm and a 7 cm search radius."""
import os, sys
import cv2
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R + "/programs")
import knob
for f in ("look_1_1_hi.jpg", "look_1_0_hi.jpg"):
    im = cv2.resize(cv2.imread(R + "/attempts/attempt_109/" + f), (1280, 720), interpolation=cv2.INTER_CUBIC)
    for rad in (0.05, 0.07):
        r = knob.detect(im, 1298 / 0.29, (620, 250), rad)
        print(f, rad, None if r is None else (r["knob_uv"].round(0).tolist(), r["frame_size_m"]))
