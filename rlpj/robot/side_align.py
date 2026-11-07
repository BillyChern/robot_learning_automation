"""Put a moved side camera back where it was: the live view over a reference image, with the offset and which way to
turn. Read-only (opens the side camera only).

    .venv/bin/python -m rlpj.robot.side_align [--ref runs/calib/side_reference.jpg] [--minutes 30] [--window]

The overlay goes to the status page (http://<host>:8091, runs/live/latest.jpg); --window also opens an OpenCV window
(needs an OpenCV build with GUI support).

Green when the view is within 1 px of the reference (about 0.1 deg): the side calibration then still holds; confirm
with `rlpj check` (table height) and FAR's move_to_check_pose.py. Q quits."""
import argparse
import time

import cv2
import numpy as np

from rlpj import config

SIZE = (640, 360)
REF = config.ROOT / "runs" / "calib" / "side_reference.jpg"


def offset(ref_gray, live_gray):
    """(dx, dy, n): median image shift of the live view against the reference (px at 640x360), from ORB matches kept by
    a RANSAC similarity fit; n = inliers (0 = no estimate)."""
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(live_gray, None)
    if d1 is None or d2 is None:
        return 0.0, 0.0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 8:
        return 0.0, 0.0, 0
    a = np.float32([k1[x.queryIdx].pt for x in m]); b = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None or inl.sum() < 8:
        return 0.0, 0.0, 0
    d = (b - a)[inl.ravel() == 1]
    return float(np.median(d[:, 0])), float(np.median(d[:, 1])), int(inl.sum())


def advice(dx, dy, tol=1.0):
    """Which way to turn the camera (the image content moved by dx, dy)."""
    if abs(dx) <= tol and abs(dy) <= tol:
        return "ALIGNED"
    out = []
    if abs(dx) > tol:
        out.append(f"pan {'LEFT' if dx < 0 else 'RIGHT'} {abs(dx):.0f} px")
    if abs(dy) > tol:
        out.append(f"tilt {'UP' if dy < 0 else 'DOWN'} {abs(dy):.0f} px")
    return ", ".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref", default=str(REF))
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--window", action="store_true", help="also show an OpenCV window (GUI build only)")
    a = p.parse_args(argv)
    from rlpj.robot.cams import SideCam
    ref = cv2.resize(cv2.imread(a.ref), SIZE)
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cam, t0, ok_n = SideCam(), time.time(), 0
    try:
        while time.time() - t0 < a.minutes * 60:
            img = cam.latest()[0]
            if img is None:
                time.sleep(0.1)
                continue
            live = cv2.resize(img, SIZE)
            dx, dy, n = offset(ref_g, cv2.cvtColor(live, cv2.COLOR_BGR2GRAY))
            msg = advice(dx, dy) if n else "no match: turn the camera toward the table"
            ok_n = ok_n + 1 if msg == "ALIGNED" else 0
            print(f"{time.strftime('%T')} dx {dx:+6.1f} dy {dy:+6.1f} px ({n} matches): {msg}", flush=True)
            view = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
            colour = (0, 200, 0) if msg == "ALIGNED" else (0, 0, 255)
            cv2.putText(view, msg + ("  - hold still" if ok_n else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(view, f"dx {dx:+.1f}  dy {dy:+.1f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            live_dir = config.RUNS / "live"                 # also on the status page (port 8091) while no program runs
            live_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(live_dir / "latest.tmp.jpg"), view)
            (live_dir / "latest.tmp.jpg").replace(live_dir / "latest.jpg")
            if a.window:
                cv2.imshow("side camera alignment (Q quits)", view)
            if a.window and cv2.waitKey(200) & 0xFF in (ord("q"), ord("Q")):
                break
            if not a.window:
                time.sleep(0.2)
    finally:
        cam.close()
        if a.window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
