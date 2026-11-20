"""Solve T_base_side from scripts/side_calib_capture.py's captures, with wristcalib.py's own maths and files.

    python scripts/side_calib_solve.py <captures_dir> [--write]
Per pose: T_base_board = T_base_tool @ T_tool_cam @ T_wristcam_board, T_base_side = T_base_board @ inv(T_side_board);
poses averaged (wristcalib.mean_T), outliers (> 3x median deviation) dropped. --write saves side_pose_<serial>.json
(back it up first) with the same fields as `wristcalib.py side-pose`, plus "automated": true.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/realsense_calib"))
import rscalib as rc  # noqa: E402
import wristcalib as wc  # noqa: E402

SIDE_SERIAL, ROBOT = "935322070708", "192.168.2.9"


def main():
    cap = Path(sys.argv[1]); write = "--write" in sys.argv
    poses = json.loads((cap / "poses.json").read_text())
    he_file = rc.RESULTS / f"handeye_{wc.kinova_id(ROBOT)}.json"
    he = json.loads(he_file.read_text()); X, order = np.array(he["T_tool_cam"]), he["euler_order"]
    board = rc.make_board(); det = rc.make_detector(board)
    robot = wc.Kinova(ROBOT); cam = wc.KinovaCam(robot)           # the wrist camera's factory intrinsics, as the GUI
    Kw, Dw, _ = rc.load_intrinsics(cam.meta, "auto")
    for f in ("stop", "close"):
        for o in (cam, robot):
            getattr(o, f, lambda: None)()
    Ks, Ds, ssrc = rc.load_intrinsics({"serial": SIDE_SERIAL, "resolution": [1280, 720]}, "auto")
    rows = []
    for n, p in enumerate(poses):
        fw, fs = cv2.imread(str(cap / f"{n:02d}_wrist.png")), cv2.imread(str(cap / f"{n:02d}_side.png"))
        o, i = rc.obj_img(board, rc.detect(det, fw)); ew, rv, tv = rc.view_rms(o, i, Kw, Dw)
        T_bb = wc.pose_to_T(p["tool_pose"], order) @ X @ wc.rt_to_T(rv, tv)
        o, i = rc.obj_img(board, rc.detect(det, fs)); es, rv, tv = rc.view_rms(o, i, Ks, Ds)
        T_sb = wc.rt_to_T(rv, tv)
        rows.append(dict(p, T_base_board=T_bb.tolist(), T_side_board=T_sb.tolist(),
                         T_base_side=(T_bb @ rc.inv(T_sb)).tolist(), wrist_px=ew, side_px=es))
    Ts = [np.array(r["T_base_side"]) for r in rows]
    T = wc.mean_T(Ts)
    dev = np.array([np.linalg.norm(t[:3, 3] - T[:3, 3]) for t in Ts])
    keep = dev <= max(3 * np.median(dev), 0.004)
    Ts = [t for t, k in zip(Ts, keep) if k]; rows = [r for r, k in zip(rows, keep) if k]
    T = wc.mean_T(Ts)
    spread_mm = float(np.sqrt(np.mean([np.sum((t[:3, 3] - T[:3, 3]) ** 2) for t in Ts]))) * 1000
    spread_deg = float(np.sqrt(np.mean([wc.angle_deg(t[:3, :3], T[:3, :3]) ** 2 for t in Ts])))
    old = np.array(json.loads((rc.RESULTS / f"side_pose_{SIDE_SERIAL}.json").read_text())["T_base_side"])
    print(f"poses {len(Ts)} (dropped {int((~keep).sum())}) | spread {spread_mm:.1f} mm, {spread_deg:.2f} deg | "
          f"camera at {np.round(T[:3, 3] * 1000).tolist()} mm | moved {1000 * np.linalg.norm(T[:3, 3] - old[:3, 3]):.0f} mm, "
          f"{wc.angle_deg(old[:3, :3], T[:3, :3]):.1f} deg from the old calibration")
    bz = [np.array(r["T_base_board"])[2, 3] for r in rows]
    devs = [1000 * np.linalg.norm(np.array(r["T_base_side"])[:3, 3] - T[:3, 3]) for r in rows]
    print("board z from the wrist (m): mean %.4f, sd %.4f | per-pose deviation (mm): %s" % (np.mean(bz), np.std(bz), np.round(devs, 1).tolist()))
    print("per-pose wrist corners / side corners:", [(len(rc.detect(det, cv2.imread(str(cap / f"{n:02d}_wrist.png")))), len(rc.detect(det, cv2.imread(str(cap / f"{n:02d}_side.png"))))) for n in range(len(poses))])
    json.dump(dict(T=T.tolist()), open(cap / "solution.json", "w"))
    if write:
        res = {"serial": SIDE_SERIAL, "name": "side D435", "intrinsics": ssrc, "T_base_side": T.tolist(),
               "n_poses": len(Ts), "spread_mm": spread_mm, "spread_deg": spread_deg, "automated": True,
               "convention": "T_base_side: side color optical frame (OpenCV) in the robot base frame, metres "
                             "(p_base = T_base_side @ p_side)", "robot_ip": ROBOT,
               "uses_handeye": {"file": str(he_file.relative_to(rc.ROOT)), "T_tool_cam": X.tolist()},
               "captures": str(cap), "poses": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        (rc.RESULTS / f"side_pose_{SIDE_SERIAL}.json").write_text(json.dumps(res, indent=2))
        print("saved", rc.RESULTS / f"side_pose_{SIDE_SERIAL}.json")


if __name__ == "__main__":
    main()
