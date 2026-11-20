"""Robot program for the automated side-camera calibration (scripts/auto_side_calib.py runs it via run_attempt).

Finds the ChArUco board on the table with the side camera (old calibration, a few cm off is fine for aiming), then
moves the wrist camera through views of it (heights, azimuths, tilts; every move model-checked) and saves at each
still pose: wrist image, side image, tool pose. The solver (scripts/side_calib_solve.py) does the maths.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.expanduser("~/realsense_calib"))
import rscalib as rc  # noqa: E402

from rlpj.robot import calib  # noqa: E402
from rlpj.robot.session import RobotSession  # noqa: E402

SIDE_SERIAL = "935322070708"
N_TARGET = int(os.environ.get("SIDECAL_POSES", "12"))
OUT = Path(os.environ.get("RLPJ_ATTEMPT_DIR", ".")) / "captures"


def rt_to_T(rv, tv):
    import cv2
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rv, np.float64))[0]
    T[:3, 3] = np.ravel(tv)
    return T


def look_at(p_cam, target, roll_deg):
    """Camera pose (OpenCV: z forward, y down in the image) at p_cam looking at target, rolled about z."""
    z = target - p_cam; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    y = np.cross(z, x)
    a = np.radians(roll_deg); xr, yr = np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y
    T = np.eye(4); T[:3, :3] = np.c_[xr, yr, z]; T[:3, 3] = p_cam
    return T


def board_ok(det, img):
    c = rc.detect(det, img) if img is not None else {}
    return c, len(c) >= 10 and rc.well_spread(c)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    he = json.loads((rc.RESULTS / "handeye_kinova.json").read_text())
    X = np.array(he["T_tool_cam"])
    board = rc.make_board(); det = rc.make_detector(board)
    Ks, Ds, _ = rc.load_intrinsics({"serial": SIDE_SERIAL, "resolution": [1280, 720]}, "auto")
    with RobotSession(program="side_calib") as s:
        s.phase("find_board")
        time.sleep(2.5)                                  # side camera auto-exposure
        c, ok = board_ok(det, s.side.latest()[0])
        if not ok:
            s.decide(f"board not seen well by the side camera ({len(c)} corners)"); s.claim("fail"); return
        o, i = rc.obj_img(board, c)
        _, rv, tv = rc.view_rms(o, i, Ks, Ds)
        centre_board = np.r_[o.reshape(-1, 3).mean(0), 1.0]
        centre = (calib.T_base_side() @ rt_to_T(rv, tv) @ centre_board)[:3]
        s.decide(f"board centre ~{np.round(centre, 3).tolist()} (old side calibration, rough)")
        # the wrist camera + hand-eye + FK give the board's pose in the base frame without the side calibration
        wi = json.loads((rc.RESULTS / "intrinsics_kinova_1280x720.json").read_text())
        Kw = np.array([[wi["fx"], 0, wi["cx"]], [0, wi["fy"], wi["cy"]], [0, 0, 1.0]]); Dw = np.zeros(5)

        def wrist_centre():
            time.sleep(1.5)
            cw, okw = board_ok(det, s.wrist.latest()[0])
            if len(cw) < 8:
                return None, len(cw)
            o, i = rc.obj_img(board, cw)
            _, rv, tv = rc.view_rms(o, i, Kw, Dw)
            T_bt = s.kin.fk(np.asarray(s.feedback()["q"], float))
            return (T_bt @ X @ rt_to_T(rv, tv) @ centre_board)[:3], len(cw)

        HOME0 = np.array([0.0, 15.0, 230.0, 0.0, 55.6, 90.0])
        c_w, n_w = wrist_centre()
        if c_w is None:
            s.phase("home0"); s.move_joint("to_home0", HOME0, speed="slow")
            c_w, n_w = wrist_centre()
        if c_w is None:                                  # overview: high above the rough centre, looking down
            for roll in range(0, 360, 30):
                q, ep, er = s.kin.ik(look_at(centre + np.r_[0, 0, 0.45], centre, roll) @ rc.inv(X), HOME0)
                if ep < 0.002 and er < 0.5:
                    try:
                        s.phase("overview"); s.move_joint("overview", q, speed="slow")
                    except Exception as e:
                        s.decide(f"overview refused: {str(e)[30:120]}"); continue
                    c_w, n_w = wrist_centre()
                    break
        if c_w is None:
            s.decide(f"the wrist camera did not find the board ({n_w} corners)"); s.claim("fail"); return
        s.decide(f"board centre {np.round(c_w, 3).tolist()} from the wrist camera ({n_w} corners); "
                 f"the side estimate was {1000 * np.linalg.norm(c_w - centre):.0f} mm off")
        centre = c_w
        HOME = np.array([0.0, 15.0, 230.0, 0.0, 55.6, 90.0])
        q_now = np.asarray(s.feedback()["q"], float)
        if np.abs((q_now - HOME + 180) % 360 - 180).max() > 2:
            s.phase("home"); s.move_joint("to_home", HOME, speed="slow")
        q_home = HOME
        from rlpj.robot.sim import LIMITS_DEG
        from rlpj.robot.kin import wrap180
        lo, hi = LIMITS_DEG[:, 0] + 5, LIMITS_DEG[:, 1] - 5

        def ik_ok(T_tool):
            best = None
            for seed in (q_home, q_now):
                q, ep, er = s.kin.ik(T_tool, seed)
                qw = q.copy(); qw[[1, 2, 4]] = wrap180(qw[[1, 2, 4]])
                if ep < 0.002 and er < 0.5 and np.all((qw >= lo) & (qw <= hi)):
                    dq = float(np.abs((q - q_home + 180) % 360 - 180).max())
                    if best is None or dq < best[0]:
                        best = (dq, q)
            return best
        cands = []
        for el, r in ((80, 0.30), (66, 0.34)):
            for az in range(0, 360, 30):
                a = np.radians(az)
                p = centre + r * np.r_[np.cos(np.radians(el)) * np.cos(a), np.cos(np.radians(el)) * np.sin(a), np.sin(np.radians(el))]
                best = None
                for roll in range(0, 360, 30):
                    b = ik_ok(look_at(p, centre, roll) @ rc.inv(X))
                    if b is not None and (best is None or b[0] < best[0]):
                        best = (b[0], b[1], roll)
                if best is not None:
                    cands.append((best[0], el, az, best[1], best[2]))
        cands.sort(key=lambda c: c[0])
        s.decide(f"{len(cands)} of 24 views have an IK solution inside the joint limits")
        views = [(c[1], c[2], c[3], c[4]) for c in cands]
        saved = []
        for k, (el, az, q_view, roll) in enumerate(views):
            if len(saved) >= N_TARGET:
                break
            best = (0, q_view, roll)
            s.phase(f"view_{k:02d}")
            try:
                s.move_joint(f"view_{k:02d}", best[1], speed="slow")
            except Exception as e:                       # refused by the model check: skip this view
                s.decide(f"view {k} (el {el} az {az}): not reachable safely ({str(e)[30:200]})"); continue
            time.sleep(1.5)                              # still, fresh frames
            fw, fs = s.wrist.latest()[0], s.side.latest()[0]
            cw, okw = board_ok(det, fw); cs, oks = board_ok(det, fs)
            if not (okw and oks):
                s.decide(f"view {k}: board wrist {len(cw)} / side {len(cs)} corners, skipped"); continue
            import cv2
            n = len(saved)
            cv2.imwrite(str(OUT / f"{n:02d}_wrist.png"), fw); cv2.imwrite(str(OUT / f"{n:02d}_side.png"), fs)
            saved.append(dict(view=k, el=el, az=az, roll=best[2], tool_pose=[float(v) for v in s.feedback()["pose"]],
                              q=[float(v) for v in s.feedback()["q"]]))
            (OUT / "poses.json").write_text(json.dumps(saved, indent=1))
            s.decide(f"view {k}: saved pose {n} (wrist {len(cw)}, side {len(cs)} corners)")
        s.summary(side_calib=dict(saved=len(saved), captures=str(OUT)))
        s.claim("success" if len(saved) >= 6 else "fail")


if __name__ == "__main__":
    main()
