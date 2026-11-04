"""Read-only preflight: python -m rlpj.robot.check [--no-robot] [--no-cameras] [--port 8090]
Calibration files, kin fit freshness, robot reachable (ping + Kortex), arm state, gripper, FK vs tool_pose, side
camera frames and table height, wrist stream, live-view port. Prints a PASS/FAIL table, exits 1 on any FAIL.
The orchestrator imports run() -> {"ok": bool, "checks": [{"name", "status", "detail"}]}."""
import argparse
import os
import socket
import subprocess
import sys
import time

import numpy as np

from rlpj import config
from . import calib
from .kin import Kin, fit_file

READY = "ARMSTATE_SERVOING_READY"
FIT_MAX_AGE_S = 2 * 86400
FK_MAX_MM = 6.0
TABLE_TOL_M = 0.008


def _workspace():
    try:
        lay = config.load()["task"].get("layout", {}) or {}
        rng = [v for v in lay.values() if isinstance(v, dict) and "x" in v and "y" in v]
        if rng:
            return ([min(v["x"][0] for v in rng) - 0.05, max(v["x"][1] for v in rng) + 0.05],
                    [min(v["y"][0] for v in rng) - 0.05, max(v["y"][1] for v in rng) + 0.05])
    except Exception:
        pass
    return [0.25, 0.70], [-0.40, 0.25]


def table_height(depth_m, workspace=None):
    """Dominant base-frame z of the side depth over the workspace (1 cm histogram mode within +-25 cm of the
    configured table height, refined by the median of the points within 1.5 cm of it); returns (z, n_points)."""
    K, dist = calib.side_intrinsics()
    P = calib.apply(calib.T_base_side(), calib.deproject(depth_m, K, dist, step=4)[0])
    (x0, x1), (y0, y1) = workspace or _workspace()
    tz = calib.table_z()
    m = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (np.abs(P[:, 2] - tz) < 0.25)
    if m.sum() < 200:
        raise RuntimeError(f"only {int(m.sum())} depth points within 25 cm of the table height over the workspace")
    z = P[m, 2]
    h, e = np.histogram(z, bins=np.arange(tz - 0.25, tz + 0.25 + 1e-9, 0.01))
    k = int(np.argmax(h))
    sel = np.abs(z - (e[k] + e[k + 1]) / 2) < 0.015
    return float(np.median(z[sel])), int(sel.sum())


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port)); return True
    except OSError:
        return False
    finally:
        s.close()


def run(robot=True, cameras=True, port=8090, log=None):
    """Run every check; nothing moves. Returns {"ok", "checks"}."""
    checks = []

    def add(name, status, detail=""):
        checks.append(dict(name=name, status=status, detail=str(detail)))
        if log:
            log(f"{name:<22} {status:<5} {detail}")

    try:
        r = config.robot()
        missing = [f for f in r["files"] if not os.path.exists(os.path.join(r["calibration_dir"], f))]
        add("calibration files", "FAIL" if missing else "PASS", f"missing {missing}" if missing else f"{len(r['files'])} files in {r['calibration_dir']}")
    except Exception as e:
        add("calibration files", "FAIL", repr(e)); r = {}
    try:
        p = fit_file()
        if not os.path.exists(p):
            add("kin fit", "FAIL", f"{p} missing: run python -m rlpj.robot.fit_kin")
        else:
            age = time.time() - os.path.getmtime(p)
            add("kin fit", "PASS" if age < FIT_MAX_AGE_S else "WARN", f"{age / 86400:.1f} days old ({p})")
    except Exception as e:
        add("kin fit", "FAIL", repr(e))

    rob = None
    if robot and r:
        ip = r["ip"]
        try:
            ok = subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True).returncode == 0
            add("ping", "PASS" if ok else "FAIL", ip)
        except Exception as e:
            add("ping", "FAIL", repr(e))
        try:
            from .kortex import Robot
            rob = Robot(ip, dry=True)
            add("kortex session", "PASS", f"{ip} TCP 10000 / UDP 10001")
        except Exception as e:
            add("kortex session", "FAIL", repr(e))
        if rob is not None:
            try:
                st = rob.arm_state()
                add("arm state", "PASS" if st == READY else "FAIL", st)
                fb = rob.feedback()
                g = float(fb["gripper"])
                add("gripper", "PASS" if np.isfinite(g) and 0 <= g <= 100 else "FAIL", f"reading {g:.1f}")
                mm = float(np.linalg.norm(Kin.load().fk(fb["q"])[:3, 3] - fb["pose"][:3]) * 1000)
                add("fk vs tool_pose", "PASS" if mm < FK_MAX_MM else "FAIL", f"{mm:.1f} mm (limit {FK_MAX_MM:.0f}), q {np.round(fb['q'], 1).tolist()}")
            except Exception as e:
                add("arm readings", "FAIL", repr(e))

    if cameras and r:
        side = None
        try:
            from .cams import SideCam
            side = SideCam()
            t0 = time.time()
            while side.latest()[0] is None and time.time() - t0 < 8:
                time.sleep(0.1)
            c, d, _ = side.latest()
            if c is None or d is None:
                add("side camera", "FAIL", "no colour/depth frame in 8 s")
            else:
                add("side camera", "PASS", f"colour {c.shape[1]}x{c.shape[0]}, depth valid {100 * float((d > 0).mean()):.0f}%")
                z, n = table_height(side.depth_median(5))
                tz = calib.table_z()
                ok = abs(z - tz) < TABLE_TOL_M
                add("table height", "PASS" if ok else "FAIL", f"depth {z:.4f} vs calib {tz:.4f} m ({1000 * (z - tz):+.1f} mm, {n} pts)"
                    + ("" if ok else "; side camera or base moved? recalibrate the side pose (CALIBRATION.md)"))
        except Exception as e:
            add("side camera", "FAIL", repr(e))
        finally:
            try:
                side and side.close()
            except Exception:
                pass
        wrist = None
        try:
            from .cams import WristCam
            wrist = WristCam()
            t0 = time.time()
            while wrist.latest()[0] is None and time.time() - t0 < 10:
                time.sleep(0.1)
            im = wrist.latest()[0]
            add("wrist camera", "PASS" if im is not None else "FAIL", f"{im.shape[1]}x{im.shape[0]} via {wrist.url}" if im is not None else f"no frame in 10 s from {wrist.url}")
        except Exception as e:
            add("wrist camera", "FAIL", repr(e))
        finally:
            try:
                wrist and wrist.close()
            except Exception:
                pass

    add(f"port {port}", "PASS" if port_free(port) else "FAIL", "free" if port_free(port) else "in use (another robot program running?)")
    if rob is not None:
        try:
            rob.close()
        except Exception:
            pass
    return dict(ok=all(c["status"] != "FAIL" for c in checks), checks=checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args(argv)
    res = run(robot=not a.no_robot, cameras=not a.no_cameras, port=a.port, log=print)
    print("RESULT:", "PASS" if res["ok"] else "FAIL")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
