"""Automated side-camera calibration: no hand-guiding, no key presses.

    .venv/bin/python scripts/auto_side_calib.py [--dry] [--no-write]

Needs: the A4 ChArUco board taped flat on the table (where the side camera sees it), the arm READY anywhere the wrist
camera can see the board or at Home, nobody in the cell. Steps: back up results/side_pose_<serial>.json, run
scripts/side_calib_capture.py through run_attempt (the arm finds the board with the wrist camera, then visits ~12
model-checked viewing poses and saves wrist + side images), then scripts/side_calib_solve.py --write (wristcalib.py's
own maths: T_base_side = T_base_tool @ T_tool_cam @ T_wristcam_board @ inv(T_side_board), averaged).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.path.expanduser("~/realsense_calib/results"))
SIDE_SERIAL = "935322070708"


def main():
    dry, write = "--dry" in sys.argv, "--no-write" not in sys.argv
    sys.path.insert(0, str(ROOT))
    from rlpj.orchestrator.attempts import run_attempt
    work = ROOT / "runs/drawers_setup/sidecal"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "scripts/side_calib_capture.py", work / "side_calib_capture.py")
    name = time.strftime("auto_%Y%m%d_%H%M%S") + ("_dry" if dry else "")
    r = run_attempt(str(work), "python side_calib_capture.py", str(work / name), not dry,
                    extra_env={"RLPJ_TASK": os.environ.get("RLPJ_TASK", "drawer_rl")}, timeout_s=2400)
    print("capture:", r.exit_code, r.outcome)
    if dry or r.outcome != "success":
        return 0 if dry else 1
    src = RESULTS / f"side_pose_{SIDE_SERIAL}.json"
    if write and src.exists():
        shutil.copy(src, RESULTS / f"side_pose_{SIDE_SERIAL}_before_{time.strftime('%Y%m%d_%H%M%S')}.json")
    cmd = [sys.executable, str(ROOT / "scripts/side_calib_solve.py"), str(work / name / "captures")] + (["--write"] if write else [])
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
