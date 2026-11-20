## Calibration
**Note 1**: commands run on 4090. Only the side camera usually needs redoing: it breaks when the robot base or the camera moves
**Note 2**: one arm at a time. kinova_a: `--robot 192.168.2.9 --side 935322070708` (D435, since 2026-11-04; was 052622072135), kinova_b: `--robot 192.168.2.10 --side 935322070708`. Both now name the same camera, and results are saved by camera serial, so a side-pose run for one arm overwrites the other arm's `side_pose_935322070708.json`

### Recalibrate the side camera, automated (no hand-guiding)
Board taped flat on the table where the side camera sees it, arm READY (at Home or where the wrist camera sees the
board), nobody in the cell, run on the 4090:
```
.venv/bin/python scripts/auto_side_calib.py          # --dry first to check the views; --no-write to only report
```
The arm locates the board with the wrist camera (hand-eye + FK, so the old side calibration only has to be roughly
right), visits ~12 model-checked viewing poses, saves wrist + side images, and solves with `wristcalib.py`'s own maths
(backup of the old file first). It prints the spread and how far the camera moved; check `rlpj.robot.check` (table
height) afterwards. First used 2026-11-20 (12 poses, spread 11 mm / 0.8 deg: partial wrist views, ~3 mm on the mean).

### Recalibrate the side camera
1. fix the robot base and the camera so they cannot turn, then back up the current calibration
    ```
    cd ~/realsense_calib/results
    cp side_pose_935322070708.json side_pose_935322070708_before_$(date +%Y%m%d).json
    ```

1. tape the A4 board (`~/realsense_calib/board_a4.pdf`) flat on the table, where the side camera and the wrist camera both see it

1. run the calibration (it only reads the robot; you move the arm by joystick, or press one of the two buttons near its EEF and move manually)
    ```
    cd ~/realsense_calib
    DISPLAY=:1 .venv/bin/python wristcalib.py side-pose --robot 192.168.2.9 --side 935322070708
    ```
    - hold the wrist button, move until both panels are green, release, step out of the side camera's view, press SPACE
    - vary height (20-50 cm) and tilt, not only sideways; press U if a pose shows `<- outlier? press U`
    - press Q when `average moved` stays around 1 mm for a few poses (usually 10-15 poses)

1. remove the board, then check again ([INSTRUCTION.md](INSTRUCTION.md), "check calibration"; add `--robot 192.168.2.10` for kinova_b)
    ```
    cd ~/CoRL_late-breaking/agentic_robotics_real/far_trials
    ~/.venvs/kinova_real/bin/python move_to_check_pose.py --robot 192.168.2.9
    ```
    PASS: new trials use the new calibration automatically. Still FAIL: redo with more varied poses
    
### Troubleshooting
if this still not fixing, prompt in this session
```
(On 5090)

cd ~/CoRL_late-breaking_backup
claude --resume b334c8ce-dfed-41bf-b59f-c24af5cc93d0
```