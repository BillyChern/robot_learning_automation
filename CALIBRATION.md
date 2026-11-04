## Calibration
**Note 1**: commands run on 4090. Only the side camera usually needs redoing: it breaks when the robot base or the camera moves
**Note 2**: one arm at a time. kinova_a: `--robot 192.168.2.9 --side 052622072135`, kinova_b: `--robot 192.168.2.10 --side 935322070708`

### Recalibrate the side camera
1. fix the robot base and the camera so they cannot turn, then back up the current calibration
    ```
    cd ~/realsense_calib/results
    cp side_pose_052622072135.json side_pose_052622072135_before_$(date +%Y%m%d).json    # kinova_b: 935322070708
    ```

1. tape the A4 board (`~/realsense_calib/board_a4.pdf`) flat on the table, where the side camera and the wrist camera both see it

1. run the calibration (it only reads the robot; you move the arm by joystick, or press one of the two buttons near its EEF and move manually)
    ```
    cd ~/realsense_calib
    DISPLAY=:1 .venv/bin/python wristcalib.py side-pose --robot 192.168.2.9 --side 052622072135
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