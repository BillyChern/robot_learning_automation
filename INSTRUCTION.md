## Experiment steps
**Note 1**: mostly the command is executed on 4090, will specify if it runs on 5090  
**Note 2**: mostly the robot by default is set to 192.168.2.9 (kinova_a), add --robot 192.168.2.10 for kinova_b

### Boot the kinova arms
1. wrist camera focus for Kinova
    ```
    cd ~/realsense_calib
    .venv/bin/python wristcalib.py focus
    ```

1. check calibration  
    **Note**: This command will move the arm, make sure you can kill the process by Ctrl+C if needed
    ```
    cd ~/CoRL_late-breaking/agentic_robotics_real/far_trials
    ~/.venvs/kinova_real/bin/python move_to_check_pose.py
    ```
    Make sure you see output like:
    ```
    == calibration check: kinova_a ==
        sideways error                  +4.9 mm   expected within +/-10 mm        PASS
        table height                   -16.4 mm   expected -17.0 +/- 5 mm         PASS
        RESULT: PASS - calibration still matches: OK to run trials
    ```
    if it fails unfortunately we have to redo calibration :(  
    check [CALIBRATION.md](CALIBRATION.md)

### Record demonstrations
1. record 5 or 10 human demonstrations (depends on task difficulty)
    ```
    cd ~/CoRL_late-breaking/kinova_realsense_recorder
    python3 record.py demo --task <task_name>
    (press enter to start/stop recording for an episode, press q to save all)
    ```

1. You can see the data in ```/home/r2r/CoRL_late-breaking/kinova_realsense_recorder/data```

### Agent-in-the-loop self correction
1. Run new seed (remember to setup the scene)
    ```
    (on 5090)

    cd ~/Documents/aaron/agentic_robotics/far_trials
    bash new_trial.sh [--robot ROBOT] [--skills SKILLS] task condition seed
    ```
    e.g: ```bash new_trial.sh cup_up_stacking baseline 1``` or  ```bash new_trial.sh --robot 192.168.2.10 --skills cup_down_stacking cup_up_stacking skill_transfer 2```  
    it will start new seed in a new session
1. After the first script finished, the session will ask you to run the command, mostly like this
    ```
    cd /home/r2r/Documents/aaron/agentic_robotics/far_runs/cup_up_stacking/baseline/seed_*/kinova_* && bash attempt.sh ~/.venvs/kinova_real/bin/python run.py
    ```
    **Note 1**: Crtl+C if the robot do something crazy, this will count as failed and the agent will keep correct the script  
    **Note 2**: record both the screen and the robot when it's doing the task (for materials on the website), we also want to record 5 suceess (or 5 trial) in a row (try vary the cup placement, color to prove for robustness)
1. After the seed success
    ```
    (In the session, enter "done", it will record the time to succeed, token usage, etc.)
    ```

    **Note**: For harder tasks like ```3cups_up_stacking```, 3 consecutive succeed will be counted as suceess. For new tasks, by default it will count as success in the first succeed, specify in ```~/CoRL_late-breaking/agentic_robotics_real/far_trials/task_rules.json``` if they require 3 consecutive succeed

### Build a skill library (for skill_transfer)
1. Needs 2 clean successful baseline seeds of the source task on the same arm (no human guidance, calibration valid)
1. In a new Claude Code session, give it the procedure
    ```
    (on 5090)

    cd ~/Documents/aaron/agentic_robotics
    claude
    > Follow far_trials/SKILL_LIBRARY_BUILD.md to build the library for <task> on <kinova_a|kinova_b> from seeds <k1> and <k2>
    ```
    It builds `far_skills/<task>/<robot>/`, lints it and copies it to 4090  
    You can see the all the skills in ```/home/r2r/CoRL_late-breaking/agentic_robotics_real/far_skills```
1. Then run it as a new seed (see "Run new seed" above), e.g.
    ```
    bash new_trial.sh --skills <source_task> <target_task> skill_transfer <seed>
    ```

### Shutdown the arm
1. Use joystick (press A) to retract the arm back to default pose 
1. Alternative: use Kinova Web App
    ```
    (on 4090, open with http://192.168.2.9 or http://192.168.2.10 on browser, username and password are both ```admin```)

    (on 5090)
    ssh -fN -D 1080 WorkStation_4090; pkill -x chrome; sleep 2; google-chrome --proxy-server="socks5://localhost:1080" [robot_ip] &
    ```
    **Note**: Change the robot ip to http://192.168.2.9 or http://192.168.2.10

### Troubleshooting
1. If the connection between arms is broken
    ```
    sudo journalctl -u kinova-net -n 20     # see what it reports
    sudo systemctl restart kinova-net       # re-sets the routes without touching an arm that answers
    ```
    if this still not fixing, prompt in this session
    ```
    (On 5090)

    cd ~/CoRL_late-breaking_backup
    claude --resume dd561906-12d9-4be5-a77d-1a7b314e422c
    ```
2. For harness, skill libraries, also use the above session


 cd /home/r2r/Documents/aaron/agentic_robotics/far_runs/battery_placement/baseline/seed_1/kinova_b && bash attempt.sh ~/.venvs/kinova_real/bin/python run.py