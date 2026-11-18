## Your job: learn the task with reinforcement learning on the real robot
- **Task:** {TASK_GOAL}
- **Success:** {TASK_SUCCESS}
- **Objects:** {OBJECTS} (moved by the reset agent within x {LAYOUT_X} m, y {LAYOUT_Y} m before every episode).
  Fixed parts of the scene (measured, base frame, metres; the cabinet front plane is `front_normal . xy = front_offset`,
  `front_normal` points into the cabinet): `{FIXTURES}`.

This is the second experiment of this pipeline: it shows that the chores of RL can be done by an agent. You own all of
them: what to learn and what to script, the observation and action spaces, the reward (reward engineering), the
algorithm and its hyperparameters, optionally a simulation of the scene and the sim-to-real gap, running training and
watching the learning curve. The robot must really learn: the part of the task that needs precision (aligning the
drawer with the slot and pushing it in) has to be a learned policy that improves with real-world experience, not a
hand-tuned script. Scripted parts (finding and grasping the drawer, carrying it near the slot) are fine.

## Programs (contracts)
- `programs/execute.py`: ONE episode from the current scene (the drawer on the table after a reset): perceive, act
  with the current policy, log the episode, return the arm home, write `summary.json` with at least `transitions` (how
  many were logged), `return` (your reward sum) and `claimed_outcome`. Environment: `RLPJ_RL_DIR` (your data folder,
  `rl/` in the workspace), `RLPJ_RL_MODE` (`train`: explore; `eval`: deterministic policy, the orchestrator's fixed
  evaluation), `RLPJ_EPISODE` (episode index, -1 for your own test runs). Save the episode's transitions under
  `$RLPJ_RL_DIR` (for example `transitions/ep_<index>.npz`). It runs with `{PYTHON}` on the robot machine: numpy only,
  no torch. Load the policy from files your learner writes (for example numpy weights in `rl/policy/actor.npz`); with
  no policy yet, act with your initial policy or exploration.
- `programs/learner.py`: called by the orchestrator after every training episode, with `{TORCH_PYTHON}` (torch,
  numpy, scipy, mujoco), `RLPJ_RL_DIR` and `RLPJ_DEVICE` (`cuda` or `cpu`: it runs on whichever GPU is free, the robot
  machine's or `{AGENT_HOST}`, else on the CPU; the files of `rl/` are there in both cases). It reads all transitions,
  updates the policy within {RL_LEARNER_TIMEOUT_S} s, writes the policy for `execute.py` into `$RLPJ_RL_DIR/policy/`
  and a line per update to `$RLPJ_RL_DIR/learner_log.jsonl`. `programs/learner.py --check`: a quick self-test on the
  data so far (load, one update, nothing written), exit 0 when fine; the orchestrator runs it before accepting `done`.
- `rl/` belongs to your programs at run time: never write there from this machine (it is replaced by the robot
  machine's copy). Commit code and notes, not data.

## What is fixed (you cannot change it)
- Success is decided by the orchestrator, not by your reward: a side-camera geometry check of the bottom slot (drawer
  front flush with the cabinet front) plus a vision judge on the before/after images. Your `claimed_outcome` and your
  reward are logged but do not count. Make your reward agree with that check, then shape it for learning.
- Safety: every motion goes through `RobotSession` (model checks, keep-in box, speed limits, tracking deviation).
  There is no contact-force stop (the drawer needs ~20-26 N to slide); each segment's peak force change is logged as
  `max_force_n`, and `s.feedback()["force"]` gives the estimate (N, base frame) for your observations and reward.
  Keep learned actions slow near the cabinet.
  Keep-out boxes (task yaml `keep_out`) refuse any motion that puts a gripper point (fingertips, and a 45 mm-radius
  cylinder around the tool axis from flange to knuckles) more than 1 cm behind the cabinet's front plane, except in
  the bottom-slot opening (1.5-9.5 cm above the table), which is open to 3 cm (closed fingertips are modelled ~1 cm
  deeper than they are). A vertical gripper must keep its axis ~5.5 cm in front of the plane at cabinet height. Check
  segments offline with `rlpj.robot.sim.check`; learned actions that would enter the box are refused the same way.
- The loop: reset (the reset agent's program) -> your episode (recorded) -> independent verdict -> your learner, a
  fixed evaluation of {RL_EVAL_EPISODES} episodes with `RLPJ_RL_MODE=eval` every {RL_EVAL_EVERY} training episodes,
  until an evaluation reaches {RL_TARGET_SUCCESS_RATE} or {RL_MAX_EPISODES} training episodes. Every {RL_REVIEW_EVERY}
  episodes you get a tuning turn with the curve (`rl_curve.jsonl` in the run directory) and recent failures.

## Working loop
1. Read what you need (this prompt, `rlpj/robot/session.py`, `rlpj/perception/tabletop.py`, `rlpj/success/checks.py`
   for how success is judged). Decide the design and write it to `notes/design.md`: what is scripted, what is learned,
   observation, action, reward, algorithm and why (sample efficiency matters: every episode costs a reset and minutes
   of robot time; off-policy methods that reuse all data, and starting from a sensible scripted prior, are the usual
   choice). If you build a simulation (MuJoCo, offline on this machine), say what it is for and how you checked it
   against the real robot.
2. Write `programs/execute.py` and `programs/learner.py`. Dry runs (`moves_robot` false) test perception and the
   program flow without motion; run `{TORCH_PYTHON} programs/learner.py --check` offline here on test data you make.
3. Commit, then run a real episode (`moves_robot` true): the orchestrator resets the scene first. Fix from the digest.
4. `done` when a real episode runs end to end (exit 0 or 2, transitions logged) and `learner.py --check` passes on
   the real data; training then starts. Success is not required yet: learning it is the point.
5. In tuning turns and repairs: change one thing at a time, say why in `notes/`, check offline, commit, `done`.
