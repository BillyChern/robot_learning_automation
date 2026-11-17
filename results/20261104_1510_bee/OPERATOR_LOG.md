# Run 20261104_1510_bee: operator log

This file records every human intervention, every unplanned stop, and every system change made while the run was
going. The times come from `events.jsonl` and `orchestrator.log` (local time on the 4090). Details are in
`events.jsonl`, the agent attempt folders, and `notes/*.sent_*.md` (the notes sent to the agents).

Setup: bee_in_bowl task, text condition, kinova_a (192.168.2.9), side camera D435 935322070708. Run with
`rlpj --autonomous run-all`.

## Human interventions (the robot could not continue without a person)

| # | when (stopped -> resumed) | what happened | what the person did |
|---|---|---|---|
| 1 | 11-04 20:39 -> 21:32 | Reset attempt 60 put the bee at a temporary spot (0.30, 0.10), near the base and the curtain-side edge. The bee was lost there. Resets 41-46 failed with "bee not found", and the run stopped (autonomous: reset failures). | Put the bee back on the table. |
| 2 | 11-04 ~21:55 (no stop) | Resets 72-73: the bee stayed on a fingertip after release and the arm carried it around. | Took the bee off the gripper and put it on the table. |
| 3 | 11-04 22:01 -> 22:10 | The bee had been put down at (0.75, 0.03). That is beyond the straight-down reach (about 0.72 m), and the reset program's own limit was 0.62 m. The run stopped. | Moved the bee to (0.63, 0.04). |
| 4 | 11-04 23:29 -> 11-05 00:13 | Attempt 92: the bee was wedged between the fingers after release, and the corner-only stuck check read 0.000. The bee was carried through attempts 93-94 and fell to the floor. | Put the bee back on the table. |
| 5 | 11-05 02:43 -> 11:15 (8.5 h) | Attempt 119: the bee stuck to the outside of a fingertip after release at (0.36, 0.09). The wrist check read 0.123, under the agent's 0.15 threshold, and the squeeze test read empty. The bee fell off the curtain-side edge while the arm went home. | Put the bee back. Changed the bee so it sticks less to the gripper. |
| 6 | 11-06 02:38 -> 11-06 ~11:20 | The robot's Ethernet link went down during the first policy evaluation (eno1 NO-CARRIER, "No route to host" to 192.168.2.9). The operator (Claude) paused the run with Ctrl+C. | Rebooted the arm. `rlpj check` passed at 11:15. The arm came back in Retract and had to be put back at Home. |

Robot time lost to human-dependent stops: about 1 h (1-3), about 0.75 h (4), 8.5 h (5), and the night of 11-06 (6).

## System changes during the run (each loaded by a safe-point restart or a relaunch)

| when | change | why |
|---|---|---|
| 11-04 16:32-18:50 | Six restarts: per-run reflections, a 100-episode round, size priors for the bee and bowl, overhang rule removed, layout tolerance 0.04, bee min height 0.025 | Fixes to perception and the success checks found during the first episodes (see `restart` events). |
| 11-04 21:50 | `collection.discard_unsure: true`: an unsure collection episode is deleted, not counted, and redone. | Operator request. One episode was discarded this way (11-05 11:36). |
| 11-04 ~21:50 | Bee layout narrowed to x 0.38-0.56, y -0.20..0.08, radius >= 0.38. | Every reset that lost the bee had a target with x < 0.36, where the side camera barely sees it. |
| 11-04 22:11 | Operator notes (`runs/<run>/notes/<role>.md`, sent with the agent's next message). First note: measured reach (straight down to about 0.72 m at the table, 0.69 m at 0.22 m height). | The reset agent's 0.62 m limit was too strict. The agent fixed it (commit ffadd03). |
| 11-04 ~22:05 | Status page resends the frame every second. | Browsers showed no picture while nothing changed. |
| 11-05 00:13 | Orchestrator `gripper_check` after every robot program: share of bee-yellow pixels in the bottom 15 % of the wrist image above 0.10 means "on the gripper". The episode is skipped, the reset agent is told, and the run stops after 2 in a row (before the bee falls). | Incident 4. Measured: stuck 0.21-0.22, bee lying below at most 0.052, empty at most 0.027. |
| 11-05 11:15 | Robot library `release_zone`: opening the gripper on the held bee (reading 60-92) is refused unless the tool is at x 0.36-0.60, y -0.22..0.08 and at most 0.20 m above the table. Bee layout y max 0.06. Notes are sent before collection resumes. A bee held in the jaws (`gripper_final`) is not counted as stuck. | Incident 5. Table edges measured with the side camera: curtain side y +0.19..+0.25, with a gap of about 10 cm to the curtain. |
| 11-05 23:40 | Training keeps one checkpoint per 5000 steps (`keep_period` 5000; was 1000). | The 5090 disk was 95 % full (about 8.9 GB per checkpoint). |
| 11-06 02:30 | lora_v1 checkpoints 1000-4000, 6000-9000 and 11000-14000 deleted on the 5090. Kept: 5000, 10000, 14999. | Operator approved, to make room for a possible extension to 30k steps. |
| 11-06 02:38 | `hosts.policy_host: Worksation_5090`, `policy_server: 100.122.29.80`. | The 4090 GPU is shared with other users' jobs (18.7 of 24.5 GB in use), and the local policy server ran out of memory. |
| 11-06 02:38 | A restart during the learn phase now hands the agent the result of its last run. Learn-phase messages carry operator notes. | So a restart does not tell the agent to "convert the new episodes, train again". |
| 11-06 11:20 | Relaunched once the arm was at Home (a read-only watcher on the joints started the launch command). Learn note sent. | After the arm reboot (incident 6). |
| 11-06 11:19 | The orchestrator crashed while starting the remote policy server: in `a && b && nohup serve &` the whole list goes to the background and keeps the ssh channel open, so ssh timed out after 60 s and the error was not caught. Fix: `( setsid nohup serve ... & )`, `ssh -n`, and a failed start now returns an error instead of crashing (with a test). | No rollout had started, and the robot did not move. |

## Milestones

- 11-04 15:10: run created. 16:11: collection starts. 11-05 18:29: 100 successes (108 episodes: 100 success, 6 fail, 2 unsure).
- 11-05 18:29: dataset bee_v1 (100 episodes, 176,072 frames, 15 fps). 18:48 -> 11-06 02:27: lora_v1 training (pi0.5 LoRA, 15k steps, batch 16, constant LR 5e-5).
- 11-06 02:27-02:38: three evaluation tries of lora_v1/14999. None ran a rollout: the first was wrapped in ssh to the 5090, the second ran out of GPU memory on the 4090, and the third was cut off when the robot link went down.
- 11-06 11:30: evaluation 03 of lora_v1/14999 started (20 rollouts, target 90 %). The policy server runs on the 5090,
  reached at 100.122.29.80:8000 (6-7 ms).
- 11-06 11:34: first policy rollout on the robot (episode 108). It reached the bee at (0.548, -0.214), with the bee at
  (0.531, -0.174), and closed the gripper, which came back empty (reading 99.1). Fail. 124 action chunks were run
  through the model checks.
- 11-06 11:28: `rlpj transcripts` and `rlpj timeline` rebuilt (timeline: 305 runs, 884.5 min, 6.2 GB).
- 11-06 ~11:55: DAgger added (operator request). After every failed evaluation rollout, the execute program takes
  over from the policy's last pose. The session retraces the return path, and only then does recording start. The
  correction is recorded as a `dagger` episode. Notes were sent to the execute agent (start-from-anywhere mode) and
  the learn agent (`convert --modes collect,dagger`). Turned on by relaunching after evaluation 03, so that
  evaluation is not cut off. Commit on branch run-20261104-bee.
- 11-06 ~12:05: DAgger is now part of the pipeline for every run. The agent prompts include it (Agent 1: correction
  mode; Agent 3: the DAgger round). `rlpj vla convert` includes `dagger` episodes by default. The report and status
  show the correction counts per evaluation. This run's agents were told by notes, since their prompts were written
  at creation.
- 11-06 13:22: evaluation 03 stopped after 12 rollouts. The bee was lost after rollout 119: the policy held it by
  the tip and returned home with it, and the reset program's held-bee carry dropped it (attempt 209). The run was
  paused for the operator (incident 7).
- 11-06 13:30: found and fixed:
  1. Policy rollouts could never score a success, because the rollout always claims fail and the verdict needed a
     success claim. Now both independent checks must say yes. Rescored by hand: 117 a real success (the VLM missed
     it), 119 a fail (the geometry check gave a false yes).
  2. The gripper check only looked for yellow, and the bee has been wrapped in orange since 11-05. The range is now
     hue 0-40 (orange bee hanging 0.27, empty 0.006).
  3. Notes sent to the reset agent (put down a held bee straight below home, no carry; new colours), the execute
     agent (new colours) and the learn agent (corrected scoring).
- 11-06 13:31: the bee watcher was fooled by a small phantom blob (h 0.029 m, 82 points) and relaunched with no bee on the table. Paused again within a minute (no robot motion). The watcher now needs h >= 0.036 m, n >= 300 points, 3 readings in a row.
- 11-06 15:00: relaunched (real bee seen: 0.041 m tall, 1282 points). Evaluation 04 of lora_v1/14999 started with
  DAgger on. Rollout 1 (15:11) failed; the orchestrator's gripper check flagged the bee on the gripper.
- 11-06 15:16 -> 18:03: hang. The reset agent's test run (attempt 220) looped at full CPU after "bowl centre median"
  for 2 h 45 min. The arm was idle at Home. Ctrl+C did not stop it (pure-Python loop), so the operator (Claude) killed
  it with SIGKILL after checking the arm was still. Fix: `collection.attempt_timeout_s: 1200` (SIGINT -> SIGTERM ->
  SIGKILL), loaded by a safe-point restart. The reset agent was told to bound its loops.
- 11-06 18:05 -> 21:48: evaluation 04 of lora_v1/14999 with DAgger: 0/20, 18 corrections (14 successful).
  21:50: bee_v2 (collect + DAgger). 22:10: training lora_h30 (horizon 30, from lora_v1, 12k steps).
- 11-06 22:30 -> 11-07 01:20: two orchestrator bugs.
  1. The new 20-minute robot time limit also applied to GPU jobs, and it killed lora_h30 at 22:30. The agent
     relaunched it detached.
  2. The agent's wait_train.sh polls were counted as training runs, used up the budget of 6 by 23:50, and sent the
     pipeline back to collecting (round 2, 11 successes by 01:15).
  Fix: the limit applies to robot runs only, and only commands that start a training count (tests added). A
  `runs/<run>/PHASE` file redirects run-all at the next restart: set to learn. lora_h30 kept training detached
  (4.8k/12k at 01:16).
- 11-07 05:07 -> 08:45: evaluation 05 of lora_h30/11999 (bee_v2 = demonstrations + 14 corrections): 0/20, with more
  corrections (36 in total, 28 successful). 08:48 bee_v3. 09:13 lora_v3 (sin/cos for the wrapping wrist joints 4/6).
  12:11 screening 06 of lora_v3/4500: 2 rollouts, then stopped.
- 11-07 12:35: arm fault (incident 8). Reset attempt 285 found the bee on the gripper at the start, with the tool
  horizontal after the policy rollout. It lowered straight down over the bowl and the arm faulted at 88 % of a
  checked segment: joint 5 at 110.5 deg, tracking 0.03 deg, so not a joint-limit fault, most likely a contact or
  torque fault. Afterwards joint 5 drifted to 128.8 deg, past its limit. Resets 286-292 failed at the start check;
  the run stopped at 12:45. The agent's fix (fed92fa: turn the tool down while high) is untested.
- 11-07 between 12:25 and 13:55: the side camera moved (about 140 px in its image). Someone also cleared the arm
  fault (the state was READY at 13:55). Training lora_v4 (bee_v3, gripper-only state) has run on the 5090 since 12:49.
- 11-07 14:19: the arm is in Kinova Retract ([360, 340, 214, 360, 310, 90]), not Home. The side camera is panned
  about 112-141 px and tilted about 16-20 px from its view before 12:25. New tool `python -m rlpj.robot.side_align`
  (reference runs/calib/side_reference.jpg, attempt 284) shows the overlay and which way to turn on the status page
  (8091), to put the camera back without recalibrating. It was started for 40 min at 14:21.
- 11-07 ~14:30: cause of incident 8, found by the operator: the arm had grasped the curtain. Evaluation rollout 209
  reached tool y 0.295 at z 0.44 with the gripper closing (reading 65); the curtain hangs at y +0.31..+0.37. The
  policy learned the move from the demonstrations: the execute program's joint moves between Home and look_bowl
  swing to tool y 0.29-0.30, in 46 of 119 episodes. Pulling the curtain probably also moved the side camera.
  Fix: work-cell keep-in box in the shared model check (`robot.keep_in` y -0.45..+0.30 for every gripper point,
  in every program and every policy chunk). Rechecking saved plans: execute 98 of 1500 would be refused (home,
  look_bowl), reset 0. Notes were sent to the execute agent (reroute) and the learn agent (data and expectations).
  The operator is re-aligning the camera with side_align and putting the arm at Home.
- 11-07 ~16:20: the 4090 rebooted, losing the tmux session, orchestrator and status page; state was saved. lora_v4
  kept training on the 5090 (6.2k of 12k).
- 11-07 ~16:40: side camera recalibrated by the operator (`wristcalib.py side-pose`, 5 poses, spread 1.2 mm / 0.08 deg).
  New T_base_side translation (1.1788, 0.0442, 0.6054), was (1.1776, 0.0561, 0.6028) on 11-04. Old file backed up as
  side_pose_935322070708_before_20261107.json. The check (`move_to_check_pose.py`) could not open the side camera:
  Chrome's video-capture service held /dev/video0. tmux session `rlpj` and the status page were recreated.
- 11-07 ~16:45: side camera recalibrated a third time (poses spread over the work area, nobody in view). Check PASS:
  sideways +4.2 mm, table -17.4 mm. Earlier tries failed at +11.0 and +14.4 mm, with similar poses and a person in
  the camera view. Drift test: < 0.1 px per minute (mount solid). rlpj check passes. The arm is at the standard Home.
  New alignment reference: runs/calib/side_reference.jpg (old one kept as side_reference_before_20261107.jpg).
- 11-07 16:52: hosts.policy_host set to local (the 4090 GPU is free since the reboot) so the policy can be tested
  while lora_v4 trains on the 5090. Learn agent asked to screen lora_v4/7000 now (5 rollouts, DAgger on).
- 11-07 16:55: the orchestrator crashed on the learn agent's first turn after the relaunch: its ssh link to the 5090
  (Tailscale, no keepalives) dropped after 300 s ("client_loop: send disconnect: Broken pipe", claude exit 255), and
  the runner's AgentError was not caught. Fix: ssh keepalives (ServerAliveInterval 30) for agent turns and remote
  jobs, and up to 2 retries of a turn that failed at the ssh level (tests). The learn note was sent again.
- 11-07 ~17:10: operator decisions: target success rate 70 % (was 90 %); full fine-tuning only, no LoRA; at least
  30k steps per training run. Enforced by the orchestrator (`training.full_finetune_only`, `training.min_train_steps`:
  other training commands are refused and not counted) and stated in the learn prompt. lora_v4 was stopped at about
  7.3k steps (checkpoints 5000 and 7000 kept). openpi JAX full fine-tuning needs more than 70 GB; the PyTorch bf16
  trainer is being set up on the 5090 (32 GB).
- 11-07 ~17:20: operator: keep collecting while training. PHASE=collect at the pending safe-point restart (round 2
  resumes at 12/100). The lora_v4/7000 screen was dropped. The 5090 has torch 2.7.1+cu128 (cu126 has no sm_120
  kernels), the transformers patch, and bf16 PyTorch base weights; full fine-tuning runs through openpi_entry
  train-pytorch (UV_NO_SYNC=1 so convert, train and serve keep that venv). Memory test (batch 2, 40 steps) started.
- 11-07 17:29-17:40: full fine-tuning made to work on the 5090 (32 GB).
  1. bf16 AdamW ran out of memory at the first optimizer step (model 7.5 GB, after backward 14.8 GB, plus about
     15 GB of AdamW state).
  2. Switched to bitsandbytes 0.50.2 AdamW8bit, which trains every weight with 8-bit optimizer state: peak
     21.4 GB at batch 2 (0.55 s/step) and 23.1 GB at batch 8 (1.37 s/step).
  3. The operator's cleanup freed about 120 GB (135 GB free).
  At 17:40 the run full_v1 started: pi05_kinova_full (PyTorch, bf16, gradient checkpointing, AdamW8bit, no EMA),
  bee_v3, batch 8, 30k steps, save every 1000 (newest whole, 5000 multiples as weights only), detached; ETA about
  05:10. Log: logs/train_full_v1.log. Meanwhile the robot collects round 2 (PHASE=collect).
- 11-07 17:42: the 5090 is not reliable for heavy GPU work. It lost power under sustained ~400 W GPU load at 12:06,
  16:50 and 17:35, with no shutdown logged (~/power_logs). The mitigation in ~/power-mitigation (clock cap 2200 MHz,
  CPU caps) is not installed; only the 400 W GPU cap is. Operator: train on the 4090; the 5090 keeps the agents and
  the VLM judge. The 4090's last session also ended without a shutdown (16:18).
- 11-07 17:55: the 4090's /home partition filled (my copy of weights and data went to /home, not to / with 145 GB),
  and the orchestrator crashed writing a log. The run folder, datasets and checkpoints moved to the root partition
  (/data/sonic/rlpj, symlinked from runs/); /home is back to 50 GB free. Agent workspaces now keep the symlinked
  path (it must match the 5090's path for the sync). Restarted 18:06, collection continues.
- 11-07 18:13: full_v1 on the 4090 (scripts/train_guard.sh in tmux window train4090: resumes after a crash):
  pi05_kinova_full, PyTorch bf16 + 8-bit AdamW, bee_v3, batch 4 (4090: 22 GB peak, about 1 step/s), 30k steps,
  checkpoints every 5000 (newest whole, 10k multiples weights only), ETA about 02:20. hosts.train_host local; the
  policy is served on the 5090 while the 4090 trains. Verdict reason text fixed (outcomes were correct).
- 11-08 01:35: operator approved: deleted checkpoints/lora_v4 (6 GB); cleanup_after_eval.sh (tmux cleanup) deletes full_v1/10000 and 20000 after the first full_v1 eval.
- 11-08 02:28: full_v1 finished (/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000). after_training.sh: policy_host local, learn note, PHASE=learn.
- 11-08 04:56: cleanup_after_eval: deleted full_v1/10000
- 11-08 04:56: cleanup_after_eval: deleted full_v1/20000
- 11-08 08:12: root partition full (the run was on it; full_v4 checkpoints + bee_v4 + attempt videos). full_v4 died
  at step 9964 while saving step 10000, and the orchestrator crashed writing a log. Nothing ran 08:12-10:40.
- 11-08 ~10:00: operator: deleted all V2D challenge files on both machines (4090: ~/v2d-track2 274 GB,
  ~/v2d-morphometric, /var/tmp/v2d-tier3-recovery, its Claude project data and cron entry; 5090: ~/v2d-track2
  784 GB, 10 finished V2D Claude jobs, its Claude project data, 9 @reboot cron entries and their 4 running
  processes). The run folder moved back to /home on the 4090 (verified byte for byte); the /data copy was deleted.
- 11-08 10:17: full_v4 resumed from full_v4/5000 (train_guard.sh, RLPJ_CONFIG pi05_kinova_full_v4 + agent 3's
  recipe, batch 4, 30k steps); policy_host back to the 5090 during training; PHASE collect (round 2 at 82/100);
  after_training.sh full_v4 will switch to learn and evaluation.
- 11-08 13:02: full_v1 0/20 diagnosed: it replays the demos' look_bowl swing and is stopped by keep-in at y~0.23. Note to agent 3: rollout_look.py (scripted look pose, then policy), re-evaluate full_v1 now on the 5090 (weights copied there), bee_v5 = episodes trimmed to start at approach. Ended agent 3's 10 h wait so it reads the note.
- 11-08 13:20: eval 09 (full_v1 from the look pose) stalled: the 5090 server crashed on the shim's empty params/ (taken as JAX). Note to agent 3 with the fix; orchestrator detects a dead remote server early (RESTART).
- 11-08 13:57: full_v1 from the look pose (ep 341) went straight to the bee and was refused at the 8 mm tip floor (it descended 7 mm below the demos' grasp height). serve.sh accepts PyTorch checkpoints (both machines); rollout tip pre-check 8 mm. Agent 3: hold at the floor, take the gripper from the whole chunk, then 20 rollouts.
- 11-08 14:26: ep 345 (full_v1 from the look pose) grasped the bee and was aborted by rollout_floor's hold counter; fixed (learn 5d9ef6d). Offline test: full_v1 steers by the images (swap -> other bee); remaining weakness = 37 % idle frames (2.9 s at the look pose). Note to agent 3: bee_v5 drops idle frames.
- 11-08 14:51: root cause: wrist-solution mismatch (106/112 bee_v3 demos flipped at the look pose; rollouts are normal). User approved: full_v4 stopped at 21.2k (20000 whole); bee_v5 (94 normal-wrist collect episodes, from the look pose, idle frames dropped, 87k frames); full_v5 from full_v1 started 14:48 (ETA ~22:55); collection round 3 after eval 12.
- 11-08 16:54: eval finished; PHASE=collect, RESTART (round 3 while full_v5 trains).
- 11-08 21:37: agent 3's 12 h wait for full_v5 (attempt 34, since 16:54) blocked the pending PHASE=collect restart; robot idle 4.7 h. Ended the wait; round 3 starts until full_v5 finishes.
- 11-08 21:50: at the user's request, stopped the idle full_v1 policy server left on the 5090 after eval 12 (8.2 GB freed).
- 11-08 22:59: full_v5 finished (/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v5/30000). after_training.sh: policy_host local, learn note, PHASE=learn.
- 11-08 23:14: eval 13 ep 388: full_v5 grasped the bee (reading 73.2) and then every chunk was refused at the 8 mm tip floor (closed tips are modelled 9 mm too low) until the time limit. rollout_floor now uses -2 mm with the gripper closed, like execute.py's lift; from the next rollout on (both machines).
- 11-08 23:18: eval 13 ep 390 stopped after 3 chunks: plan start 0.75 deg from the measured joints (> 0.5 deg session check; full_v5 moves faster, so the arm was still settling during inference). rollout_floor now re-reads the joints after inference. ep 389: joint 6 at 185.6 deg (wraps to -174 in the state) at yaw 165; the policy crept 2 mm per chunk for 2 min.
- 11-08 23:34: eval 13 ep 394: full_v5 grasped (74.6), lifted to 0.22 m and headed for the bowl, then a 1.04 deg tracking deviation at a chunk start stopped it. rollout_floor now eases chunks (rollout_smooth, 0.4 s ramp, as rollout_v3). ep 392: closed on air 5 times (missed the bee).
- 11-09 00:13: eval 13 pattern: yaw 68-85 deg 3/3 grasped, yaw >= 105 deg 0/5 (389, 392, 396, 398, 399). Note to agent 3: bee_v6 with DAgger corrections if < 70 %.
- 11-09 10:14: full_v6 eval: 3 recent rollouts grasped, none completed. 421/422: grasp ~15 mm deeper than the demos, so with the gripper closed every lift was refused (421: 290 holds; 422: the pre-check used the 0.56 reading, the session the closed command). rollout_floor: closed gripper never goes lower than its current tips (and never below -2 mm); the pre-check uses the session's grip_frac/table z. 423 released over the bowl, then a tracking deviation; verdict unsure.
- 11-09 10:15: full_v6 eval runs in a dark room (side brightness 94 vs 137 for full_v5's eval and 136 median in training); ep 423 is a success (bee in the bowl) that the VLM rejected in the dark. Asked the user to turn on the lights.
- 11-09 11:43: full_v6 2/18 (+423) vs full_v5 7/20 under different conditions (dark, 5090 serving, eased chunks). Asked agent 3 for an A/B: re-evaluate full_v5 with the current rollout (10 rollouts) before choosing full_v7's base. 4090 GPU held by LEO jobs (17.5 GB).
- 11-09 12:13: found full_v6 trained on the 5090 (01:47-08:54; the 4090 attempt OOMed on LEO jobs). The user had chosen to wait for the 4090 at 01:45, but the operator never passed that to agent 3 (the session stalled). Note sent now: training on the 4090 only.
- 11-09 15:50: user raised max_policy_seconds 150 -> 240 (A/B retest: 3/8 full_v5 rollouts held the bee over the bowl at the time limit). Takes effect at the restart after the current eval (window restart_after_eval).
- 11-09 15:51: eval 15 (full_v5 A/B, 2/10) had already ended at 13:43; RESTART now for max_policy_seconds 240.
- 11-09 15:59: agent 3's full_v7 (bee_v7 133 eps, init full_v6) is queued on the 4090: scripts/train_when_free.sh waits for 23 GB free (LEO jobs hold ~18 GB) and resumes after interruptions. bee_v7 norm stats computed by the operator on the CPU. after_training.sh full_v7 armed.
- 11-09 15:59: removed the stale HF dataset caches of bee_v4/v5/v6 on / (32 GB, regenerable); bee_v7's kept.
- 11-09 18:14: eval 17 ep 490 ended at 65 s on 'release refused' (policy opened 22 cm from the bowl). rollout_floor now skips a refused open (the safeguard still refuses; the gripper stays closed) and lets the policy continue. 1 of ~60 rollouts since ep 388.
- 11-09 19:17: robot start fixes (library): the session now waits for the watchdog to report 'watchdog up' (restart once, else abort before motion); 11 runs had moved without a watchdog after it failed to connect (INVALID_USER_SESSION_ACCESS). kortex connect retry ~10 s with a new UDP session halfway (ep 505 failed to start). Tested offline + live dry.
- 11-09 19:44: reset attempt 566: first watchdog failed to connect, the session restarted it (check works), but the main program's feedback then failed with the same session error. Removed the 'new UDP session halfway' from the connect retry (may invalidate the other process's session); kept the ~10 s retry and the watchdog check.
- 11-09 21:14: full_v5 4/6 at 240 s (eval 18) vs full_v6 3/20 (eval 17): full_v6 regressed. Put the queued full_v7 (init full_v6 + bee_v7) on hold before it started; recipe decision to agent 3 after eval 18.
- 11-10 00:58: user installed ~/power-mitigation on the 5090 (400 W, GPU clock <= 2200 MHz, CPU 125/200 W) and approved training there. full_v7 (agent 3's recipe: bee_v8, init full_v5, 30k, batch 4, constant LR 1e-5) started 00:55 on the 5090 (tmux train5090, scripts/train_guard_5090.sh): clock 2175 MHz, 260-293 W (was ~400 W with spikes to 459 W), ~1 s/step, ETA ~09:20. 4090 waiter stopped. after_training_remote.sh (window after_train) copies the result back and switches to learn; collect_after_eval switches to collection after eval 19.
- 11-10 02:12: eval 19 finished; PHASE=collect + RESTART (demos while full_v7 trains on the 5090).
- 11-10 08:57: full_v7 finished on Worksation_5090 (/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v7/30000); weights copied; note to agent 3; PHASE=learn.
- 11-10 10:41: run ended at 10:17 (bee lost after ep 624/DAgger 625; agent 3 gave up). Bee watcher started (tmux bee_watch): side camera once a minute; 3 detections then 120 s, then relaunch with PHASE=learn.
- 11-10 11:39: while the run waits for the bee: bee_v9 = bee_v8 (113) + 64 new clean normal-wrist demos (ep 560-623) = 177 episodes; full_v8 = full_v7's recipe (init full_v5, LR 1e-5 const, 30k, batch 4) on the power-capped 5090, using idle GPU time. Prep script in the operator's job dir.
- 11-10 12:09: removed the 4090's stale HF dataset caches for bee_v7/v8/v9 on / (26 GB, regenerable; training runs on the 5090).
- 11-10 12:13: full_v8 started 12:11 on the 5090 (tmux train5090_v8): bee_v9 (177 eps, 164,743 frames), init full_v5, LR 1e-5 const, 30k, batch 4; ~1.05 it/s, 2170 MHz, ~286 W; ETA ~20:10. (The prep chain's own process check had matched itself; started by hand.) after_training_remote full_v8 armed (window after_train_v8).
- 11-10 13:03: bee back on the table (side camera); relaunched run-all with PHASE=learn (bee watcher).
- 11-10 13:05: relaunch at 13:03 crashed: the agents' claude CLI on the 5090 says 'Login expired'. Asked the user to /login there; login watcher (tmux login_watch) relaunches run-all when it works.
- 11-10 15:40: agents' claude login on the 5090 renewed by the user; relaunched run-all (phase learn: evaluate full_v7).
- 11-10 15:43: user renewed the claude login on the 5090 (15:40); run relaunched, eval 22 (full_v7) running. Library: release safeguard now also refuses an open outside the zone when the wrist image shows the bee on the fingers, and works before the first gripper command (the 10:10 drop).
- 11-10 19:39: eval 22: full_v7/30000 16/20 (80%), target 70% met; run-all reached report (REPORT.md) and stopped. full_v8 still training on the 5090 (~94%); its after_training_remote handoff will set PHASE=learn for the next relaunch.
- 11-10 20:14: full_v8 finished on Worksation_5090 (/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v8/30000); weights copied; note to agent 3; PHASE=learn.
- 11-10 23:32: user: test full_v8 for the final success rate. Relaunching run-all (PHASE=learn, stale RESTART removed); note to agent 3: eval full_v8 exactly as eval 22, no further training.
- 11-11 03:09: eval 23: full_v8/30000 11/20 (55%, 1 unsure) vs full_v7/30000 16/20 (80%). Agent 3 done (target met by full_v7); run-all reached report and stopped. Best checkpoint: checkpoints/full_v7/30000.
- 11-14 14:10: (a lab member) reflect_after_every_run off, review_every 50: per-run reflections were ~82% of the run's cost and changed a program once; repairs after failures unchanged. Applies to the next run.
