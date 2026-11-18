# drawer_rl design (execute / RL agent)

## Split: scripted vs learned
Scripted (reused from the reset agent's hardware-tested code, results/.../agents/reset/programs):
1. perceive the drawer on the table (side depth, rim rectangle); the pull (front) wall is the face whose outward
   normal is closest to -N (reset places it within +/-30 deg of facing the cabinet, faces are 90 deg apart);
2. top-down grasp of the front wall (wrist-camera wall refinement `refine_wall`), lift, carry;
3. lower in front of the slot: drawer axis along N, centre line t = T_C, bottom 2.1 cm above the table, back 3 cm
   in front of the front plane; side-depth measurement of the held drawer (t offset, yaw vs the slot axis).
Learned (one policy, two stages, stage flag in the observation):
- **insert** (held, vertical gripper): per step a slow straight line: advance along N, lateral t, height z, yaw.
  Until the tool axis reaches n = OFF - 0.036 (the keep-out limit for a vertical gripper: the drawer then sticks out
  ~4.5 cm) or 16 steps. Then release, lift (scripted).
- **push** (gripper closed, tilted 45 deg toward the cabinet, fingertips on the drawer front at h 0.05): per step
  advance, t, z (yaw ignored). Fingertip (model) up to OFF + 0.02 (keep-out allows the slot opening to 3 cm), 8 steps.
  Where to stop (flush, not too deep) and how to stay aligned is what it learns.
Then scripted retreat, home, verify.

## Observation (13, all scaled ~[-1, 1])
stage, (tool n - stage goal n)/0.1, cumulative dt/0.01, dz/0.01, dyaw/5deg, measurement a/0.01 (insert: measured
drawer t - T_C; push: estimated front protrusion), measurement b (insert: measured drawer yaw vs N /5deg), force
change since stage start along N, T, z (/10 N), last step's peak force change (/10 N), step/K, holding flag.

## Action (4, tanh in [-1, 1]) and prior
advance = 0.0075 (1 + a0) m (0..15 mm), dt = 4 mm a1 (push 3 mm), dz = 3 mm a2, dyaw = 2 deg a3 (insert only).
Cumulative bounds: |t| <= 25 mm, z in [-6, +12] mm, |yaw| <= 10 deg. Prior: a = [0.6, 0, 0, 0] (12 mm straight
along the nominal line, no correction). Each step is pre-checked in the model (rlpj.robot.sim.check); a refused
step is shrunk (halved up to 3x), else the stage ends.

## Reward (agrees with the orchestrator's check, shaped)
- per step: +10 * insertion progress (m, tool along N while holding; the push stage gets no per-step progress: the
  tool moving tells nothing about the drawer), -0.02 step cost, -0.1 * max(0, peak force - 20 N)/5 force penalty (sliding takes 22-28 N);
  no force cap at all (user, 22:30: neither the task's 25 N nor my own); force is only observed and penalised, -1 for a refused motion (stage ends), -2 if the drawer is dropped.
- terminal (last transition, after the verification at home): the orchestrator's own geometric check
  `rlpj.success.checks.drawer_rl(before, after)` run on my side-depth blobs: +5 if True; plus 2 * flush share
  - 2 * front share of the bottom-slot window (`slot_profile`), -1 if a drawer blob is still on the table.

## Algorithm: SAC (RLPD-style) with a prior-initialised actor
Off-policy, reuses every transition (train, eval and test episodes). Actor MLP 13-128-128-(mu, log_std), tanh;
twin critics with LayerNorm, target nets (tau 0.005), gamma 0.95, auto entropy (target -2), UTD: ~40 updates per
new transition (capped at 3000 per call / 200 s). Actor's last layer starts at mu = atanh(prior), std 0.25, so the
first episodes are the scripted prior plus exploration; a BC-to-prior term (weight 1.0 * exp(-episodes/30)) keeps
it there while data is scarce. Learner state (nets, optimisers, alpha) persists in rl/policy/learner_state.pt, the
numpy actor in rl/policy/actor.npz (execute.py: mean action in eval, sampled in train).

## No simulation
A MuJoCo contact model of a wooden drawer in a slot with unknown clearances, friction and the 1-cm
side-camera/wrist disagreement would need its own sim-to-real work; with a residual action space around a working
scripted prior, real data is the cheaper source. The model checks (sim.check) are used offline to keep every
action inside the keep-out/kinematic envelope (tools/offline_test.py).

## Log
- attempt 3 (first real episode, prior + exploration, commit 6393799): grasp 88.6, held-drawer measurement t err
  -0.015, yaw +5.8 deg (1228 pts); slid in 10 steps to tool n-OFF -0.0755, then 40.3 N jam (t drift +7 mm);
  verdict: drawer sticks out (front share 0.47). -> after a 40 N stop the insertion continues from the measured
  pose (up to 3 stops) instead of ending, so the policy can learn force-driven corrections.
- attempt 5 setup: own 40 N stop removed on the user's instruction (operator note after attempt 4); the 'force' status
  path remains only for a stop the session might still report.
- attempt 4: learner --check + one update on the real data (1 episode, 10 transitions, cuda on the robot machine, 5.7 s).
- attempt 5 (real, commit 1507c05): insertion reached n-OFF -0.036 in 15 steps (one 58.9 N rub at step 9, t err
  -0.013 measured); release opened to 0.40 (51 mm): the inner finger sat ~2 mm in front of the cabinet face inside
  the drawer, the 6 cm lift caught the rail above the slot (99.6 N, tracking stop) and moved the cabinet 6 cm.
  -> release opening 0.75 (~21 mm): inner finger ~1.7 cm in front of the face, also in the model.
- attempt 7 (real, b09672d, 50 N task stop back): narrow release worked (23 N on the lift); insertion reached in 14
  steps; push: 5 steps to tip n-OFF -0.0125, then 50.5 N (drawer at its stop or jammed); the 6 cm retreat tripped at
  51.1 N because it released the preload inside one segment (the stop watches the change). -> unload(): 1.5 mm slow
  back-off lines after every force stop (insert and push) before anything else; recover.py handles a tilted pusher.
- ORIENTATION (user 23:50): the pull wall (grey metal label frame + tab) must face out. Scripted step `orient()`
  (programs/pullface.py): grey-pixel share (HSV S<70) on each wall's outer face in the side image on the table
  (A = grasped robot-facing wall); if not conclusive (top >= 0.08 and >= 2x runner-up), the held drawer is turned
  180/90/270 deg at 0.20 m and scored again; if the pull is B/C/D the drawer is put down turned so the pull faces the
  robot and re-grasped. Thresholds are guesses until the first real views (crops orient_*.jpg). Terminal reward:
  +5 for the check only with a known orientation (A or turned), +2 when unknown.
- attempt 11: orient_probe.py found no drawer (no reset before non-episode programs; drawer still in the slot).
- extract.py removed (operator: resets are the reset agent's job; every real run now starts with a reset that also orients the drawer). orient() stays as a check/fallback.
- attempt 14 (real, 21b7505): grey-share face scoring failed (background on mis-projected faces: D 0.99 on the
  table). Crops show the pull clearly (frame + knob, knob ~4 cm high, sticks out ~1.5 cm) on wall B. -> pullface.
  detect_pull: grey (S<100, V>45) component inside the drawer's wood hull (lower 60 %), side left/right of the hull's
  lowest point; visible_lr maps left/right to walls. Validated offline on attempt 14's four crops (tools: /tmp script,
  reconstructed crop origins): table none, rot180 -> B, rot90 -> B, rot270 none. Views stop at the first detection.
  Push: insertion ok (15 steps); push reached tip +0.0045 (45 N at step 5); the retreat stopped at 50.5 N: the
  fingertips at h 0.05 hooked the knob. -> push height 0.072 (range 0.062-0.085, above the knob); retreat unloads
  and retries once after a force stop.
- 00:20 changes (operator): orientation guaranteed by the reset -> turning step removed; pull_check() before the grasp
  only stops the episode (no insertion, fail) when my detector positively sees the pull on another wall.
  attempt 16 (3ee1451): measured held-drawer yaw err -15 deg, the prior did not correct it, 3 jams at ~-0.06 m;
  release 0.75 read 86.9 (wall still held, wedged) and the lift hit 58 N. -> (1) obs-dependent prior v2
  (rlcommon.prior): half the remaining measured t / yaw error per step, learner fits the actor to it once
  (bc_fit) and regularises toward it; old actor files (prior v1) are ignored by execute.py. (2) release(): widest
  opening keeping the inner finger 12 mm in front of the plane, verify the reading, 2 mm shifts between tries,
  if still held: stay (session home := here), no lift. SGS-RL notes: notes/sgs_rl.md.
- attempt 17 (893e3e4): pull check found the pull on A (correct); insertion with the prior's t correction (+2.4 cm from
  t err -0.024) jammed at -0.06 (3 force stops) then a tracking stop; release worked on the 3rd try (0.46 -> 45.9);
  the lift was refused by the model (load-deflected measured pose, gripper body 6 mm from cabinet_front). ->
  back off 12 mm along -N after the verified release, then up from the measured pose; prior v3 drops the t correction.
- REWARD FIX (user 01:50), commit after attempt 20:
  * terminal bonus: geometry ok -> the orchestrator's vision judge (rlpj.success.vlm_judge, task success text
    includes orientation) on before/after side + after wrist; judged not successful -> -5 (a backwards drawer is never
    paid), else +5. The orient_state condition is gone. Bonus stored in the episode meta (terminal.bonus).
  * learner.py relabels the bonus from $RLPJ_RL_DIR/verdicts/<episode>.json: final success -> +5; fail that my
    geometry paid or whose verdict mentions backwards -> -5; other fail -> 0 (shaping terms kept). Tested on
    synthetic verdicts.
  * F_SOFT 20 -> 35 N (normal sliding 22-28 N is no longer penalised).
  Reward now: per step +10*progress(m) -0.02 -0.1*max(0,F-35)/5; refused -1; force/tracking stop -1/-2;
  terminal: bonus (+5/-5/0) + 2*flush - 2*front - 1 if a drawer is still on the table.
- attempt 20 (84d20e6): full flow; insertion reached; release 0.62 -> 62.4 at once; push reached +0.02 but the retreat
  hit 136 N: the tip at h 0.074 went over the front wall top and dragged the drawer ~2 cm out (sticks out, front
  0.51). -> push at h 0.05 (0.040-0.062), 3 cm beside the pull toward +t (knob + label frame are centred), t capped at
  -0.060 so the closed fingers stay clear of the slot-side keep-out box.
- Tuning 02:35 (training episodes 0-1 failed; ep 1: advance samples -0.98..+0.85 around mean 0.5 -> 0-3 mm steps,
  front still at n-OFF -0.067 after 16 steps, no push). Changes (one theme: exploration on the insertion budget):
  exploration std on the advance x0.3, every dimension capped at 0.4 pre-tanh (rlcommon.EXPLORE_*; corrections t/z/yaw
  keep their learned std); insertion advance floor 4 mm unless the last step peaked above F_SOFT (logged action =
  executed action); K_INS 16 -> 20. SAC stays off-policy, so the narrower behaviour noise needs no learner change.
  Offline: flow test incl. injected force stops, learner --check ok.
