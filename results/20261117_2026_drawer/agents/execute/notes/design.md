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
