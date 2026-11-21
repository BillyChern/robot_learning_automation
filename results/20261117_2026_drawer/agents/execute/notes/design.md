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
- TUNING 03:08 (10 training episodes, 0 successes). Finding: insertion outcome is decided by the tool yaw the
  policy applied: reached the end in 3/3 episodes with dyaw +5.6..+9.8 deg, jammed at n-OFF ~-0.06 (forces mostly
  along -N, 30-40 N residual after unloading) in 7/7 with dyaw -5.5..-1.2. The held-drawer side-depth "yaw error"
  set those signs through the prior (v2/v3 corrected half of it per step), but it is not trustworthy: the wrist
  camera at the pre-grasp pose (new programs/wallfit.py, run offline on attempts 14-41) sees the wall within 2 deg
  of the jaws in every episode while the side measurement swings +-12 deg; t err is a steady -2 cm (partial view).
  Changes (learner + observation, one theme: stop learning from the bad measurement):
  * prior v4: straight 12 mm advance, no t/yaw correction; actors trained since v3 stay usable (MIN_ACTOR_VERSION);
    no BC re-fit on a prior change (it would wipe what was learned), one full 3000-update pass instead.
  * measured t/yaw no longer observed (obs 5, 6 = 0 in the insert stage, also in old data): they confounded the
    data (the action sign followed them), so the policy could not separate "positive yaw works" from "negative
    measurement".
  * actor update SAC -> AWAC (advantage-weighted regression on logged actions, Nair et al. 2020; critic unchanged,
    lambda 0.3, weight cap 20): offline on the 17 real episodes SAC drove the yaw to -0.55 (against the evidence,
    entropy collapsing), AWAC keeps to the data and continues positive yaw where it was taken.
  * BC-to-prior decay 30 -> 10 episodes; verdict files are named 000006.json (6 digits): relabel now finds them.
  Not changed: reward, exploration (yaw keeps its learned std ~0.3-0.4, so both signs are still tried).
- attempt 43 (22bde39, test after the tuning): the insertion reached its end with dyaw +10 (AWAC policy), release ok;
  the push jammed after ~1 cm (51 N), like every push after a reached insertion (eps 4, 5, 8). Insertions only pass
  with the tool turned +5.6..+10 deg, so the slot axis seems turned from the yaml normal, and the push went straight
  along the yaml normal. -> the push stage now pushes along N turned by the insertion's final tool yaw
  (PUSH_FRAME; pusher orientation, advance, lateral and unload directions all turned), clipped to +-10 deg.
- attempt 44 (b24aec7): insertion reached with dyaw +9.2 at only 2-3 N per step (24 N on the last), push along the
  turned axis still jammed at the first contact (43 / 51 N). The after image shows the drawer ~half out, far more than
  the tool position implies: the drawer meets a stop ~5-7 cm before flush whether held (jams at tool -0.06 with small
  yaw) or pushed. Cause not identified yet (candidates: an obstruction deep in the slot / drawer pitched in the slot /
  gripper body on the face). Added: side + wrist snapshots at the insertion and push ends (insert_end_*, push_end_*),
  and the back-off after release is now smaller than the inner finger's gap to the wall (12 mm could drag the
  drawer ~1 mm out).
- 14:50 resume (lights were off 04:30-05:17): side-image brightness now logged in summary.json (side_brightness, ~122 normal) to spot lighting failures; attempt 45 (diagnostic snapshots) never ran.
- 17:20 keep-out (slot opening only 2 cm deep, do not recess the drawer): PUSH_NMAX OFF+0.020 -> OFF+0.017 (model
  check accepts the tilted pusher up to +0.019, checked offline at push yaw 0/5/10 deg), so the last push step is no
  longer refused; terminal reward now also -2 x the slot window's "behind" share (recessed counts like sticking out);
  K_PUSH 8 -> 10 (the full push from 2.5 cm before the face to the limit is ~8.6 cm at <= 15 mm per step).
- URGENT 00:05 (episodes 11-23 all jam at tool n-OFF -0.047..-0.068; the free AWAC policy drifted to the action bounds:
  dz +0.3..+0.97 lifting, dt -0.8, yaw +-0.97; one arm fault). Data over all episodes: insertions reached their end in
  6/26, 5 of them with the tool turned >= +5 deg; jams sit at a fixed depth; ep 10's push broke free after a 47 N
  peak and then slid at 4-14 N (static friction close to the 50 N stop). Changes (anchor to what worked):
  * prior v5: 12 mm advance, yaw ramped to +9 deg (2 deg/step), dt = dz = 0.
  * executed action = prior + residual clipped to [0.25, 0.3, 0.15, 0.3] (rlcommon.anchor), in train AND eval.
  * insertion dz 1 mm per unit (was 3), cumulative z in [-3, +2] mm; held drawer 3 mm lower (BOTTOM_H 0.021 -> 0.018:
    top at ~9.2 cm under the 9.5 cm rail, bottom above the 1.5 cm floor).
  * push: after a force stop it unloads, re-reads the tip and tries again (up to 3), like the insertion; pushes at
    h 0.062 (0.057-0.068, above the knob, below the drawer top), 1.5 cm beside the pull (3 cm put the fingers into the
    slot-side keep-out once the push frame turns 9 deg: model refusal at +0.009).
  Offline: flow test 6/6 reach the insertion end and run the full push, no refusals; learner --check ok.
- attempt 70 (6e64d0a, eval mean, anchored prior): insertion jammed again at -0.06 with yaw +9.6 (3 stops, then a
  tracking abort). insert_end images: the gripper holds the front wall, the drawer still ~6 cm out, its side wall
  visible up to the cabinet face. Jams sit at a fixed depth for every yaw/dz tried, so a lateral offset is the next
  suspect: the grasp point along the wall comes from the side camera's partial rim (rect 0.096-0.12 of ~0.13, the
  hidden part is on the far side), so it is biased toward the camera (-t) and the drawer sits off-centre in the hand.
  -> find_drawer moves the rim centre along the pull wall away from the camera by half the missing width (<= 2 cm,
  summary objects.drawer.shift_w). (A wrist-image end-of-wall fit was tried offline: the 5 cm pre-grasp view does not
  show both ends.)
- 00:40 finding (all real episodes with a held-drawer measurement): with the drawer's -t edge (the side facing the
  side camera) giving its centre as t_err - t_span/2 + 0.065, insertions reached their end in 11/21 episodes while that
  estimate was -0.001..+0.011 (attempts 2-46) and in 0/16 since it moved to -0.005..-0.016 (attempts 48-71, after the
  pause). -> scripted lateral alignment before the learned steps: the insertion line t_line = T_C + (0.003 - edge),
  clipped to +-2 cm (summary held.t_line, held.meas.edge); the learned dt stays relative to that line.
- attempt 72 (89c6aea): with the lateral alignment (t_line -0.067, +2 cm) the insertion reached its end (2-24 N); the push moved the drawer ~1.5 cm then jammed 3x: it pushed at the tool t + 1.5 cm, i.e. ~2.4 cm off the drawer centre. -> push frame and push point now use the drawer centre (T_C + EDGE_TARGET + learned dt) + 1.5 cm.
- USER'S RULE 00:30 (straight first, then push): prior v6 has no yaw and the residual may not add yaw (RESIDUAL yaw 0),
  so the tool keeps the slot normal during the insertion and the push goes straight along the normal. Straightness is
  set before: (a) grasp yaw turned to the wall angle seen by the wrist camera at the pre-grasp pose (wallfit, 0.5-15
  deg; offline the jaws were within 2 deg of the wall in all but one episode), (b) lateral alignment iterated from
  the side image until the drawer's centre (camera-side edge + 6.5 cm) is within 3 mm of the slot window centre
  (T_C + 0.0045 = -0.0825; up to 3 moves; summary align, t_line). A jam: unload, back out 1.5 cm, re-measure and
  re-align (lateral steps <= 4 mm while the back is in the slot), try again (up to 3). PUSH_NMAX OFF+0.012 (at h 0.062
  the model refuses beyond ~+0.0125). Offline: flow test 4/4 complete, no refusals; learner --check ok.
- 2026-11-20 resume (base moved, side camera recalibrated, fixtures shifted in the yaml): T_C and PUSH_TMAX now come from the yaml slot_t (T_C = window centre - 0.0045 = -0.133, PUSH_TMAX = slot_t[1] - 0.015 = -0.106); no other absolute cabinet numbers in the code (OFF, N, keep-out from the yaml). Offline flow test with drawer targets from the yaml layout and the shifted keep-out boxes: 6/6 insertions and pushes pass the model check, including lateral alignment at the +-2.5 cm extremes.
- 16:35 new reset target area (drawer centre x 0.36-0.37, y -0.03..-0.01, +-5 cm; ~18 cm in front of the slot): no code change needed (drawer search, grasp and carry are relative to the perceived drawer and the yaml front plane). Offline flow test at the area and at its four +-5 cm corners with +-30 deg yaw: all complete with no model refusals.
- ep 24 (attempt 74): lateral alignment converged (+1.6 mm) but the straight (tool yaw 0) insertion jammed again at -0.07..-0.06 (3x 51 N with back-outs), then the wall stayed wedged in the jaws (release refused to lift: safe). Lateral offset is ruled out; the remaining consistent evidence is the yaw: insertions passed in 5/12 episodes with the tool at +5..+10 deg vs 1/14 otherwise -> the drawer sits ~7 deg turned in the jaws. HAND_YAW = +7 deg: constant tool yaw for the whole insertion (the drawer straight, no yaw change during it; push frame stays on the normal). A wrist-image side-wall yaw fit (wallfit.side_wall_angle) did not find the walls on the insert-end images offline; not used.
- ep 25 (c2f7fe7, HAND_YAW +7): the insertion reached its end (one jam + back-out + re-align, then free); the push jammed on contact 4x (50-57 N) at the drawer centre + 1.5 cm. All push stages so far: the centred ones slid to ~flush (attempts 7, 14 at h 0.05, 20 at h 0.072), every off-centre one (1.5-3 cm, attempts 43, 44, 72, ep 4, 5, 8, 25) jammed at first contact. -> PUSH_DT 0: push on the drawer centre line at h 0.062 (above the knob, so the retreat cannot hook it).
- ep 26 (dd7129a): insertion jammed 3x at n-OFF -0.056 (once reached -0.045), ended on force, and the push was skipped by the old 'force stop fails anyway' rule, though the side view showed the drawer nearly flush (flush 0.43, median_d 0.000). -> a jammed insertion still gets the push when the front is within reach (only a dropped drawer skips it).
- ep 27 (a127dc6): insertion reached its end (dt drifted +4 mm), the centred push SLID to the limit at 3-27 N (one 52 N blip), but the drawer ended 2.6 cm out (median_d -0.026): about the knob's protrusion -> the closed tips pushed the knob. -> pusher opened to 0.55 (~3.8 cm gap): the fingers straddle the knob and push the face on both sides (balanced, no off-centre torque), fingertip at the knob height 4.8 cm (4.3-5.4); model accepts the open pusher to +0.024, PUSH_NMAX stays +0.012. Also (operator 17:48): no learned t residual in the insertion or push (RESIDUAL t = 0): t stays on the aligned line; the policy learns advance and small z.
- ep 28 (bd8645c): insertion reached (2 jams, back-outs, re-aligned), the open pusher (straddling the knob) slid to the +0.012 limit (8-30 N, one 50.6 N retry), then the 5 cm retreat pulled 27 N and the drawer ended 1.7 cm out (was 2.6 cm with closed tips): the fingers caught the knob on the way back. -> retreat: back off 4 mm, open the gripper to 0.35 (~5.5 cm gap, model-checked) and only then retreat. Push limit stays +0.012 (the flow test refuses deeper steps at this height).
- tuning (18:10, operator: the retreat drags the knob back out; ep 27/28 tip at +0.011/+0.012 with ~1 N at the end, yet 2.6/1.7 cm out after the retreat): retreat = open to 0.35, back only to OFF-0.006 (model), UP 5 cm, then back (model-checked chain; old back-then-up chain as fallback). Diagnostic: summary slot_after_push = slot profile right after the push, before the retreat.
- attempt 84 (2c7a130): first alignment converged (1.5 cm move, measurement followed: -0.7 mm), then 3 jams at -0.06..-0.072; the re-aligns after each back-out measured a ~1.6 cm offset that did not follow the tool (drawer partly in the slot, side view unreliable) and walked the tool to the +2.5 cm clip -> jammed harder, too far out to push. -> after a jam: back out 1.5 cm and return to the aligned line, no lateral re-measure while the drawer is in the slot.
- attempt 85 (38a49c8): alignment ok (9 mm, converged), 3 jams at -0.062..-0.064 (back-outs also 40-44 N: the drawer is wedged), push skipped (front -0.070 beyond the old reach rule). ep 26 showed a near-flush drawer after such a jam, and the released pusher slid in ep 27/28. -> push also from fronts out to n-OFF -0.076 (pre-pose clipped at -0.085, model-checked offline with a jammed insertion stub).
- attempt 86 (666424a): insertion jammed at -0.062 (3x), the released push from -0.07 jammed at tip -0.023 (55 N x3); drawer ended 5 cm out. Attempts 84-86 all had a small visible drawer width (t_span 0.048-0.058) and the edge-based alignment moved the line 0.9-1.5 cm; the episodes whose insertion reached the end had t_span 0.063-0.087. -> alignment only trusted with t_span >= 0.07, else the tool (grasp centre) goes on the slot centre. slot_after_push with the arm in view is not usable (behind 0.64 = occlusion).
- ep 29 (ca98dea) + operator notes 22:35/23:18: both grasps missed after refine_wall shifts of +0.030/+0.037 (wrong way or too far). -> refine_wall max |shift| 0.012; the retry after a miss grasps the unrefined perception point. Retreat after the push: straight up first when the model allows it (at the push limit it does not: the upper cabinet_front box), else open 0.35, back to OFF-0.006 (model), up, back. Success bonus +10 (was +5; learner relabel too), failure for backwards stays -5. Offline: flow test 3/3 with the 22:12 slot geometry (slot_t [-0.087, -0.012]), learner --check ok.
- ep 30 (a0263e0): the capped refine (+0.032 rejected) left the unrefined point -> miss, and the re-perception after the miss failed (arm over the drawer). The wrist refine has said +0.02..+0.037 with strong steps in every recent episode (grasps of eps 24-28 used +0.003..+0.025 and held). -> grasp() tries up to 3 points along the wall normal without re-perceiving: the refined shift (now up to 4 cm), the unrefined point, +1.5 cm, +3 cm, -1.2 cm (skipping duplicates), logging grasp_tries. Offline: misses simulated on the first two closes, the flow completes with the model checks.
- ep 31 (5ae8227): grasp held at the first try (refined +0.033), the straight insertion reached its END at 1.5-2 N (33 N on the last step), release ok; then the move to the push pre-pose was refused (j3 past its limit -144.8: at the shifted slot the tilted pusher folds the arm more; offline j3 reached -144.2 at tilt 40). -> PUSH_TILT 40 -> 35 deg: offline max |j3| in the push 140.2 (4.6 deg margin), pushes reach the +0.012 limit, retreat ok. offline_test: TSPAN env sets the stub's visible width (0.05 -> nominal line).
- ep 32 (8eeb7f7): grasp first try (+0.033), insertion reached its end, push slid to +0.011 (peaks 26-37 N, then ~1-2 N: drawer in), but the retreat pulled 53 N (back 1.4 cm) and 42 N (up) and the final check had the drawer 0.65 cm out with front share 0.45 (> 0.3) -> fail. The fingers hook the pull tab / label plate. -> reactive retreat: 3 mm steps trying up, back, down-back; a step above RETREAT_F = 12 N is undone and the next direction tried; once 4 cm back or 5 cm up, a model-checked clear-out (back to OFF-0.04, up). Old chains stay as fallback. Not done yet: the operator's release check (drawer lifted with the gripper in ep 31) and the wrist-footprint grasp point.
- operator 00:22 (ep 32): (1) cancel the measured in-hand yaw before the insertion: if the held-drawer fit has >= 600 points and |yaw err| > 1 deg, the hand turns by -yaw_err (<= 8 deg, on top of HAND_YAW), then the lateral alignment re-measures (summary yaw_cancel). (2) retreat: open to 0.15 (~7 cm) before leaving, RETREAT_F 12 -> 10 N (steps above it are undone, next direction tried). Offline flow test (stub yaw err with 700 pts) and the non-dry retreat path pass.
- ep 33 (8068940): the new cancel_yaw line raised 'joint jump 3.7 deg' (IK branch flip) and ended the episode with the drawer in the hand. -> turn_safely(): straight line, else the rotation in 3 line steps, else a model-checked joint move, else skip (yaw_cancel_skipped). And main(): on any exception, put_down_if_held() lowers to 12 cm above the table at the current xy (if higher) and opens before the session goes home. Offline: forced joint-jump on cancel_yaw -> fallback ran, episode completed; put-down from 0.2 m opened at 0.154 m.
- operator 00:50: low force stop for approaches (Episode.low_force context: force_limit lowered, restored after): grasp final approach 20 N (a stop = a miss: back up, next candidate), descent to the insertion start 25 N, push pre-pose approach 20 N, retreat steps 20 N; the insertion and push steps keep the 65 N cap. 01:08: cart_line rotation sampling fixed in the library; the yaw cancel stays (with its fallbacks).
- operator 01:50/01:55 (eps 35, 36): (1) wrist look-down before the grasp (wrist_footprint, after the reset agent's code): tool down 0.45 m over the side estimate + (0.010, 0.058), wood pixels projected onto the rim plane, the drawer blob's OBB (re-centred while it touches the image border, up to 3 looks) replaces the side-camera outline; the front wall and the grasp point along it come from it, so the drawer sits centred in the hand and the tool-on-slot-centre fallback (t_span too small) is right. (2) push jam rule: a step above 35 N whose measured tip advance is < 3 mm, twice in a row, ends the push ('jam'). Not done: backing the drawer out and re-inserting after a push jam (the pusher cannot pull), wrist view of the slot edges. Offline: flow tests pass; the non-dry path runs the look-down move through the model check (no image offline).
