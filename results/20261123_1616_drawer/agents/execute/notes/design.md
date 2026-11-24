# drawer_lying: design (execute agent)

## Measured on the real scene (attempts 1-4)
- Drawer top face 72.3 x 113 mm (wrist), height ~91.5 mm (side depth). Opening 75.7 x 115 mm: clearance ~3.4 mm and ~2 mm.
- The pull is a flat label frame plus a curled tab of about 12 x 12 mm, about 2 cm from the drawer's centre line.
- The measured opening is at (0.4519, 0.0029), yaw 18.3 deg. The task yaml says (0.4469, -0.011), yaw 21.2 deg. The side
  camera's cabinet outline (long axis ~17.6 deg) agrees with the wrist, and wrist and side agree on the drawer within
  1 mm, so the program aligns to the measured rim. The yaml still defines the orchestrator's check window and keep-out
  boxes, which I cannot change. Unverified: whether the check window still reads a correct insertion as success.

## Scripted
Side depth finds the drawer. Wrist look-downs from 0.30 m measure the rim (before anything is held) and the drawer and
tab. Pinch the tab with the jaws along the drawer's long axis (20 N approach limit, reading check), lift, turn so the
long side lies along the opening and the tab points to the near end wall (keeps the fingers away from the divider),
carry, and lower to 8 mm above the rim.

## Learned
- Action (4, bounded residuals on a straight-down prior): corrections along a, b, yaw, and the descent step.
  2 align steps (3 mm, 1.5 deg per unit), then up to 18 descent steps (1 mm, 0.5 deg per unit; 3.5-10.5 mm down).
  Accumulated corrections stay within 8 mm and 4 deg.
- Observation (13): phase, inserted fraction, applied corrections, force change in the opening frame, last peak force,
  step fraction, pinch point on the drawer, contacts so far.
- Reward: 30 per metre of descent, small step cost, force penalty above 15 N, -0.5 per force stop, -1 per refused step.
  Terminal: +10 on success, from the orchestrator's verdict when available, plus flush share terms from the same
  geometry as the check.
- Algorithm: AWAC with twin critics and a decaying pull to the prior (`programs/learner.py`), reused from the standing
  cabinet run. Off-policy with all data, because each episode costs minutes of robot time. It stays near the logged
  actions, which matters with few transitions.

## Choices
- No simulator yet. The hard part is millimetre contact with real wood and an uncertain grip, which a quick sim would
  not capture; the operator asked for a first real rollout as soon as possible.
- SGS-RL not used and not reviewed: it targets PPO in simulation, and this design learns directly on the robot.
- Force limits follow the operator: 20 N for the grasp approach, 25 N for the descent, 65 N session cap.

## Open issue in the fixed success check (for the operator)
`checks.drawer_lying` picks the "cabinet" as the largest blob inside the size prior (extent 0.12-0.30 m). The real
cabinet measures 0.28-0.29 m and the drawer up to 0.147 m, and the two blobs sometimes merge (attempt 6: 0.446 m).
Dry attempt 5 reported "the cabinet moved 24.3 cm" with nothing moved, so a correct insertion could be failed by this.
My program finds the drawer in its own window, so it does not depend on the merge. The check itself is not mine to change.

## First real episode (attempt 10, commit 2f0398e)
Ran end to end, exit 2, 6 transitions. Pinch held (reading 84.1). The drawer met the rim at a bottom height of ~0.096 m
and three 25 N stops ended the descent as a jam, with no insertion. Corrections applied: 1.7 mm, 1.2 mm, 0.67 deg.
The commanded "bottom" is not a true drawer height after contact: the tool sank 8 mm while the drawer stayed on the rim.
Reward changed after this episode: a force-stopped step earns no descent reward.

## Reflection on episode 0 (attempt 12)
Jammed at the rim again, with the drawer's lower edges within ~2 mm of the opening in the back-projected side image.
Hypothesis, not confirmed: the drawer hangs tilted ~3 deg from the off-centre tab (estimate from a blurry image, +-2 deg).
At 72.3 mm wide and 91.5 mm tall, a tilt above ~2 deg needs more than the 75.7 mm opening, and the action space has no
tilt. Changed: a read-only tilt measurement at the hover, with the standing drawer as reference. Next step depends on it.
Commit 0550191 (no descent reward on force stops) is not in the history, so episode 0 ran without it; committed again.

## Reflection on episode 1 (attempt 14)
The tilt reading at the hover: -3.7 deg along a, +6.0 along b (standing reference -1.9, -0.2), and the jam image shows the
far (+a) side low with the near edge on the rim. Sag about the jaw axis fits the physics (centre of mass ~2 cm off the
tab). Added: the tool levels the drawer about the jaw axis at 2 cm above the rim from side depth, up to 6 deg, with a
re-measure. The +6 deg along b has no physical explanation yet and may be a partial-view artefact, so it is logged, not
corrected. The rim fit failed in episode 1 (too few edge samples on one long side); a tighter search window did not fix
it offline, so it was not kept. The stored rim it falls back to agrees with the fits of attempts 10 and 12 within 1 mm.

## Reflection on episode 2 (attempt 16)
Levelling turned off (measure only). Its readings moved -0.95 deg after a -1 deg turn (expected about +1) and +4.3 after
a -1.9 deg turn (expected about +1.9), so the side-depth tilt repeats to ~2 deg, the size of the tolerance. Tilt remains
an unconfirmed cause. The +8..10 deg along b is probably the fingers in the fit: the drawer's long edges look parallel
to the rim in the image.
Changed: wider exploration on alignment steps (extra std 0.6 action units, about 2.5 mm and 1.3 deg over the two align
steps), one lateral alignment step after each contact, up to 5 contacts per episode. Episodes 0-2 tried the same
alignment within ~1 mm, so the data could not show where the drawer fits.

## Reflection on episode 3 (attempt 18)
Five contacts at different offsets all stopped at the rim, some within ~2-3 mm of the correction measured below, so that
correction alone may not be enough. New evidence from the
recorded wrist frame at the hover: the held drawer's top face is 0.2-2.2 mm off its planned position in a, -1 to +2 mm in
b, and about 1 deg in yaw, repeatable within ~0.3 mm across frames of one episode. That is as large as the clearance.
Changed: `held_offset` measures this at the hover and the tool corrects it (bounded 5 mm, 2.5 deg), then the learned part
starts from the corrected pose; exploration reduced to std 0.35 around it. With the tool at yaw 108 the measurement's
angle check rejects it (attempts 10, 12), and that case keeps the old uncorrected prior.

## Reflection on episode 4 (attempt 20), operator note 17:58
Measured in one recorded wrist image at 2 cm above the rim (drawer's long sides against the cabinet's near wall): the
drawer sat 3.3-3.7 mm off the opening along b in all three episodes checked (16, 18, 20), about 2 mm of it not seen by the
previous correction. That exceeds the ~1 mm clearance along b and is the best explanation so far for the jams.
Changed:
- Grasp yaw chosen so the camera always ends on the near-wall side, where drawer and cabinet are both visible.
- 2.5 cm above the rim: measure drawer vs opening, correct, repeat once, record what is left (bound 10 mm, 4 deg).
  b is differential (same image). a and yaw are the drawer against the rim measured before the grasp, because the near
  wall's outer edge moved ~8 mm between episodes (its outer face is in view).
- On contact: 3 mm back off, then a scripted search point around the measured alignment (+-3 mm in a, b; +-1.5 deg),
  plus the policy's residual. Up to 7 contacts. Residual range +-8 mm, +-4 deg around the measured alignment; observation
  now holds the residual.
Not done: the tilted-entry trick. Added only if the measured alignment still jams.

## Tuning after 5 episodes (0 successes)
The curve has no signal: every return comes from rim contacts. The dry run of 94268dd showed the learned policy drifting
0.3-0.5 mm per step (3.8 mm over a descent), trained on data where c meant something else, and that would undo the
measured alignment. Changed: prior version 2. The old actor and learner state are ignored and only v2 episodes train.
Before the first contact the residual scale is 1 mm, 0.5 deg per unit; the 3 mm scale and extra noise apply at search points.

## Real test of the measured alignment (attempt 24, commit 1cbb663)
The alignment worked: residual after two corrections 0.3 mm, -0.4 mm, -0.09 deg. All 7 contacts still stopped ~3 mm above
the rim. So the top-face position is not the limiting error. Tilt is the remaining suspect: a far-side sag of ~3 deg about
the jaw axis makes the footprint 72.2 + 4.8 = 77 mm against a 75.7 mm opening, and puts the lowest edge ~3 mm below the
commanded bottom. Not confirmed: side-depth tilt repeats to ~2 deg, and the wrist edge-convergence measure I tried reads
+1.0, -6.1, -4.1 deg on three looks at one pose and -6.5 on a standing drawer, so neither is used for control.
Changed: the first try tilts the tool -3 deg about b through the pinch point, and the contact search sweeps tilt (-5, -1.5,
+2.5) and then +-2 mm around -3. Tilt stays scripted for now, and the log shows which values were tried and entered.

## Real test of the tilt sweep (attempt 26, commit c668ca8)
Tilts -5, -3, -1.5, +2.5 deg about b and +-2 mm all stopped at the rim (commanded bottom ~0.096-0.097 m). Tilt about b is
not the explanation I expected. The put-back descents reach 3 mm above the table with almost no force (2-12 N), so the
held drawer's lowest point is less than ~3 mm below plan: a large tilt is unlikely.
Gap in the searches so far: one axis at a time around the measured centre. The drawer fits only in a window of about
3.7 x 2 mm, and the half-width assumed for the a edge (36.8 mm) is itself uncertain by 1-2 mm.
Changed: 2-D probe grid (a 0, +-3, +-6 mm; b 0, +-2, +-4 mm), nearest first, up to 25 contacts, stop lateral moves once the
drawer is 4 mm below the rim. `insert.contact_map` records the contact height per point. Contact penalty 0.5 -> 0.15.

## Touch search (operator notes 18:30, 18:32; attempt 28)
Attempt 28's 25-point grid: no entry, but contact height falls steadily toward +a (96.5 mm at a = -6 mm to 93.9 mm at +6 mm,
relative to the corrected pose), so the opening lies further along +a than anything tried.
Changed:
- `programs/touch.py`: line searches along a, b, yaw from the best (lowest) contact, then a 4 mm spiral over +-15 mm, up to 36
  touches, 4 mm lift between touches. A drop of 1.5 cm below the rim saves the offset to `rl/touch_centre.json`, which is the
  start of the next episode and the centre of the RL action.
- Each contact logs the sideways force in the opening frame, to test the sim's claim that it points to the opening.
- Hover measurement logged only, no correction (operator). Its b reading did follow the tool's correction (3.2-4.8 mm to
  <0.7 mm), so the near wall is at least partly in view, but it did not lead to an insertion.
- Opening detector: the look-down clipped the -y long side (outer corner at image row -78, principal point at row 251 of 720).
  The look now aims 28 mm toward -y. Not yet confirmed on a real image.
- Floor 5 mm instead of 13 mm (sim worker), so a force stop at 2 cm no longer counts as seated.
Limits: a 4 mm grid can step over a window of ~2 mm across b; the line searches rely on the height trend. Offline success on a
toy contact model was 39-100% depending on the offset, so "cannot miss" does not hold. The sim's lying_policy files are not on
this machine, so Descent and Guided are not used or compared yet.

## First real insertion (attempt 30, commit f4e80c5)
Straight descent at the offset a +3.0 mm, b -3.0 mm, yaw 0 from the plan, with no hover correction: seated with 0 touches, median
top 96.9 mm, flush share 1.0, peak 17 N at the floor. The side image shows the drawer in the nearest compartment, pull up.
- Likely cause of the earlier jams: the plan is ~4 mm off, and the hover "correction" moved the wrong way along a (it read +5 mm
  here, while the working offset is +3 mm). Not proven: attempt 28 touched almost the same offset (+2.9, -2.8, -0.14 deg) after a
  lateral move and did not enter. One success says nothing yet about the success rate.
- The touch search was not exercised (0 touches), and the sideways-force question from the sim is still unanswered.
- This success came from the scripted prior at a found offset. No learned policy has improved anything yet.
- The independent geometric check said "the cabinet moved 3.3 cm". It did not: phase correlation of before/after side images on the
  far compartments, far wall and long side gives shifts <= 0.11 px (~0.2 mm). The blob's centre moves because the filled
  compartment changes the top surface. The check can therefore fail correct insertions.
- Learner: a failed verdict no longer turns my positive terminal into -5 unless it states a wrong orientation; it pays 0.

## Reflection on episode 5 (attempt 32, commit f4e80c5)
Second real insertion: the vision judge says success, and the geometric check says "cabinet moved 3.3 cm" again, so the verdict is
unsure and my terminal reward was 0. Side-image shift of the cabinet in this episode: 0.2 px.
Changed:
- My own check: flushness from `checks.drawer_lying` without its blob-centre comparison; cabinet motion from side-image phase
  correlation (stop above 6 px, ~1 cm; a synthetic 9.8 px shift reads 9.73). The orchestrator's check is unchanged.
- Learner: an unsure verdict with vlm=True and the drawer down at the floor pays the success bonus.
- `touch_centre.json` held the pose at the end of the descent, including exploration drift (+1.8, +1.1 mm, 0.75 deg). It now
  holds a running estimate of the entry offset; old files are ignored and the start returns to (+3, -3 mm, 0).
- Once 4 mm inside the opening, no lateral residuals: the walls guide the drawer.
Note: commits made in a turn that ends with `done` right after a run (0550191, b0af540) were not kept in the history; their file
changes were, and are committed here.

## Reflection on episode 6 (attempt 34, commit 87de17e), operator note 18:56
Third real insertion in a row at the saved offset (entered at +2.9, -3.1 mm with 0 touches, top 96.5 mm, flush 1.0). My program
claimed success; the verdict was still unsure because the orchestrator's perception check said no.
Changed: own verify also uses the cabinet's outline centre (`checks.outline_xy` when the robot machine has it, else the same
formula in `vision.py`; this machine's copy of `checks.py` does not have it yet). The cabinet counts as moved only when the outline
(> 2 cm) and the side-image shift (> 6 px) agree, because the outline's far edge varies on a static scene.

## Reflection on episode 7 (attempt 36, commit dd1c01b)
First episode accepted by both independent checks. The touch search found the opening after 2 contacts, at a +9.8 mm; attempts 30-34
entered at +2.9..3.0. Cause unknown: the tab-to-frame and outline-to-frame distances are the same in all 13 episodes measured
(21-23 mm, 2.2-2.8 mm), so the pinch offset is not it. A different sag in the grip is possible but unmeasured.
Changed: the start offset is now the median of the last 9 entries (the running mean had moved it to +5.0, between the two values,
where neither fits). The a and b line searches drop a direction once its contact is higher than the best one.

## Scripted baseline arm (operator 19:10) and episode 8 (attempt 38)
Training episodes with index % 3 == 2 (from 11; 8 ran before this change) run with zero residuals and no exploration, everything else
identical. `summary.arm` is scripted, learned or learned_eval; transitions of all arms are logged and train the AWAC learner.
`SCRIPTED_REV` in `execute.py` names the scripted part; bump it whenever that part changes.
Fix made alongside: at search points the deterministic policy's residual was dropped in eval mode, so eval did not run the learned
policy there. It now applies in eval too.
Episode 8 failed at perception: a blurred wrist frame broke the thin label frame in the mask ("metal too small"). The program now
tries the frame, a smoothed copy, then fresh frames. On a 640 px copy of the failed frame the detector finds the tab (s -21.3 mm).

## Tuning after 10 episodes (attempt 42, commit a0c9a76)
No program change: real test succeeded (independent geometric check True, top 96.7 mm, 3 touches). The blurred-frame retry was not
needed (1 try), so the episode 8 fix is still unexercised on the robot.
Pattern in the entry offsets: first-try entries are all at a 2.9-3.0, b -3.0..-3.2 mm, yaw within 0.06 deg (attempts 30, 32, 34, 40).
First touches 0.3-1.0 mm away missed (attempt 28: +2.9, -2.8; attempt 36: +3.7, -2.1; attempt 42: +3.8, -2.5), and the last two then
entered at +9.8. So the +9.8 entry is probably not a different grasp but a second way in after the narrow window at +3 is missed.
Hypothesis only (6 entries). In attempt 42 the deterministic policy's residual itself moved the first descent off the window
(+0.8, +0.6 mm), so the learned residual currently costs touches. Left as it is to keep the scripted/learned comparison clean;
the scripted arm (first at episode 11) will show it.

## Reflection on episode 10 (attempt 44), two causes found
1. Run identity (my mistake). The orchestrator stores the last verified command as the episode command, and my test command carried
   `RLPJ_EPISODE=-1 RLPJ_RL_MODE=eval`, which overrides its environment. Episode 10 therefore ran as an eval test: no exploration, no
   arm schedule, no verdict link. `run_identity()` now reads the run's name from the orchestrator's attempt.json. Test commands must
   not set these variables again.
2. Pinch height explains the entry offsets. The grasp height follows the side-depth reading of the drawer top (91.0-94.5 mm):
   Episodes that tried the +3 mm offset: face <= 91.7 mm entered on the first descent, 4 of 4 (attempts 30, 32, 34, 40);
   face >= 92.7 mm stopped on the rim there, 0 of 4 (attempts 28, 36, 42, 44), and 36, 42, 44 entered only at +9.8 / +15.9 mm.
   Earlier episodes tried other offsets and say nothing about this.
   Likely mechanism, not measured: a higher pinch holds less of the tab, so the drawer hangs differently (attempt 44: 94.4 mm, tilt
   reading -6.8 deg, 17 touches). This is a correlation over 8 episodes, not an experiment. Changed: the pinch uses a fixed top height of 91.2 mm (scripted r2,
   both arms), unless the measurement is more than 8 mm away.

## Reflection on episode 11 (attempt 46, first scripted-arm episode)
Success by both checks, and the run identity fix works (train, 11, scripted). The fixed pinch height did not remove the offset jump:
at the same pinch (h_g 139.2 mm, reading 84.1) the first descent at +3.0 mm stopped on the rim and the drawer entered at +6.0 mm.
So pinch height is not the (whole) cause. The side-depth reading of the top was again high (92.9 mm), so that reading still goes
with a larger offset (0 of 5 first-try entries at >= 92.7 mm, 4 of 4 at <= 91.7 mm), but why is unknown.
Changed (scripted r3): the start offset is the centre of the densest cluster of past entries. The median of an even number of
entries had just become +4.5 mm, between the clusters at +3 and +6, where the drawer never entered.

## Reflection on episode 14 (attempt 52, scripted r3)
Success, but the start offset was running away. The entry comes one or two 3 mm search steps past the first touch: start +3 -> entry +6 (attempts
46, 48, 50), start +3 -> +9.8 after two touches (36, 42), start +6.2 -> entry +9.2 (attempt 52). So entries after a rim touch do not locate the opening, and feeding them back moved
the start to +9.8 mm and would keep moving it. Possible mechanism, not measured: the pads are ~22 mm wide along a and the tab ~12 mm, so
a rim contact may shift or level the drawer in the grip, after which a range of tool offsets enters.
Changed (scripted r4, both arms): only entries on the first descent (no rim touch before) update the start offset. The stored file is
reset to the four first-descent entries (attempts 30, 32, 34, 40), so the start returns to about +2.9, -3.1 mm.

## Scripted part frozen at r4 (operator 19:49, commit 48fa784)
From here the scripted part changes only to fix a failure that actually happened, with the episode named here. Improvements go through the
learner (actor, critic, reward, observation).
One edit was needed to make the freeze real: the start offset was still updated at run time by first-descent entries, including those of
the learned arm, so the scripted baseline could inherit what the learned arm found. It is now the constant r4 value (the centre of the
entries of attempts 30, 32, 34, 40, about +2.9, -3.1 mm) for all arms, the same value r4 would start from. Entries stay logged in
`summary.entry`. `SCRIPTED_REV` stays r4.

## Tuning after 15 episodes: the critic had diverged (learner only, scripted r4 untouched)
No failed episode this round, but the learner log showed divergence: Q mean -6 -> -75 (min -131) while returns-to-go are -2.8..12.2,
critic loss 0.7 -> 25, temperature alpha 0.15 -> 1.4, entropy -9.7 against a target of -2.
Likely cause: most logged actions are exactly 0 (scripted arm, frozen lateral action inside the opening), so the AWAC fit collapses the
policy std, log-probabilities become large, the auto-tuned alpha keeps growing, and the entropy term in the critic target drags Q down.
Changed: no entropy term in the critic target for AWAC, no alpha update, targets clamped to [-20, 20], minimum policy std 0.1 (it had
collapsed toward the 0.018 floor), and a learner-version check that restarts actor and critic on all v2 data.
Offline on a copy of the real data (300 transitions): Q vs return-to-go MAE 82 -> 2.1, correlation 0.63 -> 0.88, Q range -2.7..12.1.
This shows a sane critic on the training data. It does not show that the policy will do better on the robot.

## Reflection on episode 15 (attempt 55): the learner fix had not reached the robot
Success (1 rim touch, entry at +5.3 mm). The learner fix of the tuning turn was not in the git history, so the robot machine's learner
still ran the old code after this episode: Q -92, critic loss 34, alpha 1.7. Commits made in a tuning turn that ends with `done` have now
been lost four times (0550191, b0af540, 3676c98, 22c6f05), while the edited files stayed in the working tree here. Commits made in
reflection turns are kept. The learner fix is committed again in this reflection turn.

## Sim arm (operator 20:27): episodes 31, 34, 37, 40, 43
The sim policy runs unchanged (`sim/drawer_insert/lying_policy.py`, `weights/policy_lying.npz`) after the same perception, grasp, carry and hover.
- Frames: believed centre = measured opening + the frozen r4 offset; axis = measured opening yaw. My jaws lie along the opening's y, so
  `yaw_straight` = opening yaw - 90 and the policy's tool frame is the opening frame at the hover (the sim draws the soft jaw axis along tool x or y).
- Estimates: `off` from the wrist look-down before the grasp, yaw 0 (squared by the carry), tilt 0 (not measurable; README: costs ~3 points).
  hold = tool to top face, 48 mm, because I pass the Kortex tool point; it only shifts the z reference.
- Added around the policy, not in it: force stop 35 N, command floor at my lowest bottom height (5 mm), seated rule (README's, or a stop at the
  floor as in the other arms). The episode ends on two stops above 40 N, on a stop at the rim 14 mm or more from the start, or after 20 stops.
- Data: per-decision log as the README asks, in `rl/sim_arm/`, not in `rl/transitions`: the action layout differs, so my learner never trains on it.
- The learned and scripted `insert()` code is unchanged. A test run can force the arm with RLPJ_ARM=sim; orchestrator episodes ignore it.
Not done yet: the dry run required before episode 31, which needs a run turn.

## Reward = the report's fixed return (operator 20:21)
`rc.fixed_rewards`: -0.05 per step, -1 per force-stopped step, at the last step -0.1 per N of its peak above 15 N and +10 for a success. The sum equals
`scripts/rl_plots.py fixed_return`. As in that formula, `contacts` includes the stop at the floor and `peak` is the last step's peak, not the episode's maximum.
- All stored episodes are relabelled when the learner loads them (files unchanged); the learner restarts once (version 3).
- Exception to an exact match: episodes 5 and 6 (verdict unsure only because of the old cabinet-move bug, vision judge positive, drawer at the floor) get the +10
  in the learner, and the plot gives them 0. Test runs have no verdict and use my own check.
- First descent: the residual already acted there (1 mm, 0.5 deg per unit, on the 3-4 steps above the rim), but with std ~0.1 it barely explored. The learned arm
  now adds noise of std 0.5 units before the first contact. The scripted arm still has zero residual and no noise.
