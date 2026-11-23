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
