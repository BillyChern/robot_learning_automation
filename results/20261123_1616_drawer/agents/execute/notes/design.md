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
