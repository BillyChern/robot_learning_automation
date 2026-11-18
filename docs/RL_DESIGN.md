# Design: agent-run real-world RL (drawers into a cabinet)

Second experiment on the same arm and pipeline (see DESIGN.md for everything not changed here). Goal: show that the
chores of RL (building a sim, closing the sim2real gap, reward engineering, RL hyperparameter tuning, running and
babysitting training) can be done by agents, with the robot learning in the real world.

## 1. Task

A small wooden cabinet with three drawer slots is taped to the far end of the table, front facing the robot. Drawers are
light, have no stops, and have a metal pull the gripper can hold. Only the bottom slot is used (user, 2026-11-17): one
drawer, pulled out of the bottom slot by the reset and put back into the bottom slot by the RL policy. The upper two
slots are empty.

- Episode: the drawer lies on the table (pose from the reset's targets); the policy puts it back into the bottom slot.
- Success: the drawer fully inside the bottom slot (front flush with the cabinet, within a tolerance), gripper clear,
  the cabinet unmoved.
- Headline metric: success rate in the orchestrator's fixed evaluation, never the agent's own reward.

## 2. Roles

| Role | Produces | Notes |
|---|---|---|
| Agent RL | `programs/episode.py` (one episode with the current policy: perceive, act, log transitions and its own reward), `programs/learner.py` (training between episodes), its reward function, optional `sim/` (MuJoCo) and sim-to-real work, hyperparameters | decides what to learn and what to script, but the insertion must be learned in the real world |
| Agent RESET | `programs/reset.py --targets t.json`: pull the drawer out of the bottom slot, put it on the table at the given pose | same contract as before |
| RL loop (orchestrator) | reset -> episode (recorded) -> verdict -> learner update; every `eval_every` episodes a fixed evaluation; learning curve in `rl_curve.jsonl` | the agent never edits the loop or the verdict |

## 3. What stays fixed (the agent cannot change it)

- Safety: every motion through RobotSession (model check, keep-in box, speed limits) plus a contact-force limit on the
  arm's estimated tool wrench; a force stop ends the episode as a failure.
- Success verdict: depth-geometry check of the drawer front against the cabinet front, plus the VLM judge on the
  before/after images; the agent's reward is logged but not trusted.
- Evaluation protocol and the 70 % target.

## 4. Open choices left to Agent RL

Sim or no sim; observation (state from perception and proprioception vs. images); action space (Cartesian deltas,
residual on a scripted approach, ...); algorithm (SAC/RLPD-style off-policy expected, given the sample budget); reward
shaping; when to stop training. Its notes must say why.

## 5. Measured setup (2026-11-17, side camera recalibrated: 17 poses, ~2 mm)

- Cabinet: 27 cm tall, front face about 12 cm wide; front face line x = 0.563 - 0.401*y (base frame, m), i.e. about
  0.52 m from the base at y 0.11, turned 22 deg (insertion direction (0.93, 0.37) in xy). Taped to the table.
- Bottom slot: front about 12 x 8 cm, pull about 4 cm above the table (z ~0.03 base frame). Upper two slots empty.
- Main difficulty: the pull is only 4 cm above the table, so a horizontal grasp of the pull puts the gripper body at
  table height; grasping from above (pull, or a drawer wall) and pushing is likely. Left to the agents.
- Still to measure: the drawer itself (once out), and a contact-force limit for pushing.
