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

## 6. Implementation (2026-11-17)

- Task `config/tasks/drawer_rl.yaml` (`mode: rl`): fixtures (cabinet front plane, bottom-slot window), `keep_out`
  (cabinet body from 1 cm behind the front plane; the bottom-slot opening, 1.5-9.5 cm above the table, open to 5 cm (3 cm until 2026-11-18: a drawer recessed 2.3 cm was out of reach)
  because closed fingertips are modelled ~1 cm too deep: reset attempt 16 held the tab yet was refused), drawer layout
  and priors, `rl` settings.
- `rlpj.orchestrator.loop`: `run-all` for RL is check -> develop reset -> develop execute (the RL agent; prompt
  `rlpj/agents/prompts/rl.md`, its episode program keeps the name `programs/execute.py`) -> `rl_train` -> report.
  `rl_train`: reset -> episode (mode `rl`, no self-claim: both independent checks must say yes) -> `programs/learner.py`
  -> every `eval_every` a fixed evaluation (mode `rl_eval`) -> every `review_every` a tuning turn. Curve:
  `runs/<run>/rl_curve.jsonl`; state: `state.json` key `rl`.
- Learner placement: the robot machine's GPU if `learner_gpu_gb` (default 4) is free, else the 5090's, else this CPU.
  `rl/` in the RL agent's workspace is orchestrator-owned (pushed to the 5090, never pulled); `rl/policy/` is copied back
  after a remote learner run.
- Safety added for all tasks: `rlpj.robot.executor` stops a segment when the estimated tool force changes by more than
  `rl.force_limit_n` (Kortex `tool_external_wrench_force_*`, ~0.1 N noise at rest); 25 N stopped drawer pulls
  (22-28 N), so the user removed it (22:30); without it a snag reached 99.6 N and moved the cabinet 6 cm, so it is 50 N
  since 23:10 (user); the peak is logged per segment; `rlpj.robot.sim.check` refuses any
  gripper point inside a `keep_out` box.
- Success check `rlpj.success.checks.drawer_rl`: share of bottom-slot points on the front plane (flush: 0.996 measured
  with the drawer in), behind it (empty slot: 0.41), in front (sticking out); no drawer-sized blob on the table; cabinet
  unmoved.
