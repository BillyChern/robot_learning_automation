# Design: agentic data collection -> pi0.5 on one Kinova Gen3

Contract for every module. Short on purpose; details live in code docstrings.

## 1. What the system does

One arm (kinova_a, 192.168.2.9) in two modes. A foundation model (Claude Opus 5.5 via headless Claude Code)
writes and repairs the programs; the orchestrator runs them.

| Role | What it produces | Runs where |
|---|---|---|
| Agent 1 EXECUTE | `programs/execute.py`: perceive, plan, do the task, verify, claim outcome | code on 5090, program on 4090 |
| Agent 2 RESET | `programs/reset.py --targets t.json`: undo the task, put objects at given targets | same |
| Collection loop | reset -> check layout -> execute (recorded) -> verdict -> episode | orchestrator on 4090 |
| Agent 3 LEARN | convert episodes -> LeRobot, LoRA fine-tune pi0.5, serve, decide hyperparameters | 5090 |
| Evaluation | policy rollouts (recorded) with Agent 2 resets between them; success rate vs target | 4090 + policy server on 5090 |

Task specification conditions for Agent 1: `text` (goal sentence) or `one_demo` (one human RGB-D recording). Both may
reuse the libraries listed in the task yaml. Compared on time, tokens, cost and attempts to first success only.

## 2. Machines

| | 4090 `zen-System-Product-Name` (this repo's home) | 5090 `Worksation_5090` (ssh alias; `WorkStation_4090` from there) |
|---|---|---|
| has | arm network, side RealSense, repo venv `.venv` (uv, `--extra robot`), demos | Claude Code (subscription), openpi venv `third_party/openpi/.venv`, GPU 32 GB |
| runs | orchestrator, robot programs, policy rollouts | agents, VLM judge, training, policy server (port 8000) |
| path | `/home/r2r/robot_learning_final_pj` | same path; `scripts/sync.sh push|pull` (rsync, excludes runs/ episodes) |

The 5090 cannot reach the arm. Robot programs run only on the 4090, started only by the orchestrator.

## 3. Repo layout and ownership

```
config/pipeline.yaml, config/tasks/<task>.yaml   settings; rlpj.config.load()/robot()
rlpj/robot/      calib kortex cams kin plan sim executor home  (done)  + session recorder watchdog liveview fit_kin check
rlpj/perception/ tabletop (blobs from side depth)  layout (targets, checks)  wrist (projection helpers)
rlpj/success/    checks (geometric task checks)  vlm_judge (Claude on 5090)  verdict (combine)
rlpj/agents/     runner (headless Claude Code over ssh)  workspace (per-agent dir, git, sync)  prompts/*.md  schema.py
rlpj/orchestrator/ state  attempts  operator  loop  cli   (`rlpj` entry point)
rlpj/vla/        convert (episodes -> LeRobot)  kinova_policy (openpi transforms)  train_config  train.py  serve.sh  policy_client  rollout
scripts/         setup_openpi.sh  sync.sh  record_demo.sh
tests/           offline only (no robot, no network); pytest
runs/            all outputs (git-ignored)
```

## 4. Run directory

```
runs/<run_id>/
  state.json                      orchestrator state (phase, counters, sessions, datasets, checkpoints, eval results)
  agents/<execute|reset|learn>/   agent workspace: git repo; programs/, notes/, inputs/ (calibration copy, demo link),
                                  attempts/attempt_<N>/ (log.txt, summary.json, images, plans/), session.json, usage.jsonl
  episodes/ep_<000123>/           one robot execution (collection or evaluation), format below
  datasets/<name>/                LeRobot dataset (built on the 5090, HF_LEROBOT_HOME points here)
  checkpoints/<train_run>/        openpi checkpoint dirs
  evals/<eval_id>.json            per-checkpoint evaluation summary
  calib/<robot>/kin_fit.json      per-arm kinematic fit (rlpj.robot.fit_kin)
```

### Episode format (`rlpj.robot.recorder` writes, `rlpj.vla.convert` reads)

```
meta.json     episode, run_id, task, instruction, mode (collect|eval), program {name, commit, command}, robot {name, ip},
              fps, layout_target, layout_before, layout_after, t_start, t_end, n_steps,
              outcome {claim, perception, vlm, operator, final, reason}, calibration_hashes
side.mp4      640x360 @ fps, BGR source, from side_left colour
wrist.mp4     640x360 @ fps, from the Kinova wrist colour stream
steps.npz     t (N,) s; q (N,6) deg; dq (N,6) deg/s; tool_pose (N,6) m+deg; gripper (N,) 0..100 reading;
              q_cmd (N,6) deg target at that tick; gripper_cmd (N,) 0..1 last command; phase (N,) int index into phases
phases.json   [{"name": str, "i0": int, "i1": int}]  (program phases, e.g. perceive, approach, grasp, lift, place, home)
summary.json  the program's summary (copied from the attempt)
before_side.jpg after_side.jpg before_wrist.jpg after_wrist.jpg   640 px
log.txt
```

q_cmd = the executor's current reference joints while a segment runs (or the policy's commanded target), else the measured
q. gripper_cmd = last commanded value (0 open .. 1 closed); before any command, measured reading / 100.

## 5. RobotSession (the one API every robot program uses)

```python
from rlpj.robot.session import RobotSession, Abort
with RobotSession(program="execute") as s:     # dirs/env: RLPJ_ATTEMPT_DIR, RLPJ_RECORD_DIR (optional), RLPJ_MOTION
    s.phase("perceive")                         # named phase (recorded, shown in the live view)
    color, depth, t = s.side.latest()           # 1280x720 BGR, depth m (gain applied), aligned
    img, t = s.wrist.latest()                   # 1280x720 BGR
    fb = s.feedback()                           # q, dq, pose, gripper, state
    blobs = s.tabletop()                        # rlpj.perception.tabletop.detect on a median side depth (base frame)
    s.view.set_scene(objects=[...], targets=[...], path=None)   # overlays (dicts: xy, z_top, r, label)
    s.decide("why")                             # logged + shown + in summary
    s.move_joint("name", q_goal_deg, speed="normal"|"slow")
    s.move_line("name", T_goal, speed="slow", tip_margin=0.008)   # straight tool line; plan.tool_down_T(xyz, yaw) helps
    reading = s.gripper(0.8, "grasp")           # 0 open .. 1 closed; waits until settled; returns 0..100 reading
    s.summary(key=value)                        # merged into summary.json at once
    s.require(cond, "message")                  # False -> Abort -> outcome fail, return home
    s.claim("success")                          # or "fail"; default "fail"
```

Behaviour (ported from the verified FAR programs): reads start state (must be `ARMSTATE_SERVOING_READY`, waits up to
30 s), saves the start joints as home, verifies model FK vs tool_pose (< 6 mm), starts the watchdog process (stops
and homes the arm if the program dies or stops heartbeating while moving), checks every segment in the model before
sending it (limits, speed, table clearance, base column, reach) and refuses to move if a check fails, tracks with
joint-speed streaming and aborts at > 1 deg / 10 mm deviation, returns home on Abort / Ctrl+C / SIGTERM along a
checked path, records the stop moment (step, segment, joints vs plan, gripper, last frames), serves the live view on
port 8090 and writes `dashboard.mp4`. Speeds: normal 20 deg/s, 0.10 m/s; slow 5 deg/s, 0.02 m/s. Motion needs
`RLPJ_MOTION=1` (set by the orchestrator only); without it every command raises and the program is a dry run that
still plans, checks and writes replay videos. Exit codes: 0 success, 2 fail, 130 operator stop, 3 exception.

## 6. Program contracts

- `programs/execute.py` (Agent 1): no arguments. Does the task once from the current scene. Must `s.claim(...)` from its own
  verification and write `summary.json` keys: `claimed_outcome`, `step`, `first_failed_check`, `objects` (what it saw,
  base frame), `verify` (numbers behind the claim).
- `programs/reset.py --targets <json>` (Agent 2): `{"<object>": [x, y], ...}` in the base frame. Undoes the task if needed,
  moves each object to its target, writes `summary.json` with `measured` `{"<object>": [x, y]}` after placing and `claimed_outcome`.
- `rlpj.vla.rollout` (ours): policy rollout through the same session, recorded like an execute episode.

Programs run from the agent workspace on the 4090 with `.venv/bin/python` of this repo on `PYTHONPATH`. They import
only `rlpj.*`, the standard library, numpy/scipy/cv2/mujoco, and their own files.

## 7. Agent <-> orchestrator protocol (`rlpj.agents.runner`)

Headless Claude Code on the 5090: `claude -p <prompt> --session-id <uuid> | --resume <uuid> --model <m> --effort <e>
--output-format json --json-schema <schema> --permission-mode acceptEdits --setting-sources project,local
--strict-mcp-config --disable-slash-commands`, cwd = agent workspace. Verified: the result JSON has `structured_output`,
`session_id`, `total_cost_usd`, `usage`, `modelUsage`, `num_turns`, `duration_ms`, `is_error`.

Every turn ends with one structured action (`rlpj/agents/schema.py`):

```json
{"action": "run" | "done" | "need_operator" | "give_up",
 "command": "python programs/execute.py",      "moves_robot": true,
 "what": "one line: what the run should do",   "report": "short text (done/give_up/need_operator)"}
```

`run`: the orchestrator syncs the workspace to the 4090, runs the command as an attempt (operator gate if `moves_robot`),
syncs results back and resumes the session with a digest: exit code, `summary.json`, log tail, list of images
(the agent may Read them). `need_operator`: the orchestrator shows `report`, waits for the operator, resumes.
Workspace `.claude/settings.json` denies writes outside the workspace, reads of other runs, and any Bash containing
`RLPJ_MOTION`, `ssh`, `scp`, `rsync`. Reads allowed: this repo, the task's `reuse` paths, the demo. Agents may run offline
analysis on the 5090 with `uv run` in this repo (numpy, cv2, mujoco, scipy; no kortex, no cameras).

Usage per turn is appended to `usage.jsonl`; `session.json` keeps the session id and condition. Resume on crash.

## 8. Perception and success

- `rlpj.perception.tabletop.detect(P_base, workspace) -> [blob]`, blob = `{xy, z_top, height, extent, n, points}`; plus
  `table_fit`, `depth_correction` (range-dependent scale so the table comes out flat), `cloud(side_cam)`.
- `rlpj.perception.layout.sample(task_cfg, rng) -> {"<object>": [x, y]}`; `check(task_cfg, targets, blobs) -> (ok, report)`
  assigns blobs to objects by the size priors in the task yaml.
- `rlpj.success.checks.<task>(before_blobs, after_blobs, task_cfg) -> (ok|None, report)` geometric, independent of
  the program's own check (bee: a small blob inside the ring's hole and none elsewhere).
- `rlpj.success.vlm_judge.judge(images, task_cfg) -> {"success": bool, "confidence": 0..1, "reason": str}`: Claude on the
  5090 reads the before/after images; `--json-schema`; cost logged.
- `rlpj.success.verdict.decide(claim, perception, vlm) -> {"final": "success"|"fail"|"unsure", "reason"}`:
  success needs claim and at least one independent yes with no independent no; `unsure` goes to the operator queue.

## 9. VLA data

state (7) = `[q (6, rad), gripper reading / 100]`; actions (7) = `[q_cmd (6, rad), gripper_cmd (0..1)]`, absolute, converted to
deltas for the 6 joints by openpi's `DeltaActions(make_bool_mask(6, -1))`. Images: `image` = side, `wrist_image` = wrist,
stored at 640x360 (openpi resizes to 224x224 itself). `task` = the instruction. LeRobot `fps` = recording fps.
Train config `pi05_kinova_lora` (LoRA, batch 16-32, `pi05_base` weights, prompt_from_task, `use_quantile_norm` default),
registered without editing openpi (our `rlpj/vla/train.py` adds it to `_CONFIGS_DICT` then calls openpi's `main`).
Serve: openpi `serve_policy.py policy:checkpoint --policy.config=pi05_kinova_lora --policy.dir=<ckpt>`. Client on the 4090
speaks openpi's msgpack-numpy websocket protocol (own implementation; the openpi client needs numpy < 2).
Rollout control: the policy returns a chunk of 10 joint targets at the recording rate; the session streams them with the
executor's joint-speed tracker after a model check per chunk; hard limits: joint limits, table clearance, max radius 0.65,
10 cm/s, 20 deg/s, deviation 1 deg / 10 mm, per-episode time limit, and Ctrl+C.

## 10. Safety

The operator is always present with Ctrl+C and the e-stop. The orchestrator prints what will run and counts down before any
command that moves the robot. Everything that moves the arm goes through the executor's model checks. No agent process ever
has `RLPJ_MOTION=1`.

## 11. Conventions

Concise, agent-readable code and docs: one-line docstrings, plain names, no emojis, no decorative comments. Tests are
offline and fast (`uv run pytest -q`). Config values come from `rlpj.config`; nothing hard-codes the arm, task or paths.
