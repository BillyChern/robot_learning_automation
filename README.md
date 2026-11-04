# rlpj: autonomous data collection → pi0.5 on a Kinova Gen3

16-831 final project. One arm, two modes (EXECUTE / RESET). Claude writes the robot programs from a task
specification (text, or one human demonstration); the orchestrator runs them, records successful episodes with
randomised layouts, LoRA fine-tunes π0.5 (openpi), and evaluates until the target success rate. Contracts:
`docs/DESIGN.md`. Lab procedures: `INSTRUCTION.md`, `CALIBRATION.md`.

Default task: `config/pipeline.yaml` → `config/tasks/bee_in_tape_roll.yaml`. Switch with `--task <name>` or
`RLPJ_TASK`.

## Machines

| | 4090 (this machine) | 5090 (`ssh Worksation_5090`) |
|---|---|---|
| runs | orchestrator, robot programs, policy rollouts | Claude Code agents, VLM judge, openpi train, policy server |
| has | arm `192.168.2.9`, side RealSense, `.venv` | GPU 32 GB, openpi `.venv`, Claude subscription |
| path | `/home/r2r/robot_learning_final_pj` | same path (`bash scripts/sync.sh push`) |

The 5090 cannot reach the arm. Only the orchestrator on the 4090 may set `RLPJ_MOTION=1`.

## Setup (once)

```bash
uv sync --extra robot
bash scripts/sync.sh push
ssh Worksation_5090 'cd ~/robot_learning_final_pj && bash scripts/setup_openpi.sh'   # openpi + pi05_base (~12 GB)
```

## Prompt templates (yes)

Agents do **not** get a free-form chat. Each role gets a filled template written to
`runs/<run>/agents/<role>/PROMPT.md`:

| File | Used for |
|---|---|
| `rlpj/agents/prompts/common.md` | machines, `RobotSession` API, program contract, hardware facts, safety, action protocol |
| `execute.md` / `reset.md` / `learn.md` | role-specific working loop |
| `condition_text.md` / `condition_one_demo.md` | how Agent 1 receives the task (text vs one RGB-D demo) |
| `reuse.md` | skill-library APPLY / ADAPT / REJECT rules |

`rlpj.agents.workspace.render()` fills `{TASK_GOAL}`, `{TASK_SUCCESS}`, `{OBJECTS}`, layout ranges, `{CONDITION_BLOCK}`,
`{REUSE_BLOCK}`, etc. from the task yaml. Edit those markdown files to change agent behaviour globally; edit the task
yaml for per-task wording. The old FAR `PROMPT_TEMPLATE.md` is the ancestor of `common.md` (hardware facts kept;
from-scratch rules replaced by `RobotSession`).

## Commands (4090, from repo root)

### Preflight (arm on, nothing should move)

```bash
# wrist focus + side calibration: INSTRUCTION.md / CALIBRATION.md
.venv/bin/rlpj check                 # cameras, READY, FK, table height
.venv/bin/rlpj fit-kin               # ~1 min read-only; runs/calib/<robot>/kin_fit.json
bash scripts/record_demo.sh          # optional: one human demo for --condition one_demo
```

### Full pipeline

```bash
.venv/bin/rlpj new --condition text          # or one_demo; prints run id, e.g. 20261104_1200_bee
.venv/bin/rlpj run-all <run>                 # check → develop execute → develop reset → collect → learn → report
```

Step by step:

| Command | What it does |
|---|---|
| `rlpj develop <run> execute` | Agent 1 writes `programs/execute.py` (Claude on 5090; you approve each robot attempt) |
| `rlpj develop <run> reset` | Agent 2 writes `programs/reset.py` |
| `rlpj collect <run> [--n 50]` | reset → randomize → execute → record → judge; refine programs after repeated fails |
| `rlpj learn <run>` | Agent 3: convert → LoRA train on 5090 → eval on robot |
| `rlpj eval <run> --checkpoint <dir> [--n 20]` | policy rollouts only |
| `rlpj eval-zero-shot <run>` | same with base `pi05_base` |

Flags: `--hands-free` (countdown without Enter), `--yes` (skip non-robot prompts), `--task <name>`.

### Reading results

```bash
.venv/bin/rlpj status <run>     # phase, episode counts, success rate, agent $/time, checkpoints
.venv/bin/rlpj review <run>     # label episodes marked unsure
.venv/bin/rlpj report <run>     # writes runs/<run>/REPORT.md
```

Under `runs/<run>/`:

| Path | Contents |
|---|---|
| `state.json` | counters, costs, deployable checkpoint |
| `agents/<role>/` | `PROMPT.md`, `programs/`, attempts, `usage.jsonl` |
| `episodes/ep_*/` | `side.mp4`, `wrist.mp4`, `steps.npz`, `meta.json` (`outcome.final`), images |
| `datasets/`, `checkpoints/` | LeRobot data and LoRA weights |
| `evals/*.json` | success rate vs `evaluation.target_success_rate` (default 0.8) |
| `REPORT.md` | summary for the course writeup |

Per-episode success: `meta.json` → `outcome.final` = `success` / `fail` / `unsure` (program claim + perception + VLM).

### VLA only (no agents)

```bash
# 5090 (or ssh from 4090)
bash scripts/train.sh runs/<run> <dataset> <exp> --num-train-steps 5000 --batch-size 16
bash scripts/serve.sh runs/<run>/checkpoints/<exp>/<step> 8000

# 4090 after serve is up
.venv/bin/python -m rlpj.vla.policy_client
.venv/bin/rlpj eval <run> --checkpoint runs/<run>/checkpoints/<exp>/<step>
```

Dummy smoke (already verified): synthetic episodes → convert → 10-step LoRA → `runs/dummy_ft/checkpoints/dummy10/`.

Also: `rlpj vla convert|train|serve …` (thin wrappers).

## Adding a new task

1. **Copy a task yaml** — `cp config/tasks/bee_in_tape_roll.yaml config/tasks/<name>.yaml` and edit:

   | Field | Purpose |
   |---|---|
   | `name` | must match the file stem and any geometric check function |
   | `instruction` | π0.5 language prompt |
   | `goal` / `success` / `reset` | Agent 1 / Agent 2 / VLM wording |
   | `objects` | names used in layout targets and perception priors |
   | `one_demo` | path to `episode_000` if you use `--condition one_demo` |
   | `reuse` | optional skill libraries / past trials agents may read |
   | `layout` | per-object `x`/`y` ranges (or `from`+`distance`+`angle`), `min_gap`, `radius`, `tolerance` |
   | `perception` | size priors for `layout.assign` (`hollow`, `outer_r`/`inner_r`, `extent`, `height`) |

2. **Select it** — set `task: <name>` in `config/pipeline.yaml`, or pass `--task <name>` / `RLPJ_TASK=<name>`.

3. **Geometric success check** (recommended) — add a function in `rlpj/success/checks.py` named like the task
   (`def <name>(before, after, task_cfg) -> (ok|None, report)`) and register it in `run()`. Without it, collection
   still works via the program claim + VLM, but independent perception checks return unknown.

4. **Measure the objects** — update `perception` and `layout` ranges from the real pieces (callipers + a side-camera
   snapshot). Wrong priors → bad resets and false fails.

5. **Optional demo** — `bash scripts/record_demo.sh <name>` (writes under the recorder’s `data/<name>/`; point
   `one_demo` at `episode_000`).

6. **Reuse** — list related FAR skill libraries or past trials under `reuse:` so agents can APPLY/ADAPT/REJECT.

7. **Sync and run** — `bash scripts/sync.sh push`, then `rlpj new --task <name> --condition text` → `run-all`.

You do **not** need to change agent prompts for a new task: `goal` / `success` / `reset` / `objects` / `layout` /
`perception` flow into the templates automatically. Only change `rlpj/agents/prompts/*.md` if you want different
global behaviour (e.g. a stricter working loop).

`cup_down_stacking.yaml` is already present as the next task (check function also exists).

## Safety

Robot motion only inside orchestrator attempts (`RLPJ_MOTION=1`), after a countdown you can refuse. Ctrl+C stops and
homes the arm. Every segment is model-checked before send. Agents never command the robot; they only return the
command string.

## Layout

```
config/          pipeline.yaml, tasks/<task>.yaml
rlpj/            robot/ perception/ success/ agents/ orchestrator/ vla/
rlpj/agents/prompts/   common + execute/reset/learn + condition_* + reuse   ← templates above
scripts/         setup_openpi.sh sync.sh record_demo.sh train.sh serve.sh
tests/           .venv/bin/python -m pytest -q
docs/            DESIGN.md
runs/            all outputs (git-ignored)
```
