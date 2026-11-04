# rlpj: autonomous data collection -> pi0.5 on a Kinova Gen3

16-831 final project. One arm, two modes. Claude (headless Claude Code) writes the EXECUTE and RESET programs from a
task specification (text, or one human demonstration), the orchestrator alternates them to collect successful
episodes with randomised layouts, pi0.5 is LoRA fine-tuned on the episodes (openpi), and the policy is evaluated on
the robot until it reaches the target success rate. Design and contracts: `docs/DESIGN.md`.

Task in use: `config/pipeline.yaml` -> `config/tasks/bee_in_tape_roll.yaml` (foam bee into a tape roll).
`cup_down_stacking.yaml` is the next task.

## Machines

- 4090 (this machine, `zen-System-Product-Name`): robot 192.168.2.9, side RealSense, orchestrator, robot programs.
- 5090 (`ssh Worksation_5090`): Claude Code agents, VLM judge, openpi training, policy server. Same repo path.

## Setup

```bash
uv sync --extra robot                      # 4090: robot venv (.venv)
bash scripts/sync.sh push                  # copy the repo to the 5090
ssh Worksation_5090 'cd ~/robot_learning_final_pj && bash scripts/setup_openpi.sh'   # once: openpi + pi05_base (12 GB)
```

## Daily run

```bash
# robot: boot, focus the wrist camera, check calibration (INSTRUCTION.md), then
.venv/bin/python -m rlpj.robot.check       # read-only preflight
.venv/bin/python -m rlpj.robot.fit_kin     # read-only kinematic fit (about 1 min), after any recalibration
bash scripts/record_demo.sh                # one_demo condition only: record one human demonstration

.venv/bin/rlpj new --condition text        # or one_demo; prints the run id
.venv/bin/rlpj run-all <run>               # check -> Agent 1 -> Agent 2 -> collect -> Agent 3 (train, evaluate)
.venv/bin/rlpj status <run>                # progress, cost, success rates
.venv/bin/rlpj review <run>                # operator verdicts for 'unsure' episodes
```

The steps also run one at a time: `develop <run> execute|reset`, `collect <run>`, `learn <run>`, `eval <run>
--checkpoint <dir>`, `eval-zero-shot <run>`. Everything lands in `runs/<run>/` (see `docs/DESIGN.md` section 4).

## Safety

Robot motion happens only inside orchestrator-run attempts (`RLPJ_MOTION=1`), after a countdown the operator can
refuse, with Ctrl+C stopping and homing the arm. Every segment is checked in a model before it is sent.
Agents never run robot commands; they only return the command to run.

## Layout

```
config/      pipeline.yaml, tasks/<task>.yaml
rlpj/        robot/ perception/ success/ agents/ orchestrator/ vla/      (see docs/DESIGN.md section 3)
scripts/     setup_openpi.sh sync.sh record_demo.sh train.sh serve.sh
tests/       offline tests: .venv/bin/python -m pytest -q
docs/        DESIGN.md
INSTRUCTION.md, CALIBRATION.md   lab procedures for the arms and cameras
```
