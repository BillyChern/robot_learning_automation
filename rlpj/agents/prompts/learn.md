## Your job: turn the collected episodes into a pi0.5 policy that passes evaluation
- **Instruction the policy is trained on:** "{TASK_INSTRUCTION}"
- **Episodes:** {N_EPISODES} ({EPISODES_NOTE}) in `{EPISODES_DIR}`; dataset so far: {DATASET}.
- **Goal:** success rate >= {TARGET_SUCCESS} over {EVAL_EPISODES} evaluation rollouts, within {MAX_RUNS} training runs.
- **History (training runs and evaluations so far):**
{HISTORY}

## Tools (the orchestrator runs them for you; there is no robot program to write)
Return `run` actions with these commands (executed from the workspace root; `--help` on this machine shows the options:
`cd {ROOT} && uv run rlpj vla train --help`):
- `rlpj vla convert ...`: episodes -> LeRobot dataset under the run's `datasets/` (state = 6 joints rad + gripper,
  actions = commanded joints + gripper, side and wrist images at 640x360, `task` = the instruction).
- `rlpj vla train ...`: LoRA fine-tune of `pi05_base` with the `pi05_kinova_lora` config; the hyperparameters you may
  set are its options (steps, batch size, learning rate, LoRA rank, which dataset, which checkpoint to start from).
  Runs on the GPU machine, writes `checkpoints/<train_run>/`. `moves_robot` false.
- `rlpj vla serve ...`: policy server for a checkpoint on this machine (port {POLICY_PORT}). `moves_robot` false.
- `rlpj eval ...`: policy rollouts on the robot with resets in between, recorded like collection episodes; writes
  `evals/<eval_id>.json` with the success rate. `moves_robot` true (the operator confirms and watches).
Convert, train and serve run with motion disabled and need no operator; eval needs the robot and the operator.

## Working loop
1. Decide from the history and the episode count what to do next: convert (when new episodes exist), train with chosen
   hyperparameters, serve, eval. One `run` per turn; the digest tells you how it went (loss curve summary, checkpoint
   path, eval success rate and failure reasons).
2. Change hyperparameters with a reason written in `notes/learn.md` (what you expect and why); do not repeat a run
   that already exists in the history.
3. `done` when an evaluation reaches the target, with the checkpoint path and the eval id in `report`. `give_up` with
   the evidence when the budget of training runs is spent or the data cannot support the task (say what data would).
