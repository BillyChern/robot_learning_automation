## Your job: `programs/execute.py` that does the task on the real robot
- **Task:** {TASK_GOAL}
- **Success:** {TASK_SUCCESS}
- **Objects:** {OBJECTS}. Identify them by geometry (size, height, shape, position), never by colour. Their places
  change between runs within x {LAYOUT_X} m, y {LAYOUT_Y} m (base frame); nothing else is on the table.

{CONDITION_BLOCK}
{REUSE_BLOCK}
## Working loop
1. Read what you need (this prompt, `rlpj/robot/session.py`, `rlpj/perception/tabletop.py`, the inputs). Write
   `programs/execute.py` with phases perceive, plan, approach, grasp, lift, place, verify, home: perceive the scene
   (`s.tabletop()`, refine with the wrist camera if millimetres matter), decide which blob is which object, plan
   straight tool lines above the objects (`rlpj.robot.plan.tool_down_T`), `s.require()` every assumption, verify the
   result from the cameras (what moved where), then `s.claim()` and fill the summary keys of the contract.
2. Commit, then `run` a dry run first (`moves_robot` false): it tests perception, the plan and the checks without
   motion. Inspect the summary and one replay frame or image if needed.
3. Commit, then `run` the real thing (`moves_robot` true) with `what` saying what the arm will do.
4. Read the digest. A fail, a refused check or an operator stop (exit 130) means: find the cause in the log and the
   images, fix, commit, run again. Change one thing at a time; keep what worked.
5. `done` when an attempt is a verified success on the hardware (exit 0, `claimed_outcome` success with the numbers
   behind it) and the program is ready to run unchanged many times: after `done`, the orchestrator runs
   `programs/execute.py` on randomised scenes for data collection. Robustness beats a lucky run: handle both object
   positions anywhere in the ranges above, and re-perceive instead of assuming.
6. The orchestrator may later resume you after repeated failures during collection, with the digest of the failed
   attempts: repair the program the same way and end with `done` again.

Dry runs and offline analysis cost nothing on the robot; real runs cost operator time. Prefer one well-checked real
attempt over many guesses, but never spend more than a few turns without a run.
