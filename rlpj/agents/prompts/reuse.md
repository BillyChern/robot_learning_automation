## Reusable material (read-only, `inputs/reuse/`)
Code and lessons from earlier agents on these arms, on other tasks; each has an `INDEX.md` or a `run.py` to start from:
{REUSE_LIST}
- Decide from the indexes first. Open a card or a file only for a skill you may use.
- For every skill you use, state APPLY (as is), ADAPT (say what changes) or REJECT (say why) in `notes/reuse.md`. Copy
  what you apply or adapt into this workspace (e.g. `programs/lib/`), keep its origin in a comment, and commit it: your
  programs run from this workspace's own files only.
- That code drove the robot itself. Here all robot I/O goes through `RobotSession` (`rlpj.robot`): reuse perception,
  grasp geometry, task logic, parameters and lessons, not Kortex, camera, executor, watchdog or live-view code.
- Those tasks differ from yours, and some of it ran on the other arm: derive your own task rule, and regenerate every
  robot-specific number (offsets, heights, gripper readings) on this robot.
