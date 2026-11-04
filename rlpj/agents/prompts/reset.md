## Your job: `programs/reset.py --targets <json>` that puts the scene back for the next episode
- **The task being collected:** {TASK_GOAL}
- **Reset instruction:** {TASK_RESET}
- **Objects:** {OBJECTS}. Identify them by geometry (size, height, shape), never by colour.
- **Targets:** `--targets '{"<object>": [x, y], ...}'` in the base frame (metres), one per object, sampled by the
  orchestrator within x/y ranges {LAYOUT} (radius {RADIUS} m from the robot axis, centres at least {MIN_GAP} m apart).
  Randomisation is the orchestrator's job, not yours. A reset counts when each object is within {TOLERANCE} m of its
  target, measured independently by the orchestrator from the side camera after your program ends.

## Behaviour of the program
1. Perceive (`s.tabletop()`, refine with the wrist camera when needed) and decide which blob is which object.
2. If the scene is in the task's done state (the task has been performed), undo it first, then place.
3. Move each object to its target, slow near objects, re-perceiving after each place. Pick the order so the arm never
   has to put an object where another one still is (move the blocker first, use a free spot if needed).
4. Measure the final positions from the side camera, write `summary.json` with `measured` `{"<object>": [x, y]}`,
   `claimed_outcome` (success if every object is within tolerance, else fail) and the per-object errors, and
   `s.claim()` accordingly. Also write it when the program fails or is stopped.
5. If an object is missing, out of reach, or cannot be grasped after two tries, give up cleanly (`s.require` with a
   clear message): the orchestrator then asks the operator to reset by hand.

## Working loop
1. Read what you need and write the program. Commit, then `run` a dry run (`moves_robot` false) with targets you choose
   inside the ranges, e.g. `{PYTHON} programs/reset.py --targets '{EXAMPLE_TARGETS}'`.
2. Commit, then run it for real (`moves_robot` true). The operator may arrange the scene in the done state or in any
   start state before a run; your program must handle both without being told which.
3. Fix from the digest and repeat until a real run places every object within tolerance from both a done-state scene
   and a scattered scene. Then `done`: afterwards the orchestrator calls `programs/reset.py` with its own targets before
   every episode, and resumes you with the digests if resets keep failing.
