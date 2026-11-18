Your pull detection flips between looks (attempt 42: turn 114 deg, then "pull faces 106 deg away", turn -130, then 17).
Two things that should help:
1. The side camera's first frames after opening are dark (mean brightness 53 vs 122 settled, ~0.5-1 s of
   auto-exposure). Read frames for ~2 s after opening and use the last one before any colour/grey test.
2. You may call the same vision check the orchestrator uses, from reset.py on this machine:
     from rlpj import config; from rlpj.success import vlm_judge
     t = config.load()["task"]
     r = vlm_judge.judge({"after": "<full-size side jpg>"}, dict(goal=t["reset"], success=t["reset_orientation"]))
     r["success"] True/False/None, r["confidence"], r["reason"] (it names the wall with the pull, e.g. "lower right")
   It takes ~20-40 s and judged the pull direction right on every image tested (attempt 14's turned-drawer images,
   reset 9's result). Use it to decide the turn (ask it which way the pull faces) and to verify before claiming.
The orchestrator restarted at 00:42 (exposure fix in its own check); your repair continues on the next failing reset.
