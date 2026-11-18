# Operator log: run 20261117_2026_drawer (drawer_rl)

Interventions, stops and mid-run changes by the operator (the user, or Claude Code acting for them). Times are local.

| Time (2026-11-17) | Who | What | Why |
|---|---|---|---|
| 20:15 | user | Cleared an arm fault, released the e-stop, put the arm at Home in the Kinova Web App | arm was in Retract after a fault; programs treat the start joints as home |
| 20:19 | Claude | Stopped the first launch (run 20261117_2019_drawer) after 2 min, before any robot motion; deleted it | `config.load` cached the default bee task, so the RL run had rendered bee prompts (fix 48ee5c7) |
| 20:27 | Claude | Started this run | |
| 20:58 | Claude | Note to the reset agent: geometry of the cabinet keep-out box; the same text added to the RL agent's prompt (462629a) | 3 of the first 4 robot runs were refused by the box, which the prompt did not describe |
| 21:37 | Claude | Keep-out box: bottom-slot opening (1.5-9.5 cm above the table) may be entered up to 3 cm behind the front plane instead of 1 cm (70787e7); note to the reset agent | attempt 16 held the pull tab (gripper 86.9) but its pull was refused: closed fingertips are modelled ~1 cm deeper than the real ones (10.1 mm vs the 10 mm limit) |
| 22:04 | orchestrator | Reset agent stopped at its budget: 25 runs (18 on the robot), $18.68, no accepted done | `agents.max_attempts` |
| 22:05 | Claude | Relaunched run-all: the reset agent resumes with a fresh budget of 25 runs | steady progress on every part (pull-out attempt 17, carry 18, table pick and place 21 success); arm left holding the drawer half in the slot, the agent's next run starts from there |
| 22:12 | orchestrator | Reset agent's done accepted (28 runs, 21 on the robot); RL agent started (develop_execute) | resets passed the layout check from the scattered (21) and part-out (28) starts; a flush start was never run end to end (agent's own report) |

Force stops so far (25 N limit) were all real contacts: finger dragging on the pulled-out drawer (30.8 N), drawer meeting the
table while placing (27.3 N), lifting while still gripping (26.6 N), drawer edge meeting the slot (25.4 N). Free moves: at
most 8.5 N over 49 `ready` moves.
