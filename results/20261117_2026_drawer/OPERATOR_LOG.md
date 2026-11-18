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
| 22:38 | user / Claude | Contact-force stop removed: `rl.force_limit_n` null (was 25 N); peak force still logged per segment; notes to both agents | user: "no force cap at all! just get it done"; drawer pulls and pushes needed 22-28 N |
| 22:38 | orchestrator | Two resets crashed in 0.4 s (`--targets` given as a file, the program parsed it as JSON); reset agent asked to repair | contract mismatch the reset agent's own tests never hit |
| 22:50 | Claude | Executor ignores one bad joint reading and stops only when the deviation persists for 2 samples (commit above) | reset attempt 31 stopped on a 47 deg / 230 mm "deviation": joint 5 read exactly 360.0 vs planned 312.9 for one sample; the wrist camera did not move |
| 22:52 | Claude | Note to the RL agent: remove its own 40 N force stop (F_STOP in programs/execute.py) | user: no force cap at all; RL attempt 3 stopped at 40.3 N mid-insertion |
| 22:58 | orchestrator / RL agent | RL attempt 5 (no force cap): after a good insertion to ~4.5 cm out, the release left a finger under the rail above the slot; the 6 cm lift reached 99.6 N, tracking-deviation stop at 9.8 mm; **cabinet moved 6.0 cm**; arm left at the cabinet; RL agent asked for a person | no force stop by the user's choice; the deviation stop is the only contact stop |
| 23:05 | Claude | Stopped the pipeline (Ctrl-C at a safe point, no robot program running) | cabinet off its measured pose: keep-out boxes and the success check no longer match it |
| 23:10 | user / Claude | Contact-force stop set to 50 N (`rl.force_limit_n: 50`); RL prompt says not to add a lower stop; notes to both agents | user: "cap 50", after the 99.6 N snag moved the cabinet; drawer pulls/pushes needed 20-40 N |
| 23:03-23:28 | Claude | Freed the arm (runs/drawers_setup/escape): a first try (back 2 cm, up) hit 51.5 N and its automatic return drove back to the start pose; a joint move to Home from a lifted pose swept into the cabinet top (51.4 N stop); final: straight back 12 cm, up to z 0.40, Home checked against the cabinet box grown by 6 cm, at most 26 N | user: "do it yourself"; the wrist camera was against the shelf |
| 23:30 | Claude | Measured the cabinet: front face 10 mm behind the yaml plane (11 mm before, reset agent's triangulation), top 0.271 m; no geometry change; notes to both agents; relaunched | the "moved 6.0 cm" was the blob centre shifting with the gripper beside the cabinet and the drawer sticking out |

Force stops so far (25 N limit) were all real contacts: finger dragging on the pulled-out drawer (30.8 N), drawer meeting the
table while placing (27.3 N), lifting while still gripping (26.6 N), drawer edge meeting the slot (25.4 N). Free moves: at
most 8.5 N over 49 `ready` moves.
