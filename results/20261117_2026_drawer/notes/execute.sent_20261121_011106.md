Operator note 00:50 (library): the force stop fired 4 times tonight at 65-67 N on approach moves (pinch approach x2,
down_slide, and a 61.9 N escape). You can now lower the limit for a single move: set `s.executor.force_limit = 20`
(or 25) before an approach or press into an unknown contact and restore it after (`s.executor.force_limit =
s.executor.force_cap`); a contact then stops at ~20 N instead of ~65 N. You cannot raise it above the task's 65 N cap.
Use the low limit for every final approach toward the drawer, knob or table; keep 65 N only for the deliberate
insertion push and the pull-out of a stuck drawer.
Operator note 01:08 (episodes 33 and 34): both crashes ("line cancel_yaw: joint jump 3.7 deg", "line
retreat_clear_back: joint jump 3.8 deg") were a library bug: s.move_line sampled a line by its translation only, so a
turn in place (or a short move with a large turn) got 2 samples. Fixed in rlpj/robot/plan.py (commit "cart_line: sample
by rotation too"); your next program run uses it. Keep the yaw cancel (episode 34 measured +9.1 deg in the hand again).
