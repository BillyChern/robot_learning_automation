Operator note 00:50 (library): the force stop fired 4 times tonight at 65-67 N on approach moves (pinch approach x2,
down_slide, and a 61.9 N escape). You can now lower the limit for a single move: set `s.executor.force_limit = 20`
(or 25) before an approach or press into an unknown contact and restore it after (`s.executor.force_limit =
s.executor.force_cap`); a contact then stops at ~20 N instead of ~65 N. You cannot raise it above the task's 65 N cap.
Use the low limit for every final approach toward the drawer, knob or table; keep 65 N only for the deliberate
insertion push and the pull-out of a stuck drawer.
