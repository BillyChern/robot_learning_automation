Operator note 15:30 (geometry, approved by the user): the slot was ~2.3 cm off along the front. Two wrist views agree:
reset attempt 204 (05:07, aim t -0.055: pads at the opening's right edge, opening centre ~19 mm image-left = +t, judge
"knob left") and RL attempt 114 (drawer 2-3 cm on the -t side of the opening when aimed at -0.0585). The task yaml now
has slot_t [-0.0735, 0.0015] (centre -0.036, knob ~ -0.036) and every cabinet keep-out box moved +22.5 mm along t
(xy -0.0084, +0.0209); the front plane (0.5283) is unchanged. Re-read the yaml and discard slot-relative values from
before 15:30. RL agent: still measure the opening with the wrist camera before each insertion (my 15:20 note) and
report the residual offset; reset agent: report your knob measurement on the next pinch.
Operator note (user's new scheme, near-goal / reset-free RL, rl.near_goal in the task yaml): resets no longer go to the
table. Your targets.json now holds {"drawer_out": d} with d in 0.03-0.05 m: leave the drawer IN the bottom slot, the
right way round (pull out), roughly straight, with its front d in front of the cabinet front (+-2 cm), gripper clear,
arm home. You are called (a) after an RL success (drawer flush: pinch the knob and pull it out d), (b) when the drawer
is no longer partly in the slot (start states outside 0.5-9 cm out), (c) after 3 failed episodes in a row from the same
state (jammed/cocked: straighten and pull to d). The orchestrator checks it with the side camera (front d +-2 cm out)
and the vision check (rl.near_goal.orientation). Use your knob pinch-and-pull (short pulls), low forces on approaches.
Update programs/reset.py for these targets and test offline; table/full resets are only needed if the drawer leaves the
slot.
