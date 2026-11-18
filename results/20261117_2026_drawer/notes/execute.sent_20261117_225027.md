The user's instruction is no force cap at all, and that includes your own: remove F_STOP (programs/execute.py sets
`executor.force_limit = 40` around learned steps). Your first real episode (attempt 3) stopped at 40.3 N at the
ninth insertion step, with tracking at 0.38 mm, i.e. the drawer was going in and met resistance; the drawer needs
~20-28 N to slide and more where it rubs. Keep reading the force for observations and reward (penalise it there if
you want smooth insertions), but do not stop a segment on it. What still stops motion: model check, keep-in/keep-out
boxes, speed limits, the tracking-deviation stop (2 consecutive samples), and the arm's own protections.
