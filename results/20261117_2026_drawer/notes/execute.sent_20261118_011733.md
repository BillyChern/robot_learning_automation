ORIENTATION IS NOW GUARANTEED BY THE RESET (00:20). Attempt 14 inserted the drawer backwards again: your grey score
for the pull wall was 0.93 on wall B but under your threshold, so "pull wall None", and the episode inserted anyway
(the user saw it). Your own images show the pull clearly (orient_rot90.jpg, orient_rot180.jpg). From now on the
orchestrator checks after every reset, with the vision judge on a full-size side image, that the drawer is upright
with its metal pull on the wall facing the robot (away from the cabinet); otherwise the reset counts as failed and is
redone. So at the start of every episode the pull faces the robot: grasp that wall (or the drawer by its side walls)
and insert the OPPOSITE wall (the back) first, so the pull ends outside. Remove the turning step. If your perception
ever disagrees, do not insert: end the episode as a failure.

FREEDOM AND RESEARCH (user, 00:15): any RL algorithm and any simulation you want, if needed. You may do deep research
to find papers and implement them to improve performance: WebSearch and WebFetch are now allowed in your sessions.
The user recommends looking at SGS-RL (CoRL 2026): it adds resets to PPO so that a really simple reward is enough
for sim-to-real; the user says it works really well and is worth trying, but you are not forced to use it. Read the
paper first and say in notes/ what you take from it.

Resets stay the reset agent's job (no extract.py of your own).
