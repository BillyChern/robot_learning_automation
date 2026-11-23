Operator brief (new experiment 20261123_1616_drawer, task drawer_lying; user wants RL results ASAP, return growth
plotted): the cabinet LIES ON ITS BACK, compartments open to the top. Each episode starts with the drawer STANDING on
its back wall on the table beside the cabinet, pull facing up. Insert it into the compartment nearest the robot
(fixtures.cabinet.target_centre, opening 0.076 x 0.115 m, rim_h 0.095, yaw_deg): grasp the pull from above, lift,
carry over the opening, align with a wrist look-down at the opening, lower straight in (gravity helps) until the top is
flush with the rim, release, rise. A vertical insertion: design the policy around the in-plane alignment (x, y, yaw)
and the descent with force feedback; keep the learner simple and fast (prior + bounded residuals, dense reward from
measured depth and alignment + success bonus), log per-episode return/depth/forces. programs/previous_standing_cabinet/
holds the last run's code (wrist footprint, grasp, force rules, sim-policy hook) to reuse. Low force for approaches,
65 N cap.
