Operator (14:55): collection round 3 starts after the current policy evaluation (about 16:45) and runs while full_v5
trains. These demos feed the next training run, so one rule matters more than any other: keep the wrist in the normal
configuration (joint 4 near 0 deg) for the whole episode, as your READY route does now. Policies are trained on joint
angles. 106 early demos had joint 4 near 180 deg (the same gripper pose, other wrist solution), and the policy
trained on them failed from the look pose for that reason. The converter now drops every episode in which joint 4
ever enters 90..270 deg. Do not change the route to the look pose, its height or how the yaw is chosen: the policy
rollouts start at exactly that look pose.
