
EPISODE 10 (22:09), for the next tuning turn: the push policy's mean actions sit at the edges: dt ~+0.87..+0.90,
dz ~-0.80..-0.88 on every step (yaw ~-0.7), so the tip ran to the sideways bound (dt +2 mm logged = clipped?) and down
to h 0.040, with low force (4-14 N) while the drawer stayed ~1 cm out (front share 0.48). That looks like the tip
pushing beside or below the drawer front rather than on it. Worth checking whether AWAC on ~150 transitions is
extrapolating to the action bounds, and whether the push's dt/dz bounds let it miss the front.

THE LEARNED POLICY IS NOW WORSE THAN THE PRIOR (22:57), for your tuning turn at episode 20:
- Episodes 4-5 (before your AWAC switch, close to the prior): the insertion reached its end (n-OFF -0.036); the push then jammed.
- Episodes 11-14 (attempts 48, 50, 52, 54): EVERY insertion jams at n-OFF -0.049..-0.055 with 40-54 N peaks, ending in
  a force stop or a tracking-deviation abort ("insertion ended with abort/force: no push"); the drawer stays ~5-10 cm
  out (after_side images). The policy mean has dz +0.4..+0.7 every step (LIFTING the held drawer ~1-2 mm per step)
  and large dt (-0.7..-0.86), yaw +-0.7. Lifting presses the drawer into the rail above the slot: that matches the jam.
- The slot readings of all recent episodes are near identical (front 0.46-0.49, flush ~0.2): the drawer never got near.
So the update is moving the policy away from what worked. Consider: re-anchor to the prior (stronger/slower-decaying BC,
or a trust region), bound dz (the slot is only ~1 cm taller than the drawer), check the AWAC temperature / advantage
normalisation with ~15 episodes of all-fail data, and check that the observation still tells the policy where it is.

ARM FAULT IN EVALUATION EPISODE 20 (23:40): your policy's mean action at insert step 13 was advance 0.79, dz +0.97,
yaw -0.97 (both near the bounds) after steps of dz +0.66/+0.28, yaw +0.64 and cumulative yaw +10 deg; the drawer jammed
and the arm's own controller faulted: JOINT_FAULT, actuator 5 FOLLOWING_ERROR (torques 10/34/-50/4/-28/-3 Nm), before
the 50 N / tracking stops could act. The operator cleared it; robot sessions now clear such non-e-stop faults at
start. This is the same regression as noted at 22:57 (lifting and yawing at the bounds); the eval uses the mean.

URGENT (00:05): THE USER IS VERY SHORT ON TIME. Push the hardest possible to get the first real successes NOW.
- The evaluation after 20 episodes was cut after 3 episodes (all fail, one arm fault): no need to measure a policy we
  know is broken. Evaluations are now 5 episodes every 30; your tuning turn is every 5 episodes.
- Facts from your own data: the prior-like policy (episodes 4-5) brought the insertion to its END (n-OFF -0.036); every
  learned-policy episode since your AWAC switch jams ~5 cm short, lifting (dz +0.3..+0.97) and yawing at the bounds.
- So in THIS turn: go back to what worked (prior actions for the insertion, or a strong anchor to them with small
  bounded corrections, dz bounded to ~+-1 mm per step), fix the push so it lands on the drawer front and stops at flush
  (do not recess past ~1 cm), and run a real test episode before you finish. Get one verified success first, then let
  the learning refine. Keep every change small and tested on the robot.
