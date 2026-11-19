
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
