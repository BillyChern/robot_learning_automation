
EPISODE 10 (22:09), for the next tuning turn: the push policy's mean actions sit at the edges: dt ~+0.87..+0.90,
dz ~-0.80..-0.88 on every step (yaw ~-0.7), so the tip ran to the sideways bound (dt +2 mm logged = clipped?) and down
to h 0.040, with low force (4-14 N) while the drawer stayed ~1 cm out (front share 0.48). That looks like the tip
pushing beside or below the drawer front rather than on it. Worth checking whether AWAC on ~150 transitions is
extrapolating to the action bounds, and whether the push's dt/dz bounds let it miss the front.
