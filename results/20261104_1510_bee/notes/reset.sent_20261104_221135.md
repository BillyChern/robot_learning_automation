Your reach limit of 0.62 m is too conservative; the bee at (0.63, 0.04) that you are about to see is within reach.
I checked tool-down IK on the fitted model (rlpj.robot.kin, joint limits with a 3 deg margin, yaws every 15 deg):
- r <= 0.70 m: every yaw solves at tool z 0.037, 0.08 and 0.15;
- r 0.72 m: still solves at the table (tool z 0.037-0.10); nothing straight down beyond about 0.73 m;
- at tool z 0.22 (your hover) only up to about 0.69 m, which is why attempt 74 failed IK at (0.75, 0.03).
Please change programs/reset.py: allow objects up to r 0.70 m, and for objects beyond about 0.65 m approach from a lower
hover (e.g. tool z 0.15) instead of 0.22. Check reachability with your own ik() at the heights you will use rather than
a fixed radius. Only stop for a person when the IK at grasp height really fails (beyond about 0.72 m).
