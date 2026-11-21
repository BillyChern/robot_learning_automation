Operator note 23:24, CORRECTION to my 23:18 note: the 12 mm cap was the wrong fix. Episode 30 (attempt 90): in
wrist_front_pre.jpg the left pad is ON the front wall (x~190 px) and the right pad is on bare table (~440 px): the
perceived wall point is ~3 cm outside the wall (toward the robot), so the needed correction is ~+3 cm along d_in, and
refine_wall measured +0.032, which looks right, but the cap threw it away. Raise the cap back to ~0.04. The open
question is why episode 29's refined grasps (+0.030, +0.037) still missed: compare the episode 29 stop_wrist.jpg with the
pre image (did the shift move the pads inward over the wall, did wall_angle's turn move them, or did a pad land on the
wall top?). Also fix the perception's front-wall point (it sits ~3 cm toward the robot in both episodes 29 and 30),
so the grasp needs no large correction.
