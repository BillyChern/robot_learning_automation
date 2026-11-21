Operator note 00:00, CORRECTION to my 23:43 cap: in attempt 158 the rejected refines (+0.033, +0.035, step 87-88) were
RIGHT. wrist_wall_pre0.jpg: the drawer is at the right of the view, the right pad at its corner, the left pad on bare
table: the side-camera wall point is ~3.5 cm short of the wall. The same ~3-3.5 cm offset showed in the RL program's
grasps (episodes 29-30). The misses in 150/155/156 came from NEGATIVE shifts (-0.05, away from the drawer centre).
So the cap is not the rule; closed-loop is:
1. Accept a refine toward the drawer centre (positive) up to ~0.045 when the step is strong (>= 70); reject negative
   ones beyond -0.015.
2. Move to the refined pre-pose, take a second wrist image, refine again, and grasp only when the residual is
   <= 0.006 (max 3 iterations), as the knob measurement does.
3. Log the side-camera vs the wrist wall position for each pick, so we can see if the side outline has a fixed bias.
