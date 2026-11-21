Operator note 23:12 (attempt 150): extraction worked; the drawer is out, upright, pull toward the robot (judge 0.8),
at [0.44, -0.086] yaw -37, 9.2 cm from the target. Only the table corrections failed: 4 end-wall grasps in a row closed
on nothing (reading 100) after refine_wall shifted the grasp -0.018..-0.024 m "toward the drawer centre" each time
(the first refine, +0.042 near the cabinet, worked). Check the sign and the edge choice in refine_wall on the table;
after one refined miss, retry with the unrefined OBB grasp point (or cap |shift| at 0.01); a fully closed reading means
the pads are inside or outside the wall, not on it. The next start is "drawer on the table": only the correction is
left.
