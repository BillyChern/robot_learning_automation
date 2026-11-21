Operator note 22:58 (attempt 149): good progress: the deeper pinch held and pulled the drawer from -2.3 cm to +2.2 cm proud.
1. The arm is parked where the 149 return failed: gripper squeezing the drawer's outer wall (reading 83.4),
   q [348.4, 38.4, 224.8, 11.1, 33.5, 357.9]. Start attempt 150 by opening the gripper fully, then back off along -N
   and up in short model-checked lines (the return planner refused return_home from there).
2. At +2 cm the knob pinch held twice (86.5, 86.9), but every pull (6.5/4.8/4.5 cm) or the lift after it failed the
   model on joint limits. Add short pulls (3.0, 2.0, 1.5 cm) with a release-and-back-off along -N (no lift) after each,
   and repeat pinch -> short pull until the drawer is ~5-6 cm proud; then the wall clamp or a top grasp has room.
   Try another wrist roll / IK seed for the pull if a short one is still out of limits.
3. The wall-clamp pull (48_pull_back_drawer) at 2 cm proud was out of joint limits too: use it only once the drawer
   is further out.
