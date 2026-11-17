
NEW, enforced by the robot library: the work cell has a keep-in box, config robot.keep_in y -0.45..+0.30 (base frame).
Every gripper point (fingertips and the 4.5 cm-radius body) must stay inside it on every planned segment; any
segment that leaves it is refused. The curtain hangs at y +0.31..+0.37. On 2026-11-07 a policy rollout went to tool
y 0.295 at 0.44 m height, grasped the curtain and faulted the arm, and the side camera got dragged out of place.
Your program taught it that move. Your joint-space moves between Home and the bowl-look pose (look_bowl, and home
after it) swing the tool out to y 0.29-0.30 at about 0.43 m height, in 46 of the 119 collection episodes. Rechecking
your last 1500 plans, 98 would now be refused: home 50, look_bowl 38, setup_policy_pose 7, return_home 1, above_bee 1.
Please route those moves inside the box. For example, use a straight tool line (move_line) at constant height, or a
joint waypoint over the table centre (about x 0.50, y 0.0), and check each plan's gripper y in a dry run. Every
recorded move becomes training data for the policy, so keep the paths short and inside the table area.
