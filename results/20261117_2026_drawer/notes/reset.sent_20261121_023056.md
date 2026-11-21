Operator note 02:21 (re-sent: reset 177 failed the same way as 173/174, drawer against the cabinet, both end walls refused):
Operator note 01:36 (attempts 173, 174): scene_after.jpg: the drawer sits on the table right against the cabinet front,
in front of the bottom slot, pull wall facing the robot. Both end-wall grasps are refused by the cabinet_front
keep-out (13_above_drawer_wall: keep_out_violations ['cabinet_front'], tips 17 cm up: the gripper body over the
wall reaches into the front box; cabinet dist 0.014-0.026). Grasp the PULL wall instead (the front wall, facing the
robot, ~10 cm from the cabinet: the RL program grasps it this way, tool down, jaws across the wall), drag the drawer
8-10 cm straight away from the cabinet along -N under low_force(25), release, then do the normal end-wall pick.
