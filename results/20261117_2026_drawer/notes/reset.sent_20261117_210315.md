The "keep_out_violations: ['cabinet']" refusals come from a keep-out box the orchestrator added for this task (task
yaml `keep_out`): no gripper point may be more than 1 cm behind the cabinet's front plane, anywhere across the cabinet's
width (t from -0.155 to -0.005 m along the front) and up to 0.305 m above the table. The model's gripper points are the
fingertips plus a 45 mm-radius cylinder around the tool axis from the flange to the knuckles (`rlpj.robot.sim.gripper_points`).
So with the gripper vertical, the tool axis has to stay about 5.5 cm in front of the front plane while the gripper body
is at cabinet height; a tilted gripper (pointing forward and down at the pull) keeps the body out of the box. Check your
segments offline with `rlpj.robot.sim.check` (it uses the same box) before a real run. The box stays: it keeps the
arm off the cabinet.
