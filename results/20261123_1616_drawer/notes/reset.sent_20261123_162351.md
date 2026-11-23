Operator brief (new experiment 20261123_1616_drawer, task drawer_lying; user wants it fast and fully recorded):
The cabinet now LIES ON ITS BACK, compartments open to the top (geometry in the task yaml, fixtures.cabinet: centre,
yaw, size, rim_h 0.095, compartment 0.076 x 0.115, target_centre = the compartment nearest the robot). Your job:
take the drawer out of the cabinet (it starts INSIDE the target compartment, pull up, flush: pinch/grasp the pull and
lift straight up ~12 cm) and STAND it on its back wall at the table target in targets.json (beside the cabinet), pull
facing up, gripper clear, arm home. Recover from any state (partly in, tilted, other compartment, lying on the table,
held, arm low). programs/previous_standing_cabinet/ holds the last run's reset code (knob pinch, wrist look-downs,
low-force approaches, escapes) to reuse. Rules that worked: positions from the WRIST camera (look straight down),
low_force(20-25) for every approach into contact, open the gripper and rise before moving away, never move the cabinet.
