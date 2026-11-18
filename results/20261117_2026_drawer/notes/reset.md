
ANY STATE (user, 23:55): "the reset agent should be able to reset from any scenario, since the RL agent can make all
kinds of mistakes". Make reset.py classify the start state and recover from each of these, and test the ones you can
set up yourself on the robot:
- arm not at home: stopped low beside the cabinet, possibly touching it, gripper open or closed on the drawer.
  Escape in straight lines: open the gripper, back away from the cabinet along -normal first, then up, then home
  (a joint move to home from near the cabinet can sweep into it; check that path against the cabinet box).
- drawer in the slot, right way round (pull outside): flush or partly out; partly out may be cocked or jammed.
- drawer in the slot backwards (plain wall outside): flush (nothing protrudes) or partly out (as now: 1.4 cm).
- drawer on the table: anywhere reachable, any yaw, upright, on its side or upside down.
- drawer still held in the gripper.
Then place it upright with the pull facing the robot. You now get up to 5 repair turns per failing reset (was 2).
A state only a person can fix (drawer off the table, cabinet knocked over): say so in need_operator.
