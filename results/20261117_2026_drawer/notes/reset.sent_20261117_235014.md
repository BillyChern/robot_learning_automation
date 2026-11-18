ORIENTATION (user, 23:50): the drawer's front is the wall with the metal pull. After a reset it must lie on the table
with that wall facing the robot (away from the cabinet), yaw within +/-30 deg of that, so the RL agent inserts it
back wall first. Your reset.md placement did not check which wall carries the pull. Detect it (side image: the grey
metal frame; wrist camera to confirm) and rotate the drawer before placing if needed.
Right now the drawer is in the slot BACKWARDS: plain wall outside, ~1.4 cm protruding, no pull to pinch (reset 5
tried the tab 4 times). Get it out by the protruding walls (e.g. pinch the top edge of the outer wall or a side wall
top), then place it the right way round.
Also: the force stop now needs the total force above 50 N as well as the change from the segment start, so letting go
of a push no longer stops the retreat (commit c30c878). The cabinet was re-measured at 23:28: unchanged.
