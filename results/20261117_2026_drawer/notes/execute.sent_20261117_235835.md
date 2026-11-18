Force stop fixed (23:47): attempt 7's retreat (`push_back`) was stopped at 51.1 N although it was letting go. The stop
compared the force with its value at the segment start, and that start was under the push's ~50 N preload. It now
also needs the total estimated force above 50 N, so releasing a preload no longer stops a segment; new contact or
pressing harder still does. Your arm is still on the drawer front from attempt 7 (gripper closed, drawer ~2 cm out);
a retreat will run now.

ORIENTATION (user, 23:50): the drawer has a front: the wall with the metal pull (small label frame, short tab
below it). It must go into the slot BACK WALL FIRST with the pull facing out, toward the robot. Your episodes grasp
"the front wall" as whichever wall faces the robot; in attempt 7 the wall facing out after the push was plain wood,
so the drawer went in backwards. The task text (goal/success) now says so, and the vision judge counts a backwards
drawer as a failure, so the reward must too. Before the insertion, find the pull wall (side and wrist images: the
grey metal frame) and rotate the drawer so the pull ends up outside; put it in your observations or prior.
