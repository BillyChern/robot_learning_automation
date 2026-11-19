
KEEP-OUT DEEPER AT THE SLOT OPENING (16:50): the bottom-slot opening (t -0.120..-0.045, 1.5-9.5 cm above the table)
may now be entered up to 5 cm behind the cabinet front plane (was 3 cm); the body box starts at 5 cm, the rest of the
front keeps the 1 cm limit. Reason: the drawer was pushed 2.3 cm past flush and its tab was out of reach. The RL push
(tip up to 2 cm past the plane) can leave the drawer recessed too, so the reset must handle that.

KEEP-OUT BACK TO 3 CM, AND DO NOT PUSH PAST FLUSH (17:05): a drawer pushed ~2 cm past flush (no stop) cannot be
reset by the robot: the gripper cannot reach its tab (reset attempts 85-87; a person had to pull it out). The slot
opening is open to 3 cm again. Make sure the push never recesses the drawer more than ~1 cm (the reset handles up to
~1.5 cm): stop at flush, and count "behind" (recessed) as a failure in the reward like "sticking out".

PREVENTION (17:20): the slot opening keep-out is now 2 cm behind the front plane (was 3). With the model's ~1 cm
fingertip offset, a push can no longer recess the drawer more than ~1 cm, which the reset can undo; a push step that
would go deeper is refused by the model check (status "refused", your -1). Adjust PUSH_NMAX to stay inside it.
