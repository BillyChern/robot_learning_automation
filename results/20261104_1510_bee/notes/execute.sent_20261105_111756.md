The bee was lost off the table edge again, in a reset this time: it stayed on the outside of one fingertip after a
release. Your squeeze test is blind to this case, and the wrist colour can be as low as 0.12.
NEW, enforced by the robot library (RobotSession.gripper): opening the gripper while the jaws hold the bee
(reading 60-92) is refused unless the tool is inside x 0.36-0.60, y -0.22..0.08 and at most 0.20 m above the table.
All 58 of your releases over the bowl so far were inside (x 0.41-0.56, y -0.17..0.05, at most 0.16 m up), so you
need no change for that.
Please add a side-camera confirmation before leaving the bowl:
1. after the release, rise about 8 cm and move about 8 cm toward the robot base;
2. take a side-camera reading: the bee must be in the bowl (no bee-sized blob left on the table, and the bowl
   reading taller or showing the bee inside, as your verification already checks);
3. if the bee is neither in the bowl nor on the table, it is on the gripper: go back over the bowl, low, shake it
   off, and check again;
4. if it is on the table, leave it there, end the run and claim fail.
Never go home with the bee unconfirmed.
