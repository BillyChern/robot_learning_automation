The bee fell to the floor again in attempt 119. You put it at the temporary spot (0.36, 0.09). It stayed on the
OUTSIDE of the right fingertip: wrist bottom strip 0.123, under your 0.15; squeeze 99.1, because nothing was
between the jaws. The arm then went home, and the bee dropped off the curtain-side table edge.

Table edges, measured from table-height points of the side camera (2026-11-05):
- curtain side: y about +0.19 at x 0.30, +0.21 at x 0.40, +0.23 at x 0.50, +0.25 at x 0.60;
- then a ~10 cm gap to the floor; the curtain hangs at y +0.31..+0.37;
- e-stop side: about -0.21 to -0.29.
So (0.36, 0.09) and (0.30, 0.10), where the first bee was lost, are only 10-12 cm from a drop.

NEW, enforced by the robot library (RobotSession.gripper), not optional: while the jaws hold the bee
(reading 60-92), opening the gripper is refused unless the tool is inside x 0.36-0.60, y -0.22..0.08 and at most
0.20 m above the table. A refused release raises Abort ("release refused ...") and the bee stays held in the jaws.
Your temporary spots (0.36, 0.09), (0.57, 0.09), (0.37, -0.24) and (0.30, -0.26) are all outside that zone.
Please change temp_spot to x 0.40-0.56, y -0.18..0.05, so they are inside with room to spare. Bee targets are now
sampled in y -0.20..0.06.

Wrist colour alone cannot separate a bee on the outside of one finger (0.12) from a bee lying below (0.11).
Confirm every bee release with the side camera instead:
1. rise about 10 cm and move about 8 cm toward the robot base, so the gripper does not hide the spot;
2. take a side-camera reading: there must be a bee-sized blob within 4 cm of the release spot;
3. if there is not, treat the bee as stuck: go back to the release spot, low, wiggle and slide as you do, and
   check again.
Never go home or travel high before the side camera has confirmed the bee is on the table.

If you start a run with the bee held in closed jaws (a refused release ends that way), put it down at a spot
inside the zone first.
