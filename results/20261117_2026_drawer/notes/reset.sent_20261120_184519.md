
ORIENTATION LOOP + COLLISION (18:39), attempts 133-134: the judge keeps saying the pull is on a hidden wall, the program
turns 180 deg each time, and in attempt 134 the turned drawer's place_down near the cabinet (grasp point (0.408, 0.007))
hit something at 58.6 N. Two changes:
1. Find the pull wall DIRECTLY before any turn: take wrist-camera images of the drawer from above and from 2 sides
   (the knob + grey label frame is unmistakable up close) instead of inferring "opposite the robot-facing wall" from
   a side image that may not show it. Turn by exactly the angle that brings that wall to face the robot (90/180/270).
2. Do the turning somewhere with room: lift the drawer, turn and set it down well away from the cabinet (e.g. around
   x 0.36, y -0.15, >= 25 cm from the cabinet face), then carry it to the target area (x 0.36-0.37, y -0.03..-0.01)
   without turning again. Never place or turn within ~20 cm of the cabinet face.
