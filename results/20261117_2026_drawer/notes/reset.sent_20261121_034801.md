USER INSTRUCTION 03:22 (the user, verbatim): "you can push it the side and lie it down to get back to normal; be careful
not to dropping the drawer to the ground from the edge of the table".

So, for the drawer standing on an end wall at the cabinet's corner:
1. Push it sideways, away from the cabinet corner, toward the middle of the table (toward -y and toward the robot),
   in short pushes (3-5 cm each) with closed fingers at mid-height of the drawer (~5-7 cm above the table) under
   self.low_force(25); re-measure with the wrist camera after each push. Stop when it is >= 8 cm clear of the cabinet.
2. Then lay it down so it is the right way up: push its top edge from the open side toward its bottom side, slowly,
   under low_force(25), so it tips onto its bottom (open side up). Lying on a side wall is acceptable too: the normal
   reset (tipped-drawer handling, pull facing the robot) takes it from there.
3. TABLE EDGE: the curtain side of the table is near y = +0.30 (cell keep-in y [-0.45, 0.30]) and the drawer is within
   ~10-15 cm of it. Never push toward +y (the curtain/table edge) or away from the robot past the cabinet; every push
   must move the drawer toward the table centre. If a push would leave the drawer's outline within 10 cm of a table
   edge, do not do it.
