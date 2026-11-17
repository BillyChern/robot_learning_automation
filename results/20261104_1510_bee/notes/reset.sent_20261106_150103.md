The bee was lost again after evaluation rollout 119.
- The policy grasped it by the tip and the arm went home with the bee dangling (gripper reading 98.7).
- Your start check found it (whole-image share 0.072). Your held_bee path then carried it toward its target
  (0.542, -0.172), and it slipped out on the way: "jaws empty after the carry". It was not on the table afterwards.
Please change two things:
1. The bee is now wrapped in ORANGE (since 2026-11-05). Bee colour is hue 0-40, S > 120, V > 110 (OpenCV HSV).
   In the bottom 15 % of the wrist image at home, measured: bee hanging 0.27, empty <= 0.006. Your old yellow range
   (15-40) reads 0.016 for the hanging bee: use the new range in every colour check.
2. If the bee is in or on the gripper at the start (any squeeze reading < 99.0, or colour above the threshold), do not
   carry it anywhere first. Lower straight down from home over (0.58, 0.00), which is inside the release zone, to
   about 0.10 m above the table, checking first that no blob lies within 0.10 m of that spot (otherwise use the
   nearest free spot of the zone). Open, rise, and only then continue as usual. A bee held by its tip falls out of
   the jaws on any longer carry.
