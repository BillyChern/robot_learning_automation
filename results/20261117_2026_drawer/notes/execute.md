Operator note 01:50 (episode 35, attempt 99): three grasps missed (99.1, 99.1, 100.0) after refine +0.045.
wrist_front_pre.jpg: the drawer is at the lower LEFT of the view, the left pad at the drawer's corner, the gripper
centre over bare table: the side-camera front-wall point is off both across the wall and ALONG it, so retries along
the wall normal cannot reach it. This is the side camera's ~5 cm bias (my 00:08 note); the reset's table picks only
work since it locates the drawer with a wrist look-down. Please do that now, before the grasp: tool down ~0.40-0.45 m
above the side-camera estimate + (0.010, 0.058), segment the drawer in the wrist image (agents/reset/programs/reset.py
"wrist footprint", re-centre while the outline touches the image border), take the front wall from that outline, then
grasp. Without it most grasps miss and the episodes teach nothing about insertion.
Operator note 01:55 (episode 36, attempt 100): the grasp held and the drawer went partly in, then jammed:
push_end_wrist.jpg shows it skewed in the opening with a corner on the cabinet. "align: t_span 0.015 too small to
trust, tool on the slot centre" while the fit said t err -0.059: the alignment was skipped again. Pushes 06-09 then
pressed at 60-62 N total for four steps without moving (it slides at 2-6 N once free).
1. Jam rule: when a push step reaches > 35 N and the tool advanced < 3 mm, stop pushing: back the drawer out ~3 cm,
   re-measure (wrist view of the slot edges and the drawer's front corners), correct t and yaw, then re-insert. Do not
   repeat full-force steps into a jam.
2. With t_span too small, do not insert on the slot centre: look again (wrist camera on the slot edges before the
   front enters the opening) until the sideways error is known to +-1 cm.
