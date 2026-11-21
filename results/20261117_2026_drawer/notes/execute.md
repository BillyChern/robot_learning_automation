Operator note 23:28 (episode 31, attempt 92): the grasp held (87.8), good. Two failures after it:
1. Insertion 5 cm off along t: "align: t_span 0.054 too small to trust, tool on the slot centre", yet the held-drawer
   fit said t err -0.051 (687 pts). The drawer went in off-centre and ins12 hit 33.6 N (insert_end_side.jpg).
   A 5 cm error is too big to ignore: with t_span < 0.07, re-observe (move/tilt for a better side view, or use the
   wrist camera on the slot edges) or at least apply a partial correction, and never push in with |t err| > 1.5 cm.
2. The release lifted the drawer: opening only to 0.62 (16 mm finger room), back_off_wall, then up_from_front: the
   drawer came up with the gripper (stop_side.jpg: it hangs in the jaws, tilted, out of the slot; up_from_front 4.6 N =
   the drawer's weight). Open fully (0.0) and wait for the reading before any move; after the first 1 cm up, check that
   the force stays at the empty-gripper level (else stop, open, back off along the jaw axis).
Operator note 00:08: the side camera's drawer outline on the table reads ~5 cm low in y. The reset logs the
same drawer from its wrist look-down footprint and from the side camera (attempts 150-158, "wrist footprint pN: centre
... (side camera ...)"): wrist - side = (+0.010,+0.057) (+0.013,+0.070) (0.000,+0.046) (-0.019,+0.056) (+0.010,+0.052)
(+0.017,+0.070). The reset's table picks only started to hold once it grasped from the wrist footprint (attempt 158).
Your grasps in episodes 29-31 missed by ~3 cm the same way, and the in-hand "t err -0.051" of episode 31 may be this
bias too. Before the front-wall grasp, take a wrist look-down of the drawer and grasp from its outline (copy the
reset's wrist footprint code: agents/reset/programs/reset.py, "wrist footprint"), and check the in-hand t error with
the wrist camera on the slot edges before trusting a side-camera t error larger than 2 cm.
