Operator note 18:30 (after attempts 24 and 26 and episodes 0-4: the drawer always rests on the rim, bottom 0.095):
1. The hover look cannot work with this wrist camera: in attempt_26/wrist_align2.jpg the held drawer fills the bottom
   of the image and the cabinet is hidden under it (only the table beyond is visible). Measure the opening BEFORE the
   grasp (wrist_opening.jpg is sharp) and fix the detector: it returned "edge not found" in nearly every episode, so
   OPEN_C [0.4519, 0.0029] (one old measurement) is used; the yaml target_centre [0.4469, -0.011] differs by 1.5 cm.
   One of them is wrong by about the clearance.
2. Since we do not know which, add a touch search that cannot miss: from the nominal pose, lower until contact (low
   force), and if the bottom stops above rim - 1 cm, lift 4 mm and try the next point of a grid/spiral over
   +-15 mm along and across the opening (4 mm steps, centre first) and yaw +-3 deg; the first point where the drawer
   drops more than 1.5 cm below the rim is the opening. Log the found offset and use it as the start of the next
   episode (and as the RL action's centre). Cost ~2 s per touch.
3. Then let the RL loop resume: the episodes are the data; long gaps of test attempts do not move the return curve.
Operator note 18:32: the lying-cabinet SIM is done (repo sim/drawer_insert, README section "Lying cabinet"; numpy only,
loads in the robot .venv: sys.path.insert(0, "<repo>/sim"); from drawer_insert.lying_policy import Policy, Descent;
from drawer_insert.lying_baselines import Guided). Use it as an option, compare on the robot:
- Key finding (sim, and the learned policy's strategy): when the drawer rests on the rim, the SIDEWAYS contact force
  (base-frame estimate minus f0 at the start of the descent) points toward the opening (cosine 0.95). Stepping 2-4 mm
  along f_xy each touch beats a blind spiral (59.6% vs 14% for the hand-written "Guided" baseline; the trained policy
  74.8%; 94.8% if the opening is known to +-3 mm). Check this on the robot first: rest on the rim on purpose and log
  f_xy direction vs the direction to the opening.
- Geometry from the sim worker: drawer ~0.091 m front-to-back, footprint ~0.073 x 0.110, so ~1-2 mm per side in the
  opening; the compartment floor is ~5 mm above the table (not 13). Flush top = 0.095-0.097.
- Descent(Policy(".../weights/policy_lying.npz"), centre, axis_deg, rim_z, yaw_straight, est, tool_xyz, tool_yaw, f0):
  .next(f_base) -> (xyz, yaw) target; .done(peak, stopped, tool_xyz, tool_yaw) after each move. Frames, signs and
  what to log: README "Lying cabinet". A 40 N stop is advised for repeated touches.
