# drawer reset notes
- Scene (attempt 2 survey): cabinet front plane per task yaml fits the side depth (front at n-OFF ~ -0.002..-0.004).
  Pull = small metal pin under a label plate on the drawer front, not resolved in side depth.
- Home q [0,15,-130,0,55,90], tool horizontal at (0.58,0.005,0.43). Direct joint moves from home into the cabinet
  region sweep the keep_out box -> go via READY tool-down (0.40,-0.10,0.25).
- Tilted pin pose: seed j4=0, j5>0 (otherwise IK flips the wrist and the path hits the box). j5 ~ 10 deg near
  the wrist singularity -> slow lines there; leave by a joint move to READY.
- tools/offline.py: fake session, checks every segment in the model (run with uv from the repo root).
- attempt 6 (real): pin missed with pad at out 0.013 / h 0.042 / PAD_Z 0.038 (gripper 99.1). Wrist image: pin
  under the label plate, near the bottom of the drawer front; the drawer front is ~ the whole cabinet width
  (slot_t window is narrower than the front). Arm left at the pin pre-pose (return_home failed keep_out).
  -> PIN_VARIANTS, PAD_Z 0.028, retreat back along the approach, escape() + HOME_Q at start.
- attempt 7 (real): missed again at out 0.010/h 0.037. Ray from the wrist image (pin root at px 345,270 of 640x360)
  intersected with the front plane: pin root at h ~0.054, t ~ -0.093, tip ~1.2 cm out. -> PIN_T -0.091,
  variants (0.012,0.052),(0.010,0.058), PAD_Z 0.040 (closed tips stay in front of the face). Perceive after the
  pull only from HOME (the arm at READY occludes the slot: front_pts 7 -> 'missing').
- attempts 9-10 (look_pin.py): 3 wrist views, triangulated (res ~1 mm): pin top edge at the face out -0.011 h 0.034,
  pin end bottom out 0.000 h 0.0225, t -0.088. Pin ~12 mm dia, ~12 mm long, axis h ~0.028. Real face ~1 cm behind
  the yaml front plane (side-depth/fixture vs wrist). -> PIN_T -0.088, variants (0.002,0.029),(0.000,0.034), PAD_Z 0.043.
- attempt 11 (real): missed with tilt 30 pads at out 0.002 h 0.029: triangulated pin projects exactly onto the pin
  in the wrist image (localisation ok); grasp geometry wrong: pin only ~12 mm long, behind the yaml plane, and
  the 22 mm pads at 30 deg tilt overlap only its end; 2F-85 tips also advance ~2 cm while closing.
  -> steep pinch: tilt 50, roll 20, pre-close to 25 mm (PRECLOSE 0.706), fingertip (z 0.041) at out 0.002 h 0.030.
  Tilt >= 55 folds j3 past its limit; release -> vertical lift.
- attempts 12-13 (real): steep pre-closed pinch, pin visibly between the fingers, still closes to 99.1. With the planned
  pose (oracle IK -> robot pose = plan), the pin in the tool frame spans y 0.006..0.013, z 0.039..0.048: only its
  last ~5 mm lie inside the pads (tips at z ~0.044 when closed on it). Cause: keep_out box is at yaml front + 1 cm,
  but the real face is ~1.1 cm behind the yaml front (triangulation res ~1 mm; attempt 13 put the model's forward
  finger edge 9 mm behind the yaml plane with max force 3.5 N = no contact). So the box sits at the real face.
  -> need_operator: shift front_offset/keep_out +0.010 m along the normal (or confirm).
- attempt 14 (diag_pin, stepwise close): the "pin" is a flat tab hanging down-out from the label plate, lying along
  the pad width (tool y). Fingers flank it in the image but close on nothing: the tab passes over the fingertips
  on the camera (+y) side (real finger +y edge appears at ~+0.006, not +0.011). -> tool-frame shift dy +4..6 mm,
  dz -2.5..-4 mm (keeps the model's forward finger point inside the box slack).
- attempt 16 (real): TAB HELD (reading 86.9) at tool-frame offset dy 0, dz +0.003 from PIN_BASE; pull refused by the
  keep-out model because the gripper was re-commanded to 0.90 (model tips deeper). -> re-command reading+0.8,
  sweep (0,.0025),(.002,.0015),(0,.0015),(.004,-.001); refused pull -> release and next offset; open first at start.
- attempt 17 (real, box widened by the operator at the slot opening): tab held at sweep offset (0.002, 0.0015),
  drawer pulled 6.5 cm (force up to 22 N). Then 6 cm lift + joint move to READY swept the fingers into the
  protruding drawer (30.8 N stop). -> 12 cm vertical lift before the joint move (offline: path stays above 0.117 m
  in the drawer zone). Backing along -N from the pinch pose is refused (j3 limit).
- attempt 18 (real): from the 6.9 cm-out state: front-wall grasp (reading 86.9), slide out 20 cm (24 N), carry OK;
  place_down hit the table at 57 % (27 N): bottom estimate clipped at 2.5 cm + tilt. -> release 2.5 cm higher
  with the bottom assumed at 1 cm. Arm left holding the drawer near the target.
- attempt 19 (real): table pick (end wall, reading 88.6) + place works; drawer is ~square (obb 0.154 x 0.14-0.15,
  h 0.074), upright after placing. Errors (blob xy vs target) 0.044/0.058/0.048: blob xy (= orchestrator's
  measure) sits 2-4 cm from the box centre. Orchestrator said "not found": session return_home went back to the
  low start pose above the drawer (occluding it, 15.8 N). -> set_home() makes HOME_Q the session/watchdog home;
  placement aims box centre = target + (box centre - blob xy).
- attempt 20 (real): both table picks missed (99.1): wrist image at the pre-pose shows the wall ~2 cm from the tool
  axis (side-camera box error), outer finger on the wall top. -> refine_wall(): intensity profile along the jaw
  axis at the wall-top height via project_base_to_wrist, strongest bright->dark step = outer edge, e = edge + 6 mm.
  Offline on the attempt-20 image: step 72, edge found where the wall is. Also pick_table goes via READY.
- attempt 21 (real): SCATTERED RESET OK: wall refine shift +1.0 cm (step 125), grasp 87.8, placed, err 1.7 cm,
  orchestrator layout ok. Local config/tasks/drawer_rl.yaml still has the old keep_out: tools/offline.py rebuilds
  the operator's new boxes (slot opening 3 cm, front 1 cm, body from 3 cm).
- insert.py: scene setup for the done-state test (pull side from depth, front-wall grasp, slide in to 6.5 cm out,
  pinch tab (attempt-17 offset) shifted by the measured protrusion, push flush).
