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
