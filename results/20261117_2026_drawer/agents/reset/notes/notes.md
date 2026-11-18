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
- attempt 25 (insert.py real): pull face found via hint, wall refine ok (step 132), front wall held (88.2), carried,
  slide-in stopped at 83 % by the 25 N contact-force limit (friction ~20-25 N, as during pulls); drawer left
  partly in the slot, gripper on its front wall. -> insert.py resumes: open, escape up, push flush via tab pinch.
- attempt 26: insert resume: drawer sticks out 0.089; tab pinch shifted that far is out of reach (j3). Offline: shift
  <= 0.07 reachable when routed done-hi -> done-pre -> line out. -> front-wall top-down push to 6 cm first.
- attempt 27: front-wall push-in jams at 28 N (> 25 N contact limit) -> flush insertion not possible within the limit;
  done-state setup skipped.
- attempt 28 (real, reset.py commit 577861e): from the drawer stuck 8.5 cm out with the arm on its front wall:
  open/escape/home, state 'out', front-wall grasp, slide-out 20 cm (24.2 N!), carry, place, err 1.3 cm, orchestrator ok.
## Status
- Hardware-verified: scattered reset (attempt 21, 1.7 cm), out-state reset (attempt 28, 1.3 cm), pull of a flush
  drawer by the tab pinch (attempt 17, offset (0.002, 0.0015), 22 N), extraction+carry (18, 28).
- Unverified as one run: flush done state -> table (pull + extract + place); the 12 cm lift after the pull is untested.
- Risk: drawer friction 20-26 N vs the 25 N contact-force stop (pull, slide-out).
- attempts 29-30 (orchestrator resets): crash before the session: --targets is a FILE path, reset.py did json.loads on it
  -> load_targets() accepts a path or inline JSON, parsed inside the session (summary written on failure).
- 22:30 operator: contact-force stop removed (rl.force_limit_n null).
- attempt 31: targets file OK; a single bad j5 sample (360.0 vs plan 312.9) aborted a segment as tracking deviation -> one retry of the table pick after lifting.
- attempt 32 (real, commit 8737b8d): targets as a file, table pick (refine +1.6 cm, grasp 88.2), placed, err 1.5 cm, orchestrator layout ok.
- 23:50 operator: drawer must lie with the PULL wall facing the robot (+-30 deg); the slot held it BACKWARDS (plain wall
  out, protr ~0.014; attempts 36-37 pinched for a tab that was not there). Force stop now needs > 50 N total.
- attempt 38 changes: state 'slot_back' (protr > 0.009); pull_back(): oblique clamp of the protruding wall's top edge
  (jaws in the vertical plane through the slot centre, tilt 40, pinch centre 35 mm along z), pull 6.5 cm, lift 10 cm;
  place() generalised: held = grasp_off (centre->grasp) + pull dir; turn so the pull wall faces -N rotated by delta;
  detect_pull(): grey (S<75, V>70) side-camera pixels on the depth-segmented drawer -> side of the pull; check loop
  re-picks if the pull faces > 40 deg off the robot side; final claim includes it.
- 00:20 operator: orientation now judged by a VLM on the side image. attempt 40: backwards pull worked (8.7 cm out),
  extract's direct joint move from home failed keep-in -> extract goes via READY. attempt 41: next reset started
  'out' with the drawer backwards; the backwards flag lived only in the previous run -> pull placed away. Side-image
  grey-pixel detection was wrong (said -y).
- New orientation sensing (attempt 42): tab_score() = share of metal-tab pixels (S 60..175, V<165) in a band outside
  a wall in a wrist image (offline: attempt 28 pull wall 0.10, attempt 41 plain wall 0.00). Used at the front-wall
  pre-grasp view (decides backwards per run) and in overview_pull() (28 cm above the table drawer, 4 bands, second
  view at yaw+180) for table picks, the correction loop and the final claim.
- attempt 43: wrist overview tab scores flip between looks (judged wrong). Operator: use rlpj.success.vlm_judge.judge
  (the orchestrator's check). -> vlm_pull(): Q1 = task reset_orientation on a full-size settled side image; if not,
  Q2 = "pull on the {image-quadrant} wall" for the other visible wall; else assume a far wall -> turn 180, re-check.
- attempt 45 (real): judge-driven 180 deg turn worked (Q1 true x3, orchestrator layout+orientation ok) but err 0.046 -> re-picks missed (refine shift 0.033 rejected) -> abort. -> GOOD 0.035, refine max 0.04, no abort on pick count.
- attempt 46 (real): OK (orchestrator layout+orientation ok) but Q1-true gave pull=None -> first placement turned it away, fixed in the next round. -> Q1 true returns the robot-facing wall as the pull.
- attempt 47 (real, commit 97f960e): one pick, turn 4 deg (pull kept toward the robot), err 1.2 cm, judge conf 0.85, orchestrator layout+orientation ok, 183 s.
- attempts 50-51: arm left pointing down 3.5 cm in front of the cabinet (gripper 45.9 around the drawer front wall); straight-up escape refused (body ring at the 1 cm front keep-out) and escape allowed only straight up -> open fully, then up / back 4 cm + up / back 6 cm then up.
- attempt 52 (real): escape OK; tab score 0.0 -> treated as backwards and flipped, judge then said the pull was hidden
  (tab score likely a false negative); 3 table picks missed: refine took the table->outer-face step (oblique face)
  and moved the grasp 6 mm the wrong way (wall ~3 cm off). -> refine uses the drop into the dark interior (inner
  edge), table grasp opens to 72 mm; 'out' state orientation asked to the vision judge (tab score only as fallback);
  escape tries straight back along -N first (operator: moves out of a keep-out box are allowed now).
- attempt 54 (episode reset): placed at (0.413,-0.024) for target (0.401,-0.03), turned 34 deg; corner almost at the cabinet -> orchestrator 'not found' (blob likely merged with the cabinet). -> place() picks the yaw keeping the footprint >= 5 cm from the cabinet, else nudges the aim <= 2.5 cm away.
- attempt 55: same spot as 54 already within 1.7 cm -> no motion; orchestrator found it this time (54 was a borderline merge). Clearance logic untested -> attempt 56 target (0.43,-0.05).
- 02:00 operator: attempt 54 was right; orchestrator's blob merged drawer+cabinet at (0.401,-0.03). Layout y now [-0.20,-0.08]. Offline: worst best-yaw cabinet clearance over the new range computed; clearance-aware yaw kept (no nudge needed there).
- attempts 64/69/72/73 (episode resets): (72) after a flipped extraction the drawer was found 22 cm from the
  placement (slipped from the one-wall grip during the 175 deg carry turn) and table picks at (0.32,+0.04) missed
  on an inflated blob outline; the 'out' judge photo was taken with the arm at READY (null). (73) judges disagreed:
  pull 10 deg off -N but ~40 deg off the robot base. -> drop checks before/after the carry, table OBB from rim points,
  wall refine window +-7 cm / shift <= 6 cm, pull aimed at the robot base (clipped +-25 deg of -N, +-5 jitter),
  'out' judge photo from home.
- attempt 74 (real, commit 91cb5d9): table start, Q1 no / Q2 'pull on the lower right wall' (conf 0.88) -> turn -103 deg, rim OBB 0.124x0.098 + refine (inner edge, step 77), grasp 89.1, placed 1.1 cm, pull toward 177 deg, judge yes, orchestrator layout+orientation ok, 194 s.
- attempts 75-76: room dim (side image mean 77 vs 116, wrist table V ~16). Every judge look: 'no pull visible'.
  Brightened images show the tall front panel facing image-left (toward the robot base): edge-on to the camera,
  pull invisible. -> pull aimed at -N + 20 deg (+-5) like attempts 46/47 (seen, judged lower left); judge images
  gamma-brightened to mean ~120 when dim; wall-refine step threshold relative (0.3 x table).
- attempt 77 (real): brightened judge image -> tab seen on the left (robot-facing) wall, success without motion; orchestrator ok. -> table OBB length clamped to 0.155 when the outline is < 0.145 (partial view).
- attempt 78 (real, dark): wrist image mean ~4 (gripper shadow fakes the 'dark interior'): the relaxed refine
  threshold accepted a bogus -3.3 cm shift -> picks missed; judge flipped on the attempt-77 scene. -> refine skipped
  when the wrist image median < 40 (absolute 50 step threshold restored); 'hidden' verdicts get a second Q1 look.
- 14:50 operator: lights were off 04:30-14:50 (side brightness 76); back to ~122. Drawer flush in the slot, correct way round. Dark fallbacks stay (inactive in light).
