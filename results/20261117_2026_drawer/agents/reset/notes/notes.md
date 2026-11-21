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
- attempt 80: drawer the right way round but pushed in past flush (face points 53, 2.5-10 cm inside 180) -> 'missing'. -> face_depth from the recessed face, state 'slot_deep' -> tab pinch shifted by the depth, pull PULL+depth; refused pin offsets skip; clear failure message.
- attempt 81 (dry): face depth (p5) 0.013, deeper pinch + pull passed the real model, lift refused (tip 3.8 mm, release 0.55) -> release 0.65; depth now the median (operator: 25 mm). Offline: at >= 2 cm the pinch violates the keep-out (tips 3.15 cm deep / lower finger below the 1.5 cm window floor); feasible only with the tab beyond the tips. -> DEEP_MAX 0.018: deeper -> clean failure with a clear message.
- attempt 83 (dry): drawer recessed 0.022-0.023 m (median): the real keep-out model refuses all 4 tab-pinch offsets (as offline). Program: state slot_deep, tries the deeper pinch (works to ~1.3-1.5 cm, real model accepted 1.3 cm in attempt 81), else fails cleanly: 'drawer not pulled out (... sat X m past flush: the pull tab may be out of reach inside the keep-out)'. Needs a person, or a keep-out allowing ~4.5 cm into the slot opening.
- 16:50 operator: slot opening may be entered 5 cm (body box from 5 cm). attempt 84 (real, before the change): clean stop, nothing touched. -> recessed (depth > 1.2 cm): level fingers (roll 0), tips +7 mm, sweep deeper along z, high pose keeps roll 20; pull only depth + 2 cm (a full pull folds j3), then the wall-edge clamp + judge + extraction. Offline (5 cm boxes): depths 1.5-3.5 cm pass; DEEP_MAX 0.04.
- attempt 85 (real): recessed 2.1 cm; deep pinch approaches passed the real model but all 4 missed (90.0) with 23-35 N: wrist image shows the pads on the label plate (+7 mm too high); miss-path climb to the rolled high pose hit joint limits (arm left at the pre pose). -> tips +3.5 mm (window floor 1.5 cm), sweep dz 3-7 mm, vertical lift on a miss in the deep case; escape from that pose checked offline.
- attempt 86 (real): 2 offsets refused, 2 missed with 29-36 N: forward finger edge on the face (real face ~3.1 cm: side 2.0 + 1.1 offset; dz 3-5 mm put the edge at ~3.2 cm). -> deep sweep uses attempt-17 offsets (dz <= 1.5 mm), shift = depth - 2 mm.
- attempts 85-87 (real, drawer recessed 2.0-2.2 cm by side depth, real face ~3.1 cm): 12 level-pinch tries within the 5 cm slot allowance, all closed on nothing with 21-42 N approach contact. The tab hangs ~7 mm above the slot floor with the label plate right above it and the face behind; the tilted 22 mm pad (17 mm tall) cannot fit around it with mm placement accuracy. -> DEEP_MAX 0.012: deeper recess fails fast with a message (a person must pull the drawer to about flush).
- 17:05/17:20 operator: slot-opening keep-out now 2 cm (prevents RL recessing > ~1 cm); the user pulls the drawer to flush. Offline with 2 cm boxes: flush pinch, recess 0.5-1.5 cm (roll-20 pinch shifted by depth-2 mm, 6.5 cm pull), escape and wall-edge clamp pass. DEEP_MAX 0.015 (deeper fails fast), level-finger variant disabled (DEEP_LEVEL 0.016). Hook into the plate window at 2.4 cm recess: plate ~3.5 cm behind the plane > 2 cm allowance -> refused by the model, not attempted.
- attempt 89 (episode reset 21:17): drawer recessed 2.1 cm again (RL push despite the 2 cm keep-out) -> clean stop as designed; message updated (2 cm allowance).
- 21:20 operator: reset role may reach 4 cm into the slot opening; asked to try a hook. Not attempted: the label frame's window is only its ~2-3 mm stand-off deep (wood face directly behind), the top channel ~1 mm, the tab hangs from the plate bar (no gap above; the wedge below opens toward the floor 7 mm away). The 2F-85 tip (~8-10 mm rubber) cannot engage; 85-87 failed on contact, not on keep-out. Recess > 1.5 cm -> clean stop, person pulls to flush.
- 21:30 operator/user: keep trying physically (reset role may reach 4 cm). -> recessed > 1.5 cm runs DEEP_VARIANTS (level pinch 6 mm less deep, level 3 mm lower, rolled attempt-17 geometry) then press-drag x2 (closed tips on the tab top, press 3 mm, drag 3 cm), re-perceiving after each; wrist images saved. Offline (4 cm boxes): all variants held/missed and both drags pass.
- attempt 92 (real, recessed 2.1 cm): 3 pinch variants + 2 press-drags, nothing moved. Wrist images: the tab sits ~1-1.5 cm to image-left of the finger centre (left finger landed on it; drag tips pressed the face beside it). Camera x = -tool x, so image-left = +t. -> sideways offsets dt +6/+12/+18 mm for the pinch, +12 mm for the drag.
- attempt 93 (real): RECESSED PULL WORKED: pinch with dt +12 mm held the tab (84.7), pulled; drawer 2.5 cm proud. Then 'missing' (no state for 1-4.5 cm proud). -> state 'slot_back' for out>0.009 with a front layer; recessed block runs before the wall-edge clamp; from_recess flag (right way round) instead of assuming backwards.
- attempt 94 (real, commit b3f382a): from 2.3 cm proud: wall-edge clamp (82.5) pulled it out, judge 'right way round', front-wall extraction (88.2), placed, one correction pick, err 3.8 cm, judge yes, orchestrator layout+orientation ok. Recessed chain proven across 93+94.
- attempts 95-96 (episode resets): drawer sticking out ~6.5 cm but skewed ~30-45 deg across the opening (wrist image): front-wall grasp along N missed twice / landed on the wall (50.4 N). -> perceive fits the protruding wall tops (minAreaRect, farthest edge from the cabinet) when skewed > 12 deg: grasp across the fitted wall, wrist refine along it, placement uses the fitted direction. Offline: skew +-30/45 pass.
- 22:14 operator: drawer ~1 cm out (median) was taken as 'out' (3rd-percentile protr 0.07), front-wall grasp hit the cabinet rail. -> slot_med (median face over a wide window): < 4 cm out = slot_near: <= 2 cm tab pinch-and-pull (shift outward; high pose stays at flush), 2-4 cm wall-edge clamp (pull to ~8.5 cm out, lift 10 cm below 2.5 cm out else 5 cm). Offline: tab -1.1/-2 cm, clamp 1.2-3.9 cm pass; drawer-zone clearance >= 11.9 cm.
- attempt 98: the drawer was already ON THE TABLE pressed against the cabinet front (side_start), merged with the cabinet blob -> no candidate -> slot_near -> clamp pulled nothing. Fix: low points (h 4-12 mm) in front of the plane = on the table; candidate from those points (rim OBB) -> pick_table. Also save the side cloud per perceive (cloud_<tag>.npy) for offline replay. Offline pick near the cabinet: 5/5 OK.
- attempts 100-101: drawer on the table along the cabinet front (n 0.445-0.515, t -0.19..-0.04); the end-wall grasp chose the end next to the cabinet side post -> cabinet_front keep-out (plan refused; 101 after a -3.8 cm wrist refine). Fix: pick_table prefers the end farther from the cabinet footprint when either end is < 8 cm from it, and on a refused model check tries the other end.
- attempts 117-118 (00:43 operator: drawer tipped over): 117 start = drawer still partly in the slot, but the near-cabinet
  table detector took its protruding part (W 0.032) for a table drawer; the end-wall pick lifted it while still in the
  slot, the transfer hit 55 N; 118 started holding it (86.9) at 0.2 m and the move home dropped it on its side
  (blob 0.106 tall). The turn itself was not the cause. Fixes: near-cabinet candidate needs W >= 0.05; at start a
  held drawer is lowered to tool z 0.10 and released before any move; a drawer taller than 0.088 (on its side) is
  tipped upright: judge says which side is open, closed fingers push the top edge (h-1.2 cm) from the open side toward
  the bottom panel (pivots on its bottom edge, lands bottom-down); slide 6 cm first if the landing zone hits the cabinet.
- attempt 120: closed-finger push at the top edge only slid the tipped drawer 1.5 cm per push (max force 5 N: the
  tip barely caught the top edge). Replaced by pinch-and-roll: fingers fully open straddle its 7 cm depth near the
  top, close, lift 1.2 cm, roll 40-45 deg about the bottom edge on the bottom-panel side (pivot path, cartesian or
  short joint moves, whole sequence model-checked first for both yaws), let go -> falls bottom-down.
- attempt 121 + 01:01 operator: pinch across the tipped drawer missed 3x (100.0; the ~7.5 cm depth vs 85 mm stroke
  + 1-2 cm side calibration: a finger landed on the edge and slid it 3 cm each time) -> drawer pushed off the side
  camera's view (y ~ -0.28; state 'slot_deep' from the empty slot, face depth 0.04). The judge contradicted itself on
  the open side (120: away, 121: toward). Now: open side from depth (camera-side face: points >= 2 cm inside the face
  plane = open; 120/121 clouds: ~1580 face / 11 inside -> solid bottom panel facing the camera, open side away);
  hook (operator option b): closed fingertips, tool tilted 70 deg (approach azimuth 0/+-35/+-60 to keep the body off
  the cabinet), 2 cm under the upper wall's free edge, lift along an arc 60 deg about the far bottom edge; slide first
  (low push on the solid face) if the upright landing spot would be out of view (y < -0.22) or at the cabinet;
  wrist-camera look (brown blob on a table grid) when the side camera sees no drawer, slide it back to y ~ -0.15.
- 2026-11-20 (resume): side camera re-calibrated, cabinet moved (+8.5, -46.2) mm in the base frame; yaml front_offset
  0.5133, slot_t [-0.166, -0.091], keep-out shifted, layout y [-0.210, -0.126]. All along-front numbers now come from
  the yaml: scene.SLOT_C (slot centre, was -0.0825), scene.CAB_T (cabinet t-span from the keep-out box 'cabinet',
  now -0.201..-0.051), PIN_T / T_C slot-relative, perception windows slot-relative. tools/offline.py uses the yaml
  keep-out (reset role). 01:12 rules: slides only toward the centre line y ~ -0.1 (slide_ok), landing zone off the
  cabinet and y >= layout_y0 - 0.05 (landing_ok), righting stops after 2 cm unintended drift (DRIFT_MAX).
  Offline: offline.py 5/5, deep variants + drags OK, clamp/back OK, escape OK, hook 3/3, wrist-find slides OK.
- attempt 122: start 'slot' but the drawer was ~2 cm in (operator): new camera reads face depth 0.0156 with 241 face pts / 0 inside (old camera: ~0.021, 50 face / 250 inside) -> classified flush, pinch at the flush pose closed below the tab (wrist image: tab above, between the fingers). Fix: face depth > 0.008 = slot_deep (recessed variants). Model: at depth 0.0156 the missed-variant lift back to the rolled high pose hit joint limits -> deep variants lift straight up and pull depth+2 cm like > DEEP_LEVEL. PIN_T/T_C were already slot-relative (2fa22a8); insert.py T_C and survey.py window now slot-relative too.
- 15:52 operator (attempt 122: 4 pinches closed on nothing): wrist images show the tab ~100 px image-left of the pads
  = ~+11 mm along t (same as the +12 mm of attempt 93). New: measure_tab at the first approach of each pinch call
  (grey low-saturation run on the wood between label frame and fingertips, rows 470-650 of the full wrist image;
  pad centre / px-per-m from the wrist projection, fallback 625 px / 9500 px/m); |offset| > 2.5 mm -> back off,
  shift the aim, re-approach (model-checked), check image. Held tab position saved to notes/tab_position.json
  (slot-centre relative) and used as the starting offset by later runs. Offline: 122 crops -> +11.1/+11.0 mm,
  attempt-93 crop (held) -> -1.1 mm; offline.py 5/5; deep variants 8/8.
- attempt 124: new side camera: the drawer at the layout targets sits at the bottom edge, its lower-right wall cut off (judge q2 'can't see'); each check guessed 'far wall' -> 180 deg turn; 3 picks never showed the pull on the robot-facing wall -> it is on a side wall (180 swaps the side walls). Fix: guess sequence far (180), camera-side side wall (90), far again; covers all 4 walls in 3 picks.
- 16:35 operator: new target area x 0.36-0.37, y -0.03..-0.01 (whole drawer in the moved camera's view), drawer
  >= 17 cm in front of the cabinet face. place(): centre >= FACE_GAP_MIN 0.17 from the face (else aim further out
  along -N); default blob bias (0.004, -0.005) from the new camera (attempt 124) instead of the old (0.02, 0.03).
  landing_ok: y -0.17..0.05 (camera cuts off y < ~-0.18). slide_ok: no slide toward the -y edge (the side it fell
  off), end must be a valid landing spot, <= 12.5 cm. Offline: offline.py 6/6 with the new targets, deep 8/8, hook 2/3
  (the 3rd needs a slide toward the cabinet front: refused, needs a person).
- 18:14 operator / attempts 131-132: right-way drawer ~2 cm out after an RL episode. 131: median 2.1 cm -> clamp
  ("backwards" path), 2 misses. 132: median 1.8 cm -> tab pinch; wrist measurement 34 mm (capped 25), the 9.7 mm
  rest seen by the check was not applied -> 4 misses (knob left of the fingers in the images). Fixes: slot_near and
  start slot_back -> tab pinch-and-pull (right way round by default); backwards only if a close wrist view sent to the
  judge shows no metal pull; tab measurement iterated up to 3x, cap 40 mm, run must have wood on both sides
  (rejects table/tape and frame edge), >= 3 rows; protruding drawer pulled only to ~7 cm out; lift after the pinch
  tries straight-up / back-up / shorter options. Offline: tab pull 1-6 cm out x dt 0/34 mm all feasible when held;
  images 122 -> +11 mm, 132 first view +27 (offline scale), 132 check/closed -> +9 mm.
- attempt 133: tab centring converged (34 -> 5 -> 3.5 mm) but the pinch still closed on nothing (knob overlapping the left finger in the image); the cocked drawer then came out with the end-wall pick of its protruding part (near-cabinet candidate) - but the orientation judge used the task's reset_orientation text, which also requires 'on the table away from the cabinet': at check1 the pull DID face the robot (drawer still at the slot) -> 'no' -> 90/180 turns made it backwards. Fix: ORIENT_Q asks orientation only.
- attempt 134: 2 end-wall misses + a marginal grasp (96.5) whose drawer hit the table on place (58 N). Cause: L forced to 0.155 when < 0.145 (old camera partial outline); the moved camera measures the whole drawer 0.12 x 0.105 -> grasp ~1.8 cm outside the wall. Fix: trust measured L (only < 0.10 -> 0.12); table grasp must read < FIRM_MAX 94.5 (good grasps 86-90 in all past attempts).
- attempts 134-135 + 18:39 operator: side judge kept saying 'pull on a hidden wall' -> 180 deg turns in a loop; the
  end-wall grasps of a drawer near the cabinet missed (outline merged with the cabinet: centre 4.5 cm off).
  New flow when the side judge does not confirm the pull: (1) wrist look-down (0.40 m above the drawer) -> footprint
  from wood pixels on the rim plane; (2) pick and set it down WITHOUT turning at STAGE_XY (0.34, -0.15), >= 25 cm
  from the cabinet face; (3) wrist footprint again + close views of each wall (camera 6-14 cm out, 22-30 cm up,
  looking at the wall's outer face) judged one by one ('the near wall carries the pull'); (4) pick and place at the
  target turning by exactly the angle that brings that wall to the robot side. A placement made with a close-view
  pull is trusted even if the side judge cannot see the pull afterwards (no loops). Offline: look-down at 4 spots,
  all 4 wall views at the stage for yaw 0/30/-40, stage + target placements OK.
- attempt 136 SUCCESS (layout ok, 1.8 cm, pull facing the robot, judge 0.88): staging + wrist footprint worked (footprint centre fixed the cabinet-merged outline), grasps 89.1. The close wall views all came back 'no' although two said the pull/label frame were visible (judge used the reset goal: 'still held', 'fingers in view'); the side judge's lower-right answer then gave the right 90 deg turn. Fix: wall views use a neutral SURVEY_GOAL. Run took 824 s (two staging rounds).
- 20:55/21:00 operator + attempts 140-141: pinches closed on nothing, the tab detector found nothing (rows 0): the
  close views show the fingers at the cabinet's side edge with the knob far image-left (= +t, the sign that converged
  in 133) by 3.6-7 cm. The yaml was shifted at 21:00 to slot_t [-0.201,-0.126] (= -t) and rewritten back to
  [-0.166,-0.091] at 21:13:19 (file mtime); side depth shows the cabinet body t -0.21..-0.049 in both 136 and 141
  (= the restored yaml). Saved tab position deleted. Detector: rows 420-710, whole image width, image-edge flank,
  cap 60 mm (finds 140's knob at the left edge: +57 mm offline scale); judge fallback 'knob left/right of the finger
  gap' -> 12 mm step. Model (current yaml keep-out): pinch aims up to t ~ -0.114 are feasible; a knob at ~-0.098
  would be refused as cabinet front -> if so, the yaml's t is off and the run will log the measured knob t.
- attempt 142 (21:17-21:28): judge (conf 0.88) put the knob LEFT (= +t) of the finger gap twice; first +12 mm step
  (aim t -0.110) executed, the next (aim -0.098) REFUSED by the model: keep-out 'cabinet_front' box at
  t -0.091..-0.051, n >= OFF+1 cm (centre (0.5261, 0.1344)) - the fingers around a knob at t ~ -0.098 (drawer
  recessed 1.7 cm) enter it. Same for all variants; the drags use the old t. => the knob/drawer face is centred
  ~+3 cm along t from the yaml slot (operator 21:00 also saw the face centred 2.9 cm 'further' but shifted the yaml
  the wrong way); the slot window and the opening's keep-out boxes need ~+3 cm in t before the reset can reach it.
- 21:30 operator: yaml slot_t [-0.130,-0.055] (centre -0.0925), keep-out moved (-13.4,+33.4) mm. A pinch at PIN_T held at 21:29 with the new yaml (tab_position.json slot_c -0.0925, t_from_slot -0.0055, depth 0.022): kept. Offline 6/6, deep 8/8 with the new boxes.
- attempt 143 (new yaml, slot centre -0.0925): wrist measurement again +35..+43 mm from the aim (aim -0.098 -> knob
  ~ -0.06), every pinch at the measured knob refused by the cabinet_front keep-out (t > -0.055 now). NOTE: the knob
  appears ~+36 mm from the aim both before (140) and after (143) the yaml +36 mm shift, while the side depth shows the
  cabinet body unchanged (-0.21..-0.05): either the cabinet keeps creeping or something is biased - unresolved.
  tab_position.json of 21:29 was written by a DRY run (held = True when dry): deleted; save_tab now skips dry runs.
  New: knob_t (measured) is used by the press-drag (feasible in the model up to +35 mm from PIN_T); drags interleaved
  after the first pinch variant; high pose ignores large tab offsets (joint limits); more lift options after a hold.
- 21:58 operator asked why the aim did not move. FK of the executed plans: approach_pin0 tip t = -0.1323 (140, 142),
  -0.0971 (143), -0.0855 (144: the yaml was -0.0925 at its start). The knob's image position moved accordingly
  (140: u ~60 px = ~7.7 cm off, beyond the 60 mm cap; 143: +35 mm; 144: +33 mm) -> the knob is FIXED at
  t ~ -0.055 +- 0.005 (aim + offset in all three). The 21:45 yaml (centre -0.0495) matched it; the 21:29 'direct
  measurement' at -0.098 was the dry-run tab file (bug, fixed). Under the -0.0925 yaml every step toward -0.055 is
  refused: e.g. aim -0.051 -> 'tip R+ in cabinet_front (t -0.031, n-OFF +0.022, h 0.034)'. New logging: commanded aim
  t on every try, and explain_refusal (box + gripper point) on refused tab steps.
- attempt 145: 4th measurement: knob at t -0.0547 (aim -0.086 + 31 mm); refusal 'tip R+ in cabinet_front (t -0.034,
  n-OFF +0.023, h 0.034)'. Offline sweep: wall-edge clamp of a 2 cm recessed drawer (tilt 10-70 deg, T_C -0.12..-0.068,
  top 0.065-0.075) never passes the current keep-out (gripper body into cabinet_front at the opening's top rail).
  No pinch / press-drag / clamp reaches the drawer under the -0.0925 yaml. GAVE UP this reset with the evidence.
- 22:12 operator: yaml restored to slot_t [-0.087,-0.012] (centre -0.0495). Attempt 146 (that yaml): the fixed pinch
  high pose (roll 20, 14 cm) -> joint limits -> every pinch/drag refused. Fixes (offline, new yaml): pick_high tries
  rolls 20/0/35/-15, heights, back-offs, IK branches (unflipped wrist first) and pre-checks the joint move, the line
  down AND the way back to READY; retreat-from-cabinet line before READY; held tab pull tries 6.5/4.5 cm + depth
  first (only depth+2 cm passes here); a drawer 2 cm proud after the recessed pull -> tab pinch again (protruding tab
  pulls 1-6 cm all feasible), clamp only as fallback. Offline: deep variants 8/8 + drags, extract+place OK.
- 22:27 operator / attempt 147: measure_tab took the grey table beyond the cabinet edge (run touching the image edge, accepted by the edge-flank exception) -> -60 mm (clipped) -> knob_t -0.103, all later tries aimed into the left cabinet_front box. Fix: runs touching the image edge are dropped (wood needed on both sides), |dt| > 35 mm = not found (-> judge), knob_t and judge steps must stay inside the slot window (1 cm margin). 147's first image now -> -8.8 mm offline scale (~ -11 mm robot) = knob t ~ -0.054.
- 22:34 operator / attempt 148: t now right (measurement -0.1 mm) but the pads closed in front of the knob's end (reading 90) and the press-drag never loaded the tab (1.5 N). New: after the first centred miss, pinch 4/8/12 mm deeper along N (+3 mm higher variants), model-checked, until held; press-drag presses in 1 mm steps (run_segment max_force) until +6 N or 6 mm. Offline: all depth steps feasible, held at 8 mm -> pull 4.3 cm OK; drags OK.
- attempt 149 prep: pull_held() pre-checks pull + a lift afterwards (flush 6.5 cm pull passed but no lift did); high-pose candidates also at the real protrusion. Model at the 22:12 slot: tab path OK for recessed and up to ~2 cm proud; 3-4.5 cm proud has no feasible tab high pose (front-wall extraction from 4.5 cm). Offline 6/6, deep 8/8 + drags.
- attempt 149: deeper pinch HELD (8 mm deeper, 86.5) and pulled the recessed drawer 4.2 cm (2.2 cm in -> ~2 cm proud). Second pinch held twice but all 4.5-6.5 cm pulls refused/no lift; then the clamp held (83.4) and its pull was refused -> stopped clamped on the wall. Fixes: pull candidates down to 2 cm (from 2.2 cm proud: 3.5 cm passes -> ~5.7 cm out -> front-wall extraction); tab pinch-and-pull repeated up to 3x while still in the slot; clamp pulls pre-checked (pull + lift) BEFORE closing on the wall, skipped if none; no feasible high pose -> skip the pinch instead of aborting.
- attempt 150: the pre-checked wall clamp (83.8) pulled the drawer OUT of the slot; table pick + place gave 3.8 cm with the pull facing the robot (judge 0.88), but GOOD 0.035 triggered corrections: 4 end-wall misses (100) drifted it to 9.3 cm. Fix: GOOD 0.045 (tolerance 0.05); wrist look-down footprint before every table pick.
- attempt 151 SUCCESS (layout ok, 4.6 cm, judge 0.9): wrist footprint pick (89.1) + place. Refine shift capped at 12 mm when the footprint is from the wrist (150: 18-24 mm shifts missed).
- attempts 155-156 (commit 03e8fa0): drawer still ~2.5 cm proud IN the slot (judge: 'partly out of the bottom slot'), taken for a table drawer against the cabinet (near-cabinet blob, W 0.07-0.09); 5 end-wall picks each closed on nothing, wrist footprints 'implausible' (no drawer-size wood in front). Fix: slot_or_table() for a near-cabinet table candidate: judge 'still partly inside the slot?' + wrist look-down; in slot -> slot_near (tab path). Also commits the 12 mm refine cap (7a0031a had not landed).
- 23:43 operator / attempt 157: judge said 'on the table' (0.55); 5 more refined end-wall picks missed (shift -0.053..-0.058). The wrist footprint found the drawer (L 0.115, W 0.063) ~5 cm along t from the side outline but was rejected (W < 0.08, partly behind the cabinet mask). Fixes: footprint W >= 0.05 accepted; refine_wall max_shift 0.02 (0.012 with a wrist footprint), larger = rejected; after a refined miss the next pick is unrefined.
- 00:00 operator / attempt 158: success by my measure (4.4 cm) but 6.1 cm by the orchestrator; 151 and 158 both landed ~4 cm toward -y. Rejected refines (+0.033/+0.035) were right. Now: refine accepts +0..4.5 cm (step >= 70) and -1.5..0 cm, closed loop (re-image, <= 3x, residual <= 6 mm), side-vs-wrist wall logged; learned placement bias (measured - aimed, persisted in notes/place_bias.json, seeded (-0.015, -0.040)), face-gap check on aim + bias; GOOD 0.03.
- attempt 159 SUCCESS (layout ok, 0.8 cm, judge 0.9): learned placement bias settled at (-0.004, -0.047); the close wall views found the pull (conf 0.85, neutral survey goal) after a 90 deg mis-turn; closed-loop refine: side-vs-wrist wall offsets -10..-14 mm along d0 this run.
- 00:25/00:26 operator, attempts 160/162: (a) start: gripper 70.3 pre-closed around the knob was not opened and the escape rose with it (61.9 N) -> now: low at start = open fully (0.0) whatever the orientation; tilted pose backs off 4 cm along the tool axis first. (b) approach_pin0 hit 65-67 N: aim t -0.043 came from a stale tab_position.json (+6.5 mm from the slot centre, saved at a hold without a fresh measurement; knob is -0.055 = PIN_T). File deleted; save_tab stores the measured knob t and only in real attempt dirs (my offline tests had rewritten it). Approaches are now force-guarded: last 2 cm in 5 mm steps, > 25 N -> back off 1 cm, re-measure, next offset (also for tab steps and deeper steps). Offline: escape from the 162 stop pose OK; proud 1.3 cm: aim -0.055, pull 4.5 cm OK; deep 8/8; offline 6/6.
- attempt 163: start escape OK (opened, backed out along the tool axis). Drawer fully out on the table beside the cabinet (judge 0.7), but 5 end-wall picks missed: the look-down centred on the side-camera xy showed the drawer cut off at the image bottom (side camera off by several cm near the cabinet) -> footprints 'implausible'. Fix: footprint_look() per look; if the biggest wood blob touches the image border or is implausible, re-centre on it and look again (max 3), look height 0.42.
- attempt 164 SUCCESS (layout ok, 2.9 cm, judge 0.9): look-down re-centred once (drawer was ~9 cm from the side-camera xy: (0.425, 0.202) vs (0.445, 0.106)), first grasp held (89.1), placement error 0.6 cm with the learned bias (now (-0.005, -0.043)).
- 00:38/00:48/00:50 operator, attempts 165-166 (65-66 N on down_slide): the wrist-find slide put the closed tips
  3.5 cm outside the drawer edge but the gripper BODY (4.5 cm radius) came down on the drawer top. Now: push point 6 cm
  outside the pushed edge (edge = blob min over the body width), slide refused -> no slide (no abort). Force stop
  lowered per move (s.executor.force_limit, restored after): 20 N for down_slide, down_to_wall, touch_tab and the
  guarded approach steps, 25 N for the slide push and place_down; a contact there = back off / miss, not a stop.
  Table look-downs start at side centre + WRIST_SIDE_BIAS (0.010, 0.058) and re-centre on partial blobs; side-camera
  points are never used raw for picks (side_biased()).
- attempt 167 SUCCESS (layout ok, 3.7 cm, judge 0.85): the wrist-find slide stopped at the new 20 N limit while coming down (no 65 N stop), the empty slot (face depth 0.032) was then tried as recessed (wasted pinches), the drawer was found on the table, staged, the close wall views found the pull (0.75), placed with the learned bias (now (-0.010, -0.056)). Open: an empty slot reading ~0.03 deep is still taken as 'recessed' when the wrist slide fails; 00:52 addendum (finger nearest the drawer must clear the edge) is covered by the 6 cm offset + jaw axis along the edge.
- 01:10 operator / attempt 168: guarded_line's _near move ran at the full 65 N (deeper_pin_8_3_near 71.1 N, 12_3_near 62.7 N). Now the _near move also runs under the low stop (max(fmax, 30) N; contact = back off 1 cm, miss), and each deeper/higher pinch step first backs out to its own pre-pose (5 cm along the tool axis) and comes in with a 2 cm guarded approach, never sideways into the knob/face. Offline: deeper hold + pull OK (recessed and 1.3 cm proud), deep 8/8 + drags, offline 6/6.
- 01:17 operator / attempt 170: knob centring measured +7.5, +6.8, +5.7 mm (aim walked 2 cm) and every back_for_tab pulled 73-89 N: the pre-closed pads dragged the knob. Now: open to 0.4 before every back-off from the knob (back_for_tab, missed pinches, deeper steps, contacts) under a 25 N stop, pre-close again before each approach; a centring step must halve the offset (else stop, pinch at the best aim), total walk <= 12 mm. Clamp approach refused by the model -> skip the clamp (170 stopped on to_back_pre).
- attempt 171: with the pads opened to 0.4 every back-off from the knob (deeper_pre x5, back_pin0) was refused by the model (rolled pose: a modelled fingertip 3.7 mm above the table < 4 mm margin) -> stuck at the pinch pose. back_off_open now tries openings 0.4 / 0.55 / pre-close with a 2 mm tip margin (still under the 25 N stop). Offline: the 171 case and recessed/flush/proud all complete; escape from the 171 stop pose OK.
- attempt 172: no escape path: with the gripper fully open the modelled outer fingertip sat 3 mm above the table at the rolled pinch pose, every escape line demanded a 4 mm tip margin. Escape lines now use a 1 mm margin (backing out only raises the tips). The offline harness used a lower table than the session; it now uses max(calib, -0.0155). Offline (stricter): escape from the 172 pose OK, offline 6/6, deep 8/8, 171-type pinch sequence completes.
- attempt 173: escape OK. Start 'slot_near' (protr 0.066, median -0.005) but the judge later said the drawer was OUT on the table beside the cabinet; the whole pinch sequence ran on the slot. The look-down (biased start) chose the LARGEST wood blob = part of the cabinet (border) -> re-centre pose refused -> biased side point picks refused (cabinet_front). Fixes: footprint takes the sizeable blob nearest the look centre; at a slot_near/slot_back start the judge is asked 'still in the slot?' - a confident no + a full wrist outline in front of the slot -> table path with that outline (forced_cand used once).
- attempt 174: judge 0.8 'free on the table' (near-cabinet), look-downs at side+bias chose a border blob ~7.5 cm away (cabinet), re-centre refused, the biased fallback point landed in the cabinet_front keep-out -> refused -> abort. Fixes: look-downs from 0.50 m (0.42 if no IK), tried at side+bias AND raw side; both end walls refused -> a miss (no abort), next pick uses the raw side outline.
- attempt 175 SUCCESS (layout ok, 2.8 cm, judge 0.88): the judge said 'in the slot' (0.55) so the pinch sequence ran on the empty slot again (~5 min wasted, no damage); the table pick then found the drawer with the raw-position look-down (re-centred), first refused/biased pick fell back to the raw outline, held 89.1, placed. Open: a near-cabinet drawer judged in-slot with low confidence still costs the full pinch sequence.
- 01:36/02:21 operator (173/174/177: drawer on the table against the cabinet front, both end-wall grasps refused by cabinet_front): drag_away(): grasp the robot-side wall (tool down, jaws across it, 20 N descent), drag 9 cm along -N under a 25 N stop, release, lift; triggered once per run when a pick (table or staging) had both end walls refused. Offline: at the refused position (0.464, 0.185) the drag passes the model; also at 3 other near-cabinet poses.
- attempts 181-182: 181's end-wall pick of a drawer against the cabinet lifted with 40.8 N (far end caught) and the
  drawer ended STANDING ON AN END WALL (11 cm tall, ~7x7 footprint) at the cabinet front; 182: depth test looked at the
  pull wall, no hook pose (IK / cabinet keep-outs). Fixes: lift_drawer_slow under a 25 N stop (contact -> release,
  drag_away, retry); end-standing drawer (L < 0.095, h > 0.10): the judge picks the open side among the 4 walls;
  right_side(o_override); if no landing zone / no hook+push here -> drag_across(): pinch it across (gripper fully open,
  jaws parallel to the cabinet face), drag 10 cm along -N (25 N stop), release, re-perceive, retry; push fallback when
  no hook pose; landing y up to 0.12 (in view). Offline: drag_across feasible at both 182 outlines; after it, righting
  is feasible for all 4 open-side directions; offline 6/6, hook 3/3.
- attempt 183: the judge found no open side (4 'no'), the depth fallback planned a hook (az 60) next to the cabinet and down_to_hook_pre hit 68.7 N (no low stop). Fixes: a tipped drawer within 12 cm of the cabinet face is dragged across/away FIRST (before any righting); hook motions run under 30 N stops with back-off. Offline: escape from the 183 stop pose OK, hook 3/3, offline 6/6.
- attempt 184: drag_across descent stopped at the new 20 N limit (fingers on the drawer top: side outline off near the cabinet); the wrist footprint rejected all blobs (end-standing top 7-10 cm < 9 cm min, projected at 0.085 instead of ~0.11); the program then tried hook/push (infeasible) and stopped. Fixes: tall drawer (h > 0.095): footprint projected at its height (<= 0.13), plausible 0.055-0.13 x 0.04-0.11; drag_across positions from the wrist outline; a failed drag-across ends the run with a clear message instead of infeasible righting tries.
- attempt 185: the drag-across descent hit 20 N again (side outline). The look-downs DID see a drawer-sized outline (0.436, 0.220) 11.2 x 6.4 cm, cut by the image border, ~9 cm +y of the side estimate: beside the cabinet's +t corner, 2.6 cm in front of the face plane. Added: best drawer-sized outline kept across looks; drag-across jaws across the narrow axis (model-checked), else a sideways slide along the face. Offline at that outline: every pinch axis and the sideways slide are refused by the cabinet keep-outs (the pusher/finger would go between drawer and cabinet). GAVE UP this reset: the tipped drawer beside the cabinet corner needs a person (or a pull from the robot side that the gripper cannot get).
- 187/188 review: drawer stands on a side wall at the cabinet's +t corner, centre ~2-3 cm from the face (wrist sighting n 0.487 t 0.042; bias-corrected side outline n 0.508). Wrist sightings scatter by >10 cm, so an 85 mm-stroke pinch on a 62-81 mm footprint has no accuracy margin; a vertical-tool pinch puts the gripper body (4.5 cm radius) into the cabinet face; no face to push it along -N; the sideways slide would end at y ~0.25 (outside the landing zone/camera view), so it is now refused (slide_ok). Gave up: a person has to move the drawer >= 15 cm in front of the cabinet. Arm left parked near the drawer after 188 (return home refused).
- 189-191 cause found: the wrist look-downs (189) show the drawer clearly at (0.445, 0.229) (n-OFF -0.015, t 0.047: beside the cabinet's +t corner, on a side wall, 0.096 x 0.069 footprint); footprint_look zeroed a cabinet zone reaching 3 cm past the cabinet side -> 25-40 % of the drawer cut -> 'implausible'. The side camera put it 11 cm off ((0.469, 0.116)): 188's pinch and 190/191's front-wall grasps hit air. 190/191 also read the tall drawer as 'out' (top 0.099 in the slot window) and ran the slot extraction.
  Fixes: footprint_look labels the unmasked mask first (close+open), drawer-sized blob nearest the look centre not centred in the cabinet body (replay of 189: (0.444,0.226) 0.102x0.069, (0.445,0.230) 0.093x0.069); slot_or_table keeps the wrist outline as the candidate; 'out' with top > TIPPED_H -> judge free-vs-slot -> wrist outline -> table path; tilted drag-across (every tool-down pose there is in cabinet_front: open finger at t~0.028; static sim.check passes only because a move may leave a box it starts in): jaws within 25 deg of the narrow axis, tool tilted 15-30 deg away from the cabinet, approach along the tool axis, drag diagonally (-N - t) 0.16 m to ~(0.382, 0.082), back out, READY; after the drag a wrist look at the drag end, side cloud shifted onto it for the depth test. Offline: drag + right_side ok for both open-side directions.
- 192: start 'out' -> judge free (0.85) -> wrist outline (0.451, 0.226) 0.099x0.069 OK; tilted search hit its 60-check cap (steep tilts: approach-line IK jumps). Fix: mildest tilt first over all jaw angles (+-30), 2 backs x 2 drag dirs, cap 160, failure reasons in summary tilt_search. Offline 192 outline: found after 5 checks.
- 193: tilted pinch held (56.8) and dragged ~5 cm (25 N stop); release FULLY open put the fingers into the cabinet_front post box (z<0.079) -> every lift/back-out entered the upper box -> stuck low. Fixes: release to 0.45; escape back-lines under low_force(30) (offline: escape from the 193 joints OK).
- 03:22 operator (push it clear, then lay it down; never toward +y / within 10 cm of an edge): closed-finger pushes behind the drawer are all refused (keep-in y<=0.30 curtain 3 cm away, cabinet_front keep-out). push_clear: closed-finger push if feasible, else the tilted straddle 'fork' (open fingers either side of the narrow axis, tool tilted 15-30 deg away from the cabinet; moving toward the robot / -y the cabinet-side finger pushes it, no grip) in 5/4/3 cm steps under 25 N, wrist re-measure after each, stop at >= 8 cm clearance; stop if a push brings it closer to the curtain. Offline: 3 pushes -> (0.334, 0.137), 0.104 m clear. Then the right loop (judge open side, right_side).
- 194: escape refused (all 3: cabinet_front) from the 193 pose; offline passes because runs/calib/kinova_a/kin_fit.json is absent here (nominal kin ~7 mm off the fitted one used by the session). Fix: escape retries with the gripper closed part-way (0.45, then 0.6): offline the finger leaves the post box from 0.2 on.
- 195: the part-closed escape (0.45) LIFTED the drawer; the start then saw no drawer, the wrist 'find' saw the held drawer, the slide pushed it into the table (25 N), the empty slot read 'recessed' and the knob approach hit the cabinet with it (66.5 N). Fixes: put_down_held (close fully: < 95 = held -> carry at SAFE_H to the staging spot, lower under 20 N, release); after a part-closed escape, and before any closed-finger push/slide, check; notes/holding_drawer.json marks a run that ended holding it (the start keeps the grip, escapes, sets it down). Offline from the 195 stop pose: OK.
- 196: held drawer set down at the staging spot (touched at 20 N) - free of the cabinet. Then: tipped (h 0.115), open side judged 169 deg -> landing y ~-0.19 below the camera view -> right_side refused -> drag-across fallback (meant for the cabinet) dragged it 0.10 m -N -> lost -> wrist find slid it +y 0.12 -> 'drift 0.075' stop. Fixes: tipped drawer clear of the cabinet at y < -0.13: slide it +y (to y ~-0.07, low push on its -y face) before righting; drag-across fallback only within 15 cm of the cabinet; deliberate moves (drag, find-slide) are not righting drift. Offline: y-slide feasible at the 196 positions.
- 197: hook roll did not catch (lift forces ~5 N), drawer slid 3 cm -> drift stop. 04:09 operator: lay it down by tipping: tip_over() = closed fingers 1 cm below the top edge on the open side, push toward the bottom side in 1 cm steps under 25 N (3 cm run-up + D + 3 cm), landing must be landing_ok and within y [-0.25, 0.20]; the hook roll only as a fallback. Offline at the 197 position: feasible (push -16 deg, lands ~(0.41, -0.15)).
- 198: side depth test flipped the open side (14 deg vs 164 deg in 197: the camera sees the pull wall); o=14 -> tip landing x 0.24 unsafe -> slide + hook (34 N contact) -> drift 2.3 cm stop. Fixes: wrist_open_side (wrist outline, close wrist views of the two narrow-axis faces, judge 'open side, see inside'); tip_over pre-slides along o (2-8 cm, slide_ok) when the landing is unsafe, re-measures, then tips; drift limit 8 cm while the outline keeps y in [-0.30, 0.21] (04:13). Offline: tip plans for the 197/198 cases.
- 199: wrist views judged BOTH faces open; the images are clear: open side (25 deg) dark inside V 21, bottom panel V 139 -> open side by darkness. The top-edge push (1 cm steps) only SLID the drawer ~10 cm (1-3 N, slippery table: tips only if mu > ~0.35). New: pinch_roll: pinch across the narrow axis, roll ~45-55 deg about the bottom panel's lower edge, let go; if the landing is unsafe or the roll is toward the robot (joint limits), lift 4 cm, carry (turning 180 deg when needed) so it lands at target+3 cm x, then roll. Offline: feasible from (0.37,-0.05), (0.27,-0.08), (0.30,-0.09) with the open side at 20 deg. Push/slide kept as fallbacks.
- 200: after the +y slide the wrist outline went to the side judge first (undecided) and the 'located from above only' require fired before the wrist views ran. Fix: wrist_open_side first (darkness, single-view absolute V<60 open / >100 panel), judge only as fallback. 04:23 operator asked for a pinch on the open-side wall's top edge; the drawer lies on a side wall so its top is closed - the pinch-roll grips the whole drawer across its narrow axis and arcs about the far bottom edge in 9-18 deg steps under 30 N (25 N for the approach).
- 201: SUCCESS (err 7 mm, pull toward the robot, layout ok): wrist darkness open side (27 deg, V 20 vs 137), pinch (68.1), carry+turn 180, joint-step roll 45 deg -> upright; staging, pull-wall views, place. 04:32 operator: if the views and judge are undecided, use the side depth test's direction (perceived from home) instead of stopping - added (offline compile only).
- 05:20 operator: cabinet re-measured (+15 mm along N, -9 mm along t; yaml updated). All geometry comes from the yaml (OFF, SLOT_T/C, PIN_T, T_C, CAB_T, keep-outs); no saved knob file exists; a tab file older than the yaml is now ignored; place_bias (robot-frame table placement) kept. New: summary knob_vs_yaml_mm + a decision flag when the first wrist knob measurement is 6-13 mm off the yaml's knob.
- 205: upright drawer (h 0.064, pull toward the robot) at the cabinet's +t corner again, outline reaching y ~0.31 (curtain keep-in 0.30): 5 picks, both end walls refused every time (one end in keep-in, one at the cabinet), drag-away refused: its joint swing from READY to above the wall crossed cabinet_front (the pose itself is clear). Fixes: joint_to_via (waypoint 12 cm toward -N, then a line) for drag-away and end-wall approaches; ready_via; drag-away drags diagonally (drag_dirs, -113 deg, 0.16 m) to ~(0.38, 0.11); free_from_cabinet: drag-away once, else drag-across (tilted search now also shifts the pinch toward -y and allows no tilt). Offline: drag-away feasible at 3 positions; regressions 6/3/3.
- 206: upright drawer at the +t corner reaching the curtain: waypoint approaches reached the via poses but the wall poses stayed refused (gripper body in cabinet_front / keep-in); drag_away came down and touched (contact), tilted drag-across: 160 refused. 12:50 operator: hook inside the open top. hook_inside_drag: closed fingers into the cavity (tips 2.5 cm above its floor, 20 N), drag 10 cm along -N (30 N), lift out; hook points anywhere in the cavity + tilt about x (body away from the curtain), static filter then full model check; a 2nd hook goes diagonally (-N -t). It runs first in free_from_cabinet (upright drawers). Offline: feasible at the 206 positions (hook point shifted to ~(0.436, 0.252)).
