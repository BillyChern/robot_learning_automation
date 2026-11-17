# reset notes
- att1 dry: all segments pass model checks; bee-in-bowl path uses wrist colour-contrast region inside the rim (untested on hw).
- att2 real: bee out of bowl (wrist region found, reading 90.4), bowl rim pinch 94.7 held, bowl placed on target; lower_bee line IK jump 3.0 deg -> exception, arm home holding bee. Added joint-move fallback + held-bee start.
- att4 real: held bee placed, bee re-grasp from table 78.5, final err bowl 4.3 mm bee 10.8 mm; orchestrator layout ok.
- att5 real: swap case; bee temp spot only 0.148 from bowl target -> side blobs merged (ext 0.23) -> 'bowl lost'. Added rim-circle split of merged blobs, wider temp-spot search.
- att7 real: 2nd tabletop (1 s after the 1st) lost the bee at (0.546,0.104) -> abort. Added perceive retries, re-perceive only after a move.
- att8 real: merged bowl+bee blob; split fragment included bowl skirt (ext 0.208) -> bee rejected -> wrist found no bee in bowl -> abort cleanly. Split now cuts r+0.028 and relabels components.
- att10 real: success (layout ok) but blob xy jitters ~2 cm (bowl 0.423->0.399->0.437 untouched; rim fit steady 0.4197); correction made bowl worse. Now: rim-fit centre for bowl, point median for bee, median of 3 final perceptions; perception log -> perception.jsonl.
- att12 real: scattered swap SUCCESS (layout ok): bowl 2 mm, bee 11 mm; bee at temp spot (0.30,-0.26) missed once (side estimate noisy there), 2nd try held. Temp spot now prefers larger x once clearance >= 0.20.
- att13 setup_done: bee dropped into bowl (reading 78.1), side sees only the bowl -> done state.
- att14 real: DONE-STATE reset SUCCESS (layout ok): wrist found bee in bowl (area 0.36), reading 88.6; bowl rim pinch; bowl 2.2 mm, bee 14.6 mm. Bee side xy jitters up to ~3 cm between frames (0.311..0.35).
- reset3 (att19) ok 326 s; the bee from the bowl went to a temp spot 0.148 from the bowl target (it also avoided the bee target, which the bowl blocks anyway) -> extra bee move. Temp spot after unpack now avoids only the bowl now/target.
- reset4 (att20) ok 243 s; bee put down from the bowl 17 mm from target was re-grasped and moved again for no gain (final 13.8 mm). SKIP 0.010 -> 0.018.
- att21/22 FAILED orchestrator check (bee not found / 16.5 cm) though my errors were 13-17 mm: bee blob near x 0.30-0.35 reads extent 0.128-0.147 (diagonal bee + depth smear) -> fails the bee size prior. Fix: put the bee down with jaw yaw 0 (long axis along y); final check = rlpj.perception.layout.check with task priors on every raw measurement (majority must pass), failing objects re-placed in the correction pass.
- att24 FAILED: bee placed long-axis-along-y at temp spot, then 2 grasps missed (99.1, 96.1): jaws now close along x = the side camera's worst axis (~2 cm bias). Fix: wrist look straight down before every table grasp (contrast region vs table ring, bowl masked) -> xy + long axis.
- att25: wrist refine worked (bee held 87.7 twice, refine moved estimate 1.4-1.8 cm); orchestrator layout ok, but my votes flagged bowl extent 0.20 (smear near x 0.38) -> claimed fail. Votes now act/claim only on the bee.
- att26 real swap: SUCCESS, layout ok (votes 3/3); bowl 2.7 mm, bee 11.3 mm after one bee correction; 436 s.
- att27: KServerException in robot FK oracle (transient) -> exit 3; now retried 3x. att27/26: lower_bee took 18 s (joint fallback rotating the wrist 180 deg with the bee) because place_bee lines used yaw 0 while the hover IK chose 180: lines now use the hover's yaw.
- att29 ok 188 s (no wrist flip now). Side reads placed bee +1.6 cm in x consistently -> place_bee aims 1.6 cm short in x.
- att30 FAILED: a phantom blob (0.341,-0.04) h 0.032 n 285 was taken as the bee while the bee was in the bowl; bowl carried with the bee inside, then 'bee lost'. Fix: bee needs n>=450, h>=0.033 (real: n 850-1350, h 0.042-0.049); if the bee is missing mid-plan, look into the bowl and unpack (unpack() helper).
- att32 FAILED: in-bowl bee grasp missed (96.5) then retry rotated jaw 90 deg (closing along the 10 cm long axis) and missed (99.1). Retry now re-looks into the bowl (fresh rim + bee detection) and keeps the jaw across the short axis.
- att35 ok 368 s; first bowl placement 2.6 cm off (side centre jitter) -> correction. Now median of 3 side centres before moving the bowl.
- att36 FAILED (orchestrator bowl 3.2 cm) while my rim-circle median said 7.8 mm: orchestrator uses the blob centre (xy_blob), biased ~-1..-3 cm in x vs rim. Now the bowl is carried by target - median(blob centre) and measured by median blob centre; grasping still uses rim/wrist.
- att39 ok 349 s: bowl correction at err 0.020 overshot (blob 0.495 -> 0.455, target 0.475) -> REDO_BOWL 0.024; real bee at (0.36,-0.19) rejected in 2/5 frames (n<450) -> bee needs h>=0.036, n>=300 (phantom att30: h 0.032).
- att42: orchestrator layout ok but my program claimed fail: first bowl placement measured 3.4 cm by my median while layout votes were 3/3 ok; the correction then made it 4 cm (votes 0/3). Now: no bowl correction when votes pass the bowl in most frames; claim success also when the votes pass in most frames.
- att56 ok but bowl blob median 3.4 cm off (blob centre jitter in the pre-move median); median of 7 frames now.
- att60/61 FAILED: bee taken out of the bowl to temp spot (0.30,0.10) where the side camera sees it only as a 16 mm blob -> 'bee missing' -> looked into the bowl, thin false region (0.08x0.026) -> 2 empty grasps -> abort; next run same. Fix: CLEAR_BOWL 0.135, temp spots x>=0.36 |y|<=0.09..-0.24; reject thin in-bowl regions; wrist search (where we put it + small side blobs) before unpacking / giving up.
- att62: side image shows NO bee anywhere on the table (bowl only); wrist look into bowl: empty (thin region rejected). The bee placed at (0.284,0.10) in att60 most likely fell off the table edge. Needs a human to put the bee back; program fixes (temp spot area, thin reject, wrist search) committed but the full repair path is untested.
- att72/73 FAILED: after release (gripper fully open, 0.4) the foam bee stayed stuck on a fingertip (wrist image att72 search shows it at the bottom-right finger; side image att73 shows it hanging at home). Fix: gripper-clean check after every bee release (saturated yellow fraction in the bottom 15% of the wrist image: stuck 0.12-0.13, bee on table below 0.012-0.02, threshold 0.06) + unstick routine (down, wiggle jaws, slide 3 cm, rise; 3 tries); same check at start -> put the bee down first.
- att74: the stuck bee had dropped off the gripper by itself and lay at (0.75,0.03), out of reach (IK fail). Added an explicit reach check at start. Unstick routine still untested on hardware.
- operator: reach up to r 0.70-0.72 tool-down at table; hover 0.22 only to ~0.69. Now: reach check = own ik() at grasp height + an adaptive hover (0.22/0.15/0.10); look_over lowers its view height when needed; vertical(None) uses the adaptive hover.
- att83 real: SUCCESS (layout ok): far bee at r 0.65 grasped, bowl 22 mm, bee 3.8 mm after one bee correction; gripper-clean check after release 0.025 (< 0.06).
- att87 ok; gripper-clean check read 0.051 (close to 0.06) because the released bee reached the bottom centre of the wrist image; check now uses only the fingertip corners (released 0.0, stuck 0.12).
- att88: orchestrator bee 4.1 cm off while my point-median said 7 mm: orchestrator's blob centre for the bee is biased -2..-3 cm in x here. Bee now measured by median blob centre (like the bowl), REDO 0.020, and a correction aims at target + (target - measured) (clipped 3 cm).
- att90 ok; task tolerance is now 0.04 in the yaml: TOL read from config.
- att91 ok 409 s; compensating bee correction worked (25 mm -> 9 mm) but was not needed with tol 0.04. Correction thresholds now scale with TOL (bee 0.65*TOL=26 mm, bowl 0.75*TOL=30 mm).
- att92/93 FAILED: bee stuck between the fingertips after release (wrist image: bottom centre, corners check 0.000); hanging at home in att93. Added squeeze test (close once: < 97.5 = something in the jaws) after each release and at start; if held, the unstick routine releases it again.
- att94: bee hangs BELOW the fingertips (home wrist image bottom centre, yellow 0.21 full width; clean home 0.007-0.01); squeeze 99.1 and corner check 0.0 missed it. Now: full-width check at home (start + end), after a release a 12 cm shift toward the base then full-width check, stuck/held bee put down from table+0.09.
- att95: bee no longer on the gripper (home check 0.010) and not on the table (side image): it fell off during att94's moves. Needs a person.
- operator: corner mask never fired; full-width bottom-15% yellow with threshold 0.10 everywhere (stuck 0.21, released below <=0.052, empty <=0.027). Applied; check runs right above the release spot (no shift move), with the squeeze test; stuck -> put down again there. att102 ok (bowl 10 mm, bee 5 mm).
- att104 ok; bee by blob metric lands ~2 cm low in x with the 1.6 cm aim-short offset: offset removed.
- att105: false alarm 0.109 from a released bee standing tall below the gripper (image), squeeze empty; unstick routine then went down over the bee and moved it 6 cm. Now: yellow high + squeeze empty -> rise 10 cm and re-check (stuck keeps >= 70 % of the fraction).
- att110/111: bee grasped in the bowl, lost in transit (fell back into the bowl), released nothing at the target, fallback found it in the bowl again (~3 min). Now: in-bowl grasp 6 mm lower; gripper reading checked after the carry (empty -> re-pick instead of a fake release).
- att113/114 FAILED: confirm-up misjudged a stuck bee (0.221 -> 0.031 when raised) -> bee dropped next to the bowl; grasping it there caught the bowl wall (readings 70.6/67.1) and dragged the bowl (att114 bowl 8.6 cm off). Fix: no confirm-up, STUCK_FRAC 0.15; near-bowl bee grasp offset 12 mm away from the bowl; reading < 74 near the bowl -> let go before lifting and stop for a person.
- att115 real: repair test SUCCESS (layout ok): bowl 1.3 mm, bee 23 mm; clean checks 0.001/0.012.
- att119/120 FAILED: bee stuck on the right fingertip after release, mostly above the bottom strip (0.126 < 0.15); later dropped and lost (side shows no bee, home check clean). Added whole-image yellow >= 0.06 as a second stuck criterion (stuck 0.08, released <= 0.036 over att 102-118).

- attempts 125-126: the bee is still missing. The side image shows only the bowl on the table, and the wrist view shows the bowl empty (the "thin" region is a shadow on the wall). The gripper is clean (squeeze 99.1). The program stops correctly; nothing to fix in the code. A person must put the bee back.

- operator note (2026-11-05): the bee from att 119 stayed on the OUTSIDE of a fingertip (bottom strip 0.123, squeeze 99.1) and fell off the curtain-side edge. The library now refuses releases outside x 0.36-0.60, y -0.22..0.08. Changes: temp_spot is now x 0.40-0.56, y -0.18..0.05; place_bee clips release spots to x 0.38-0.58, y -0.20..0.06; ensure_released confirms every release with the side camera. Sequence: squeeze; rise +10 cm; step 8 cm toward the base; require bee-height points (h 1-6.5 cm) within 5 cm of the spot (n>=150, centre<=4 cm, top>=3 cm). If they are missing, go back low, wiggle, slide and check again. The arm rises high only after confirmation.

- att 128: passed (bee 3.5 cm off). The first put landed +3.8 cm in x; the correction aimed at the full counter-offset (3 cm), but the re-put landed 0.4 cm from the aim, so it overshot. The bee correction now aims off by half the error, at most 2 cm. Side release checks: ok, n 412-1126.

- att 134: passed. The bee was put straight from the bowl onto its target, but the side read 2.3 cm off (> SKIP 1.8), so it was re-picked for nothing (bowl 1.4 cm, bee 1.65 cm at the end). The bee skip threshold is now max(SKIP, 0.6 TOL) = 2.4 cm.

- att 141/142: passed, but each re-picked a bee already on its target. Before the bee step the point median read 2.8/2.6 cm off; the blob centre (the orchestrator metric) read 0.9/2.1 cm. The bee skip now uses xy_blob.

- att 143: failed. The bee was put at temp spot (0.40,-0.07) (side check ok), but at home the side camera missed it: it was merged with the bowl blob (extent 0.18, under the 0.19 split trigger). The wrist search then looked at the OLD bee spot because bee_c was not updated after the move. Fix: bee_c = dest after each bee move (the split trigger stays 0.19: a lone bowl reads up to 0.185). att 144 passed from the leftover state.

- att 149: passed. The bee was re-picked with its blob 2.45 cm off, but the wrist saw it 0.6 cm from the spot. move_bee_from_table now skips the move when the wrist refine is within 1.5 cm of dest.

- att 151/152: 4 table grasps missed (readings 96.9-100) on a bee next to the bowl. The wrist region was too big (0.144 x 0.05, 0.12 x 0.081, 0.102 x 0.094): a flat shadow or the bowl joined it, which shifted the centre and axis. Every good grasp read 0.10-0.12 x 0.05-0.07. Fixes: an oversized region is cut to pixels over the side camera's raised bee points (+-1 cm); the region is accepted only if len < 0.13 and wid < 0.085, otherwise the side estimate is used.

- att 153: the trimmed region looked fine (0.113 x 0.063, axis agreed with the side) but still missed and pushed the bowl. Cause: the near-bowl 12 mm grip shift ran ALONG the jaw (the bee pointed at the bowl), so with a 70 mm opening a finger landed on the 6 cm bee and the bowl-side finger on the rim. Fix: no shift; the opening (85..70 mm) is chosen so both finger centres stay >= 0.088 from the bowl centre. If none fits, BeeBlocked: move the bowl to bowl_spot (away from the bee and its destination), grip the bee, and the bowl step goes to the target later.

- att 154: the bee grasp held (reading 73.7) but the wall rule (< 74) let go and stopped. The jaw segment was 9.9 cm from the bowl centre, so the wall could not be in the jaws. The wall check now applies only when the segment between the open fingers comes within 8.5 cm of the bowl centre.

- att 156: passed but took 586 s. The bee parked at the temp spot (0.40,0.05) with its jaw axis pointing at the bowl target 14 cm away. After the bowl moved there, the bee grasp was blocked, so the bowl was cleared and then moved back (3 bowl moves). Temporary placements now use jaw yaw = direction to the bowl target + 90, so the later pick is tangential to the bowl.

- att 164/175: the bee was re-put with the wrist reading 1.9/1.7 cm off and ended 1.9/1.95 cm off anyway (~100 s each). The wrist skip is raised to 2.0 cm; the final measure and correction (2.6 cm) still guard.

- att 181: the in-bowl grip read 74.1 and the bee fell back into the bowl on the lift. The release check found no bee, so 3 unstick tries ran and the program aborted, although the gripper was clean (yellow 0.01-0.02, squeeze 99.1). Now when every check says the gripper is empty, ensure_released raises BeeMissing, place_bee returns False, and the caller retries (unpack looks into the bowl again).
- att 182: the bee read 3.7 cm tall near x 0.35, so the tool went to 3.0 cm. Closed, the fingertips are below the table, and the model refused the lift (the run ended holding the bee). The grasp-height minimum is now 4.2 cm.

- att 183: the bee was put down (held-bee start), but it had been dropped from table+0.09 and lay on its side: 5.5 cm tall, over the orchestrator's 0.055 bee prior, so 'bee: not found'. A re-pick keeps that pose. Fixes: a held bee in the jaws is released at +0.065; after the final measure, a bee taller than 0.052 that the layout check fails in most frames is picked and released from 5 cm higher so it tumbles flat (2 tries).

- att 184: the orchestrator said layout ok, but my measure lost the bee. Lying on its side it read h 0.056, n ~1100, ext 0.13, which matched my bowl rule (h >= 0.055), so it went into the bowls list. The bowl rule now needs h >= 0.062 (bowls read 0.066-0.082). The flip did not trigger (votes need the bee, and it was absent).

- att 185: the 5 cm re-drop at the target left the bee standing more upright (0.062 tall, now outside the bee class, so measure said None). New flip: up to 3 tries. Pick the tilted bee (from the bee class or the new TILTED list, h 0.052-0.075), drop it from +0.08 at a mid-table temp spot, re-perceive, and if flat (<= 0.052) move it to the target.

- att 186: the tilted bee read 0.059 (> 0.058 bee class), so the start found 'no bee' and grasped the empty bowl twice. Now a TILTED blob at the start is taken as the bee.

- att 188: the orchestrator said layout ok (bee 1.1 cm, bowl 2.0 cm), but my claim failed on my votes (1/3: merged bowl+bee blobs, extent 0.30). The claim now passes a flat bee (<= 0.053) within tolerance regardless of the votes. The run took 838 s: the 8 cm flip drops bounced the bee more than 5 cm, which cost unstick loops, then BeeMissing and re-perceive. It worked out in the end.

- att 190: claimed success, but the orchestrator failed it (bowl 4.1 cm off). Bowl and bee targets 17 cm apart: the side blobs merge (extent 0.28), so its check misreads; att 191, same targets, passed. The bee is now aimed 1.5 cm further from the bowl when the targets are under 20 cm apart (bee_aim).

- att 209: the start gripper read 98.7 with the bee on the OUTSIDE of a finger (whole-image yellow 0.072). The held-bee branch called place_bee, whose 'jaws empty after carry' check (>= 97.5) declared the bee lost, opened, and travelled home. The bee fell off and is now missing (att 210: not on the table, in the bowl or on the gripper). Fix: place_bee(check_empty=False) for the held-bee and stuck-bee paths, so they always lower, release, and run the side-camera release check.

- operator 2026-11-06: the bee is wrapped in ORANGE. The colour range is now H 0-40, S > 120, V > 110 (bee_mask), STUCK_FRAC 0.10. Offline on saved wrist images: att 209 hanging bee 0.267 bottom / 0.158 whole; clean starts <= 0.002. A bee in or on the gripper at start (closed reading < 99.0, colour, or squeeze < 99.0) is no longer carried. release_at_start lowers straight down from home over (0.58, 0.00), or the nearest free zone point >= 0.10 m from all blobs, to 0.10 m above the table, opens, rises and re-checks (up to 3 tries with a jaw wiggle), then goes home and continues as usual.

- att 218/219: the start drop worked (gripper clean afterwards), but the bee landed standing on its end: blob (0.63, -0.01), h 0.086, ext 0.056 x 0.088, n 1418. That was outside the TILTED class (<= 0.075), so the program assumed the bee was in the bowl and stopped. TILTED now covers h 0.052-0.10 with ext <= 0.11 (the bowl is 0.11-0.20). The tilted bee is then moved and flipped (dropped from 8 cm at a free spot) as before.

- att 220 hung 2 h 45 min at full CPU after 'bowl centre median of 7', with no log line and no motion; it was killed (137). Offline harness (real Planner and Kin, sim.check, fake session; no calibration on this machine, so approximate wrist K/T): move_bowl on the same scene plans every motion in < 0.1 s, and every loop in reset.py is a bounded `for`. The one unbounded call on that path is the controller FK RPC (oracle, called by every IK), which runs under the robot client's lock. Changes:
  1. Each oracle call runs in a worker thread with a 3 s timeout; after a timeout the oracle is switched off and the planner uses its fitted model. Tested offline with a never-returning oracle: IK finished after 3.0 s, the next call took 0.0 s.
  2. RUN_LIMIT_S = 17 min via SIGALRM -> RunTimeLimit(Abort). It re-arms every 5 s until it reaches main, because library handlers can swallow one. Every `except Exception` / `except Abort` in reset.py re-raises it, and main cancels the alarm before the return home. Tested offline: a loop of reach_ok calls stopped after 2.0 s.
  3. faulthandler dumps all thread stacks to <attempt>/stacks.txt every 5 min, so a future hang shows its location.

- att 229/230: both claimed success, but the orchestrator found neither object. Targets were 18 cm apart and the bee ended about 3 cm from the rim, so the side blobs merged in every frame (votes 0/n, both bad). The bee also lay with its long axis pointing at the bowl. Fixes: near_bowl_yaw places the bee with its long axis across the bowl direction when the targets are < 22 cm apart; merged_layout (all votes fail both while our errors are within tolerance) triggers a bee re-put 2.5 cm further from the measured bowl, and the claim is fail if the layout is still merged.

- att 253/254: the bee was parked at temp spot (0.40, -0.10), right behind the bowl (0.53, -0.16) as seen from the side camera, which looks along -x. It was never seen again: the wrist search found it at (0.428, -0.135) but rejected it at 4.5 cm from the looked-at spot. Fixes: temp_spot penalizes spots hidden behind a bowl position (x < bowl.x - 0.03 and |dy| < 0.09); search_bee accepts regions up to 8 cm from the look point and also looks behind the bowl (3 offsets), at most 5 looks.

- att 259: the released bee bounced about 5 cm, to (0.40, -0.205) near the e-stop edge. The side check saw a centre 4.6 cm off with 97-128 points, so 3 unstick tries ran and the program aborted 'stuck' (the bee below put orange into the wrist image, so 'clean' was false). att 260 found it there and passed. Fixes: the side check uses points within 7 cm, centre <= 6 cm, n >= 80, top >= 0.025; and when all squeezes were empty, a wrist look straight down at the release area (bee up to 10 cm off) counts as released before giving up.

- att 280: exit 139 (segfault) 2 s after faulthandler's 5-min periodic stack dump (the 'Timeout (0:05:00)!' header in stacks.txt). Dumping other threads while the kortex threads run is unsafe; removed. The watchdog then failed to take the arm home ('Session not in control'). att 281 started with the arm over the table, merged with the bowl in the side view: 'bowl not found'. go_standard_home now moves to the usual home first when the start joints are more than 5 deg away (straight up first if low), and makes that pose the run's home.

- att 285: the start squeeze read 95.2 (colour clean 0.007), so the start drop ran. Lowering in the home orientation drove joint 5 to 129 deg (Gen3 limit 120) and the arm went into FAULT; att 286 could not start. Fixes: within_limits() (j2 <= 124, j3 <= 143, j5 <= 116 deg) is applied to every IK solution; the start drop first turns tool-down in place at 0.30 m over the spot, then lowers vertically; after a drop, colour clean with squeeze >= 94 counts as empty (squeeze drift).

## att 293/294 (after the arm fault was cleared)
- Start gripper read 61.4 (half closed, empty; bee on the table, no colour) -> treated as bee on the gripper -> free_drop_spot
  found no spot with 0.10 m clearance (bowl 0.49,0.01 + bee 0.52,-0.15). Fix: at start, if only the reading says "on", close
  fully and judge by that reading; drop-spot search over the whole zone with clearance 0.10 -> 0.08 -> 0.06.
- att 296 (43c73af): success, 290 s. Start reading 0.4 (open) -> the squeeze test passed, no drop. Side check at bee target
  (0.471,-0.198) n=0 three times with a clean squeeze (99.1); the wrist look after 3 unsticks found it -> released. Probably
  side-camera blind spot near y -0.20 with the arm 8 cm away; kept the operator-mandated unstick (no change). ~60 s lost.
- att 306: fail. Start "bee" (0.322,-0.062) h 0.036 n 302 was a phantom (same place as att 30/166); the real bee was in the
  bowl and was carried along; then the bee was lost. Fix: bee class h >= 0.038 (real bees at start 0.039-0.049 over 85 starts).
  att 307 then recovered (bee in bowl) -> success.
- att 322: fail in 16 s: fixed clutter at (0.717,-0.267) read h 0.038 n 330 ext 0.089x0.028 -> taken for the bee -> "out of
  reach". Fix: bee class rejects h < 0.040 with n < 450 (all phantoms; real low bees n >= 905). att 323 then succeeded.
- att 320/328 (500+ s): bee parked at a temp spot only 0.11-0.12 m from the bowl target, then blocked once the bowl was
  placed -> bowl moved away and back. Checked: no temp spot with more clearance exists inside the zone/reach (r <= 0.58)
  for those layouts. Option for later: carry the bowl with the bee inside to its target, then unpack (att 306 shows the bee
  stays in) -- not done untested.
- att 333/334: bowl (0.53,0.00) + bee (0.435,-0.17), targets 19 cm apart: side blobs merged -> orchestrator "bee: not
  found". unpack put the bee straight to its target with PLACE_YAW (long axis toward the bowl) and without bee_aim; votes
  1/3 did not count as merged so the correction skipped ("already within 1.6 cm"). Fix: unpack to target uses bee_aim +
  near_bowl_yaw; merged = both objects fail in most vote frames (was: all frames).
- att 335 (test c03efa9): success, not merged. But the bee blob read 0.16 long in y (ext limit 0.15) -> bee "not found" in
  most frames, and the correction got bee=None from the last frame and did nothing. Fix: bee ext <= 0.17; measure keeps
  the last frame that saw each object.
- att 336: our votes 3/3 but the orchestrator "bee: not found": bee blob 0.155-0.165 long (prior extent <= 0.15) lying
  along y at (0.43,-0.17). Fix: measure tracks the max bee extent; >= 0.145 -> correction re-puts the bee turned 90 deg
  (place_yaw "rotate" = old long axis as jaw).
- att 337: "rotate" turned the bee's long axis toward the bowl (19 cm) -> merged blobs, orchestrator found neither.
  Fix: correction yaw = near_bowl_yaw (long axis across the bowl direction) whenever targets < 22 cm apart; "rotate" only
  otherwise; correction aims from bee_aim (1.5 cm nudge away from the bowl).
- att 338 (d273d00): bee placed with near_bowl_yaw -> separate blobs, votes 3/3, orchestrator ok.
- att 346: bowl at target x 0.409 (rim 0.400) read by blob 0.365/0.365/0.408 -> correction carried it +4.3 cm -> blob 0.446,
  3.8 cm off the other way (orchestrator ok, barely). Fix: bowl correction carries 0.6 x blob error, at most 2.5 cm.
- att 347/348: dim light; bee in the bowl segmented as a 0.088 x 0.032 strip near the centre -> rejected "thin", stop.
  Fix: accept thin (wid >= 0.025) regions 0.07-0.13 long within 3 cm of the bowl centre.
- att 350: bowl target x 0.424: blob read 0.428/0.388/0.389 (merged with low clutter near the base) -> 3.7 cm, orchestrator
  ok. Fix: bowl_aim -- aim +1.5 cm in x for bowl targets with x < 0.43 (main move and correction).
- att 352: bowl_aim nudge made it worse (blob 0.440 for target 0.407, aim 0.422) -> BOWL_NUDGE 0.0.
- att 354/355: bowl tipped on the bee (blob h 0.105 > bowl max 0.095) -> "bowl not found". Fix: fix_tilted_bowl -- pinch
  the rim at the highest side (top-band points), lift 12 cm, carry to the bowl target (if >= 13 cm away, else 12 cm off
  the raised side), lower to level-rim height, release; then the normal reset.
- att 356 (e894d7e): tipped bowl lifted by its raised rim (95.2 held), set level at its target with the bee inside; normal unpack -> success, orchestrator ok.
- att 358: orchestrator 'bowl 4.5 cm off' though our votes 3/3 (median 0.4 cm, one frame 3.2 cm): bowl blob near (0.46,-0.16) joins intermittent low clutter at (0.32-0.34,-0.26..-0.28) in single frames. Not fixable from here; att 359 then passed.
- att 364: orchestrator 'bowl: not found' with our votes 3/3 (bowl 2.1 cm): bowl at ~(0.43-0.45,-0.16) near the intermittent low clutter (0.30-0.36,-0.10..-0.26); same family as att 358. Retry att 365 passed. No reliable fix found (clutter location varies).
- att 366/367: bowl target (0.408,0.047) rim touching flat clutter at (0.33-0.34,-0.04) (the old 'phantom bee' spot): merged blob read 4.3 cm off in 1/3 of frames, orchestrator failed. Fix: CLUTTER memory of low flat blobs (1-3.5 cm) per run; bowl_aim shifts the aim up to 1.5 cm away so the rim keeps a 3 cm gap.
- att 374 (522 s): third blocked-temp-bee case (temp 12.6 cm from the bowl target). New: unpack carries the bowl WITH the bee inside to its target first when the bee target is blocked by the current bowl (bee seen in the bowl by the wrist; rim grasp on the side away from it; jaw tips stay ~1 cm above the rim, the bee top is ~2 cm below), then takes the bee out straight to its target. Untested on the robot at commit time.
- att 378 (ea6f89e): carry-bowl-with-bee path worked (bee seen in bowl at carry, bowl carried, bee unpacked straight to target): 294 s instead of ~500 s. Bowl near base (target 0.406,0.042) again read 0.37-0.414 (3.4 cm median); orchestrator ok.
- att 380: in-bowl grip read 89.0 and carried the BOWL (wall in jaws) 13 cm to the bee spot; recovered via search + carry-with-bee. att 360 (90.4) lost the bee. Fix: in-bowl picks with reading > 88 count as a miss (let go at grasp height, next try re-looks 4 mm lower).
- att 405/406: start pose had open fingertips at the keep-in limit y<=0.30 -> model check refused go_standard_home (keep_in_violations 97). Fix: close the empty gripper before go_standard_home (fingers ~1 cm narrower).
- att 407 (b4cf238): gripper closed, go_standard_home passed the model check from the same pose; full reset ok, orchestrator ok.
- att 412/413: bee standing/leaning in the bowl, wrist region only 0.04-0.05 of the interior (<0.08) -> 'bee not found', stop. Fix: second detection with min_frac 0.025 on a plane near the rim; grip the top (tool z rim-0.5 cm, jaws tangential, <= 4 cm off centre); release 3.5 cm higher.
- att 414 (47ff0f6): standing bee in the bowl found (end-on 0.048), gripped at its top, placed at a free spot; full reset ok, orchestrator ok.
- att 416: after carry-with-bee the bowl rim (wrist) was on target but the blob median read 2.6 cm short; the bowl step moved it by the blob error -> blob 3.0 cm off the other way. Fix: move_bowl uses the mean of blob and wrist-rim displacement when they differ by > 2 cm.
- att 419: carry-with-bee pinch read 87.3 (bee in the jaws with the wall); bee fell out at the target onto the table, but unpack still looked only in the bowl -> 'bee not found'. Fix: move_bowl(rim_min=92) lets go of an unclean pinch and returns False; after the carry, a bee seen on the table ends unpack (placing steps take it).
- att 425: bee bounced 11 cm to (0.52,0.16), out of the side-check radius and wrist view; squeeze empty x3 but colour not 'clean' -> 'stays stuck' fail. Fix: before the stuck verdict (all squeezes empty), a side perception from the current pose; a bee-sized blob anywhere on the table = released.
- att 434: bowl skipped (blob 1.5 cm from clutter-shifted aim, 2.7 cm from target; one final frame read ~4 cm). Bowl skip now also needs <= 2.2 cm from the real target.
- operator note (bee held after learn ep 345 / att 440): att 440 died at start on KServerException (no motion); att 441 started with gripper 70.6, yellow 0.178 -> start_drop at [0.58, 0.01], empty after (colour 0.000, squeeze 99.6), then success. Existing start-drop path handles it; no change.
- att 440/467: KServerException INVALID_USER_SESSION_ACCESS in kortex.Robot() at session start (library retries 2 s). Added _RetryRobot (monkeypatch rlpj.robot.kortex.Robot): close half-made conns, reconnect up to 4 tries (2/4/6 s waits). Tested offline with stubbed kortex_api.
- operator notes ep 388/394/411 (bee held after learn runs): next resets att 475 (73.2), 479 (74.6), 492 (69.3) all did start_drop first and succeeded; att 487 (86.0, colour 0.078) also dropped. No change.
- operator notes ep 434/442/443: att 508 (76.3) and 514 (77.6) did start drop, gripper empty after (colour 0.000, squeeze 99.6), success. ep 443 -> next reset uses the same path. No change.
- att 521/522: bee hung from one fingertip (tape snag) with jaws open; 3 drops at 0.10 m with wiggles failed. Start drop now: try 0 as before, tries 1-3 go low (tool tz+0.06, fingertips ~3.3 cm), wiggle, slide 4 cm along y away from the nearest table object, back, rise, check.
- att 523: low slide (tz+0.06, 4 cm along y) left the bee on the finger (dashboard: bee rests on the table at low, stays on the left finger). Now 4 low tries at tool tz+0.05 with wiggle + slide 6 cm across the jaws / along the jaw axis / 45 deg twist + across, each kept in zone with >= 5 cm clearance.
- att 524: stuck bee freed on low try 1 (tool tz+0.05, slide 6 cm across the jaws, jaw yaw 0 -> +y to [0.58, 0.06]); full reset success (bowl 1.1 cm, bee 0.3 cm), 418 s.
- operator notes ep 461/462/467/470: att 533/534/538/540 started with the bee held (78.1/78.5/69.3/74.6), start drop, success. att 526: one-off tracking stop on home_before_bee (joint 4 held at 0.0 while the plan went to -1.14 deg, free air); 527 succeeded. No change.
- operator 19:17: library waits for 'watchdog up' and retries the first connection ~10 s; my _RetryRobot cut to one extra reconnect (3 s wait). 'watchdog did not start' = connection problem, not the program. ep 493 held bee -> next reset starts with the start drop (handled).
- att 604: fixed clutter at (0.72-0.74,-0.27) (r 0.77) read bee-high -> abort 'bee out of reach'. perceive now drops bee/tilted candidates with r > 0.70 (into others). Offline test with synthetic blobs: phantom ignored, real bee kept. Held-bee starts after ep 540/546 (att 594/600) handled by the start drop.
- att 613: bowl skip judged by blob median (1.0 cm) while rim median said 3.4 cm; final 3.0 cm. Skip now judged by blob/rim mean when they disagree > 2 cm (att 613 -> mean 2.1 cm -> move).
- att 641: a single-frame low blob (n 84 at 0.413,-0.061) shifted the bowl aim 1.5 cm -> final 2.45 cm. bowl_aim now uses only clutter seen in >= 2 frames (centroids within 3 cm). Offline test: single sighting -> no shift, two -> shift.
- att 650: carry-with-bee rim pinch read 90.8 (<92) -> let go at (0.358,0.025), refused by the library (outside release zone; 60-92 counts as holding). move_bowl with rim_min now only takes rim grasps inside the zone (8 mm margin); none -> return False (bee_temp path). Offline geometry for att 650: 0 deg excluded, +30 (0.369,-0.009) taken. Att 651 recovered (start reading 90.8, closed, success 400 s).
- att 653: released bee not seen by the side camera (n 1-4 at the spot), jaws empty x3 (99.6); unstick slides pushed it to (0.562,-0.094); post-loop wrist look found nothing -> 'stuck' abort. 654 found it and succeeded (118 s). ensure_released now does the wrist look-down on the first failed side check when the squeeze was empty, before any unstick move.
- att 677 (also 641/664/672): bowl_aim clutter shift of 1.5 cm left bowls 2.2-2.9 cm off. Gap target cut from 3 cm to 2 cm (shift = min(1.5 cm, 2 cm - gap)); merges seen only at 0-1 cm gaps.
- att 679/680: bee missing from the scene. Orchestrator saw it hanging on the gripper after learn ep 624; at reset start the gripper was clean (colour 0.001, squeeze 99.1). Side camera: table and bowl empty; wrist look into the bowl: empty; wrist looks behind the bowl (3 spots): nothing. Orchestrator matched the fixed clutter at (0.69,-0.28) as the bee (18-20 cm off). The bee most likely fell off the table (curtain-side gap or floor) after the learn run: not recoverable by the program.
- att 681 (test run): bee still absent (scene_before: table and bowl empty). The 'standing bee' (0.053 x 0.020, area 0.073, 4.7 cm off centre) is a rim highlight; the one real standing bee (att 414) was 0.043 x 0.016, area 0.048, 4.9 cm off: too close to separate by geometry, left as is (two empty squeezes then stop).
- att 679-691 (no operator): added sweep_bee: when the side camera, bowl look and spot search all miss, look straight down with the wrist camera over a 10 cm grid of the reachable table (21 looks, tool z 0.30, max_off 9 cm).
- att 692: sweep found the bee at (0.55, 0.15) (curtain side, invisible to the side camera, there since ep 624). Blocked-bee path re-perceived with the side camera, saw no bee and stopped. Now keeps the wrist position before clearing the bowl and re-searches by wrist after.
- att 693: false standing bee in the bowl (rim highlight) -> two empty grasps -> abort before the sweep. Now 'could not grasp the bee in the bowl' also falls through to search_bee + sweep_bee.
- att 694: bee found by sweep at (0.569,0.168), bowl cleared to (0.389,0.04), then a wrist re-search at the spot missed it -> stop. Now keeps the previous bee position after clearing (grasp refines it).
- att 692-695: the 'bee' the wrist sweep found sat 9-12 cm from the bowl centre on the curtain side and moved with the bowl each run (0.553,0.148 / 0.569,0.168 / 0.439,0.138 for bowls at 0.485,0.046 / 0.517,0.06 / 0.408,0.046); the grasp in 695 closed empty: it is the bowl's shadow. Wrist search/sweep now skip regions < 13.5 cm from the bowl. The bee is truly absent from the table.
- att 699/700: the bee is back (side image: bee in front of the bowl, blob at (0.71,-0.008), r 0.71) but the att-604 rule r > 0.70 = not a bee dropped it. Exclusion now only the fixed clutter corner x > 0.66, y < -0.23.
- att 701: bee seen at (0.718,-0.009); bowl placed; bee approach refused (max_radius 0.806 > 0.80). move_bee_from_table now grips a bee beyond r 0.69 at its near end (shift <= 3 cm toward the base), jaws tangential.
- att 702: success (133 s): far bee at r 0.718 gripped at its near end (0.69,-0.009), reading 77.6, placed 1.2 cm off; bowl 0.5 cm; orchestrator layout ok.

## att 704/705 repair
- 704: carry-with-bee rim pinch read 94.7 (looked clean) with the bee in the jaws; the rim release point g1 (0.421, 0.087) was outside the zone -> the new colour safeguard refused to open, the arm went home holding the bee. Fix: move_bowl requires g0 AND g1 in the zone (8 mm margin) for rim_min carries, and strongly prefers it for every carry.
- 705: the bee fell into the bowl at the first start-drop open, but the tool-down check kept saying stuck: the whole-image rule (>= 0.06) counted the bee lying in the bowl below (0.065-0.084, bottom strip 0.000). Fix: the whole-image rule fires only if the last 4 % of image rows also show colour (>= 0.05; stuck/held 0.11-0.37, on the table below <= 0.016).

## att 716/717 repair
- 716: after carrying the bowl with the bee to its target, pick_bee opened the jaws at the hover 0.236 m above the bee in the bowl -> refused ("object on the fingers", colour 0.11 at the image bottom = the bee BELOW, jaws empty at 31.6). Fix: open_jaws() opens before the hover move (camera not looking down at the bee), sends nothing when the jaws are already that open, and if refused opens again after the hover / at the approach height (low, < 0.20 m). Same for the bowl rim grasp.
- 717: watchdog did not start = connection problem (operator), not the program.

## att 724/725 repair
- Targets bowl (0.501,-0.005), bee (0.396,-0.156): 18.3 cm apart, bee at -120 deg from the bowl. Side blobs merged; the orchestrator found neither. 724 votes 2/3 -> no correction; 725 correction (2.5 cm push) landed 18.6 cm away, votes 1/3.
- History (all attempts): merges cluster at -110..-130 deg with d 0.15-0.199.
- Fix: bee_aim push 1.5 -> 2.5 cm when d < 0.21 (was < 0.20); MERGE_PUSH 2.5 -> 3.5 cm; merged_layout fires on any frame failing both objects.
