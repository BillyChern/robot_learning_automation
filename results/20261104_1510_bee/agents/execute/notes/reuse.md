# Reuse decisions (execute, bee_in_bowl)
- battery_placement_kinova_b / battery_grasp_place: ADAPT. Kept the structure (open, hover, one slow descent to a
  fixed grasp height, single close, check reading, slow lift, carry high, release above the rim, slow rise).
  Changed: grasp height for a 43 mm foam bee on kinova_a (tool 52 mm above table), full opening (bee ~76 mm wide,
  no bowl near the bee), release 30 mm above the rim, jaws across the PCA short axis. Re-derived in programs/execute.py.
- battery_placement_kinova_b / place_verification: ADAPT. No wrist depth here; verify with side tabletop blobs
  after homing (bee gone from its start spot, no bee-sized blob outside the bowl, bowl unmoved and upright) plus
  gripper readings (held through lift and carry, opened on release).
- battery_placement_kinova_b / battery_bowl_perception: ADAPT (idea only): classify by height and footprint;
  uses rlpj.perception.tabletop blobs and rlpj.perception.wrist.circle_refine for the bowl rim instead.
- seed_2_kinova_a lesson: APPLY: closed fingertips ~38 mm below the tool on kinova_a (used for GRASP_Z).
- marker_placement_kinova_a: REJECT (bin task; robot I/O replaced by RobotSession); lesson "rise vertically
  before carrying" APPLIED.
- All Kortex/camera/executor/watchdog code: REJECT (RobotSession does it).

# Results
- attempt 3 (commit 81a5f7f): REAL success. Grip reading 82.0 (held through lift), release 0.4, wrist rim refine
  moved bowl centre 9 mm, bowl moved 2 mm after, bee gone from table; wrist_bowl_after.jpg shows the bee in the bowl.
- After: phantom side-depth blob (~75 pts) at (0.36,-0.09) with nothing on the table -> bee blobs need >= 250 points.
- attempt 21 (episode 16): bee lay against the bowl, side depth merged them (extent 0.24 m) -> refused. Now: split the
  bee off the bowl blob (points > R+18 mm from the rim centre, below 6 cm), choose a jaw yaw whose open fingers stay
  >= 93 mm from the bowl centre (else refuse, no motion). Offline synthetic test only.
- 2026-11-05 operator note: bee can stay wedged on the fingers after opening. Added a release check after the
  retreat: wrist bottom-15 % bee-colour share (> 0.10 stuck; operator's HSV) + one squeeze (< 97.5 holding).
  If stuck: relower, close/open, shift 2 cm across the jaws, rise slowly, recheck (3 tries); still stuck -> claim
  fail and stay over the bowl (home set to the current pose so the session does not carry it away).
  Offline: saved wrist images read 0.000-0.016 (empty). Not yet run on the robot.
- 2026-11-05 operator note 2 (bee lost on the outside of a fingertip; squeeze blind, strip colour can be 0.12):
  release confirmation loop: squeeze at <= 0.18 m above the table (library refuses opening a held bee above 0.20),
  side camera from 8 cm up / 8 cm toward the base (bee-sized blob outside the bowl -> on the table -> claim fail,
  leave it), wrist from LOOK_Z above the bowl: bee-colour share inside the projected rim (bottom 15 % excluded)
  > 0.08 = in the bowl (offline 61 runs: empty 0.000, bee 0.20-0.45). Neither -> on the gripper -> relower, shake,
  shift, recheck (3 tries); unconfirmed -> stay above the bowl, claim fail (never home with it).
  The bottom-strip colour is now info only (bee in the bowl read up to 0.275). Mock-session test: ok / stuck /
  on-table paths behave as above. Not yet run on the robot.
- 2026-11-06 operator note (DAgger, orange bee): RLPJ_DAGGER=1 -> open a gripper closed on nothing (>= 97.5),
  rise straight up (same orientation, slow) to PRE_Z if lower, perceive with the arm's points removed, level
  straight move to above the bee (joint move only if the line is refused before motion), no separate bowl look
  before the grasp (wrist rim refine at the carry pose instead), then the normal grasp/carry/release/confirmation.
  Bottom-strip colour now hue 0-40, S > 120, V > 110 (info only). Offline: real planner + model checks from the
  episode-108 pose [25.7, 47.3, 288.0, 178.1, 58.3, 46.9] (tool 0.138 m up, no lift needed) and from a 0.05 m-high
  start (lift 8.2 s, then joint fallback) both run to a success claim; normal-mode mock unchanged.
- 2026-11-07 operator note (keep-in y -0.45..+0.30): home <-> workspace now goes through a fixed READY pose (tool down,
  yaw 90, over x 0.50 y 0.00 at LOOK_Z): joint move home->READY (max gripper y 0.052), then straight tool lines
  (look_bowl, above_bee, look_bowl_after) with a joint-move fallback only when a line cannot be planned (IK jump);
  return: line to READY, wrist-family switch there if needed (DAgger), joint move READY->home.
  tools/offline_flow.py (real planner + sim.check incl. keep_in): 8 layouts normal mode gripper y -0.27..0.15, DAgger
  from the episode-108 pose -0.29..0.19, all success claims, no refusals.
- att. 204 (ep 252, 22:37, room dark): wrist disc median V 23-36 (normal 165-211); EMPTY bowl read 0.164 bee colour (> 0.08) so the
  in-bowl test could confirm nothing. Added DARK_V=100: require after the first bowl look (abort before grasping, arm goes home), and at
  confirmation a dark image ends the run as "not confirmed" (stop over the bowl, claim fail) instead of a colour verdict.
- No operator to restore the light (att. 205 stopped at the dark guard). Dark mode (wrist disc median V < 100): in-bowl confirmed by
  red dominance (R > G+30, R > B+30, R > 50) share > 0.04 inside the rim; dark images att. 204/205: empty 0.000/0.000, bee 0.127.
  Requires the empty bowl to read < 0.02 before the grasp and median V >= 12. Offline: tools/offline_flow.py dark -> 8/8 success,
  darkempty -> 8/8 "not confirmed" (stop over the bowl, claim fail).
- att. 209 (ep 257, dark): false fail. Side rim fit [0.51,0.046] and ring test [0.45,0.077] disagreed; the dark wrist rim circle
  (radius ratio 1.195, 122 inliers) moved the centre to [0.525,0.024], ~5 cm from the real bowl (~[0.485,0.056] by back-projection).
  The bee went in near the rim, but the in-bowl disc around the wrong centre read red 0.001-0.003 -> 3 shake tries -> claimed fail.
  Fixes: dark mode rejects wrist circles with radius ratio outside 0.95-1.15 (dark runs 204-208: 0.95-1.09); confirmation also
  reads the disc around the other bowl estimates (side ring test, side rim fit, side bowl blob now; within 8 cm) and accepts them
  only with no table bee and empty jaws (209 offline: ring-test centre red 0.071-0.075 > 0.04). offline_flow BOWL_XY_OFF env added.
- att. 221 (ep 269, dark): side rim fit [0.483,-0.005] vs ring test [0.423,0.011] (6 cm apart, 719 top points, bowl reading 0.081 tall);
  the dark wrist circle (ratio 1.183) was correctly rejected, the release went to the fit, ~4 cm from the bowl (~[0.444,0.004] by
  back-projection); the bee went in near the rim (red 0.087; ring-test centre 0.197). New: when fit and ring are > 3 cm apart use
  their midpoint (39 earlier runs judged by the wrist rim: midpoint 3-14 mm, fit 6-33 mm, ring 10-36 mm; 209/221 midpoint 7/9 mm).
- att. 234: bee split from the bowl blob -> ring test pulled 4.8 cm toward the bee, midpoint 2.4 cm off (wrist refine fixed 1.1 cm); midpoint rule now skipped when the bee was split (rim fit was 0.4 cm from the bowl after).
- att. 254 (ep 321, dark, V 16-17): rim fit [0.378,0.072] from 588 top points 6.7 cm off (wrist back-projection ~[0.444,0.063]); midpoint 3.7 cm off; bee still went in (confirmed via the ring-test centre, red 0.048) but verify said 'bowl moved 0.08 m' comparing after vs the bad fit; wrist images before/after show the bowl did not move. bowl_moved now = min distance over before-estimates (fit, ring, used centre) x after-estimates (fit, ring): 254 -> 0.030. Tried a dark bright-blob wrist bowl locator offline: 1-2 cm biased and failed on ~40% of dark images -> not used.
- att. 270 (ep 336): split+smeared bee read wid 0.090 from its top third -> refused (safe, before any motion). Tried a top-fifth retry near the top centre on synthetic smears: width falls under 0.08 but the axis is wrong by 50-100 deg even when elongated (ratio 1.6) -> fingers could land on the bee. Not adopted; the refusal stays.
- att. 272 (ep 338): bee ~10 cm from the bowl merged into one side blob (extent 0.25x0.29) and split_bee_from_bowl found no bee-sized rest -> safe refusal before motion; att. 270 had the same layout (split but smeared). Depth not available to me -> execute.py now saves the side blob points to attempts/<n>/side_blobs.npz each run for offline analysis.
- Operator 2026-11-08 (round 3): joint 4 must stay outside 90..270 deg for the whole episode (converter drops others); look-pose
  route/height/yaw must not change. Audit of recorded plans: normal attempts 1-142 (before the READY routing) had joint 4 ~180;
  97 normal attempts since 162 stay within |j4| <= 63 deg (the 63 was a session setup segment); DAgger runs that started from
  a flipped policy pose stay flipped. Added install_wrist_guard: wraps s._execute and aborts (before execution) any segment
  entering 90..270 when the start is in the normal configuration; switches itself off when it fires so the return home runs.
  Route/height/yaw untouched. Offline: normal/dark 8/8, j4 = 0 throughout; DAgger from the normal-wrist equivalent of the
  test pose [25.7,47.3,288,-1.9,-58.3,226.9] 8/8, j4 -2..0; unit test: flipped segment caught, return home not blocked.

## 2026-11-10 operator 15:43: library refuses opening with the bee on the fingers outside release_zone / above 0.20 m
- The check also fires on the colour at the bottom of the wrist image (task gripper_check, same HSV/bottom 15 % as
  yellow_bottom_frac), even before the program's first gripper command (start reading counts). Risky opens found:
  (a) "open fully" at the look pose (0.32 m) with start reading 31.6 (common): the bee/bowl in the bottom of the
  image would be refused -> now opened at home before anything moves (home reads <= 0.006);
  (b) DAgger open at the policy's pose (the 10:10 incident) -> try_open; on refusal keep it closed, rise, side scan,
  put_down_stuck: carry height, level to a free spot (drop_spot: nearest (0.47,-0.07), bowl rim + 5 cm, inside the
  bee layout, else the zone with 2 cm margin), normal down to 0.18, slow to 0.13 m, open, <= 2 shakes, up; DAgger then
  re-scans and continues; non-DAgger goes home and ends fail ("no demo this episode").
  (c) bowl release: require bowl centre inside the zone (1 cm tol.) and release height <= 0.19 m before the grasp;
  release/retreat/relower/shift xy clamped into the zone.
- Offline harness emulates _check_release (modes stuckhome, daggerstuck; start reading 31.6 in normal mode): all pass.
