
SLOT MOVED AGAIN, FIXTURES SHIFTED (21:00): the cabinet crept ~3.5 cm along its front under the pushes. Evidence: the
reset's wrist image (attempt 140) shows the fingers 3.6 cm beside the knob, on the cabinet's side edge; the drawer face
in the slot is centred 2.9 cm further along t; the cabinet blob moved 4.0 cm. The task yaml now has slot_t
[-0.201, -0.126] (centre -0.1635, was -0.1285) and all keep-out boxes moved by (+13, -32.5) mm. Your slot-relative
constants follow the yaml automatically; re-read anything you cached (e.g. a saved knob position from before 21:00
is now 35 mm off: discard it).

CORRECTION (21:15): IGNORE THE 21:00 "SLOT MOVED AGAIN" NOTE. That -35 mm shift went the wrong way and is reverted: the
task yaml is back to slot_t [-0.166, -0.091] (centre -0.1285). Evidence from your own wrist images: in attempt 140
(these numbers) the knob was at the LEFT edge of the wrist view, ~3.6 cm (280 px at 7687 px/m) from the gap between
the pads; with the shift (attempt 141) it went almost out of view at the bottom left. So the knob is ~3.6 cm on the
image-left side of where the yaml puts it.

SLOT SHIFTED +36 mm ALONG t (21:30), from the reset's own wrist evidence: attempt 140 knob 3.6 cm image-left of the
finger gap; attempt 141 (after a wrong -35 mm shift) further away; attempt 142 the judge said "left" (= +t) and the
+12 mm correction was refused by the OLD keep-out. task yaml now: slot_t [-0.130, -0.055] (centre -0.0925), keep-out
boxes moved by (-13.4, +33.4) mm. Re-read the yaml; discard saved knob poses from before 21:30.

SLOT SHIFTED ANOTHER +43 mm (21:45), from the reset's own wrist measurement in attempt 143 ("tab +43.3 mm along t from
the aim", knob left of the left finger in the image, keep-out refused the measured position). task yaml now: slot_t
[-0.087, -0.012], centre -0.0495 (knob ~ -0.055). Re-read; discard saved poses from before 21:45.

CORRECTION 2 (21:58): IGNORE THE 21:45 "+43 mm" NOTE; it is reverted. The task yaml is back to slot_t [-0.130, -0.055]
(centre -0.0925), which matches the knob position the reset measured directly at 21:29 (t ~ -0.098).

FINAL ON THE SLOT (22:12): the reset measured the knob at t ~ -0.055 in four runs (attempts 140, 143, 144, 145; the
earlier -0.098 came from a dry run that wrote the tab file). The task yaml is RESTORED to the 21:45 geometry: slot_t
[-0.087, -0.012], centre -0.0495, keep-out boxes moved accordingly. This replaces both corrections; no more slot
changes unless your own wrist measurements say otherwise.
Operator note 22:35 (from rl_curve.jsonl): 25 training episodes, 0 successes, return mean -2.7 with no upward trend.
1. Episodes 25, 27 and 28 reached full insertion and lost it on the retreat. Make the retreat unable to drag the drawer:
   open the gripper fully and wait for it to open, move straight up (or back along the jaw axis) before any move along -N,
   and if the force rises during the retreat, stop and re-open. Then re-check that the front is flush before you claim success.
2. Use a dense, monotone shaping term for insertion depth and orientation (straight, pull facing out), plus a large
   success bonus; log depth, yaw error and success per episode next to the return, so progress is visible before the
   first success.
3. Run episodes as fast as possible now: resets have blocked training since 18:09.

Operator note 23:18 (episode 29, attempt 88): both grasps closed on nothing (99.1) after refine_wall shifted the grasp
+0.030 and +0.037 m. In wrist_front_pre.jpg / stop_overlay_wrist.jpg the drawer is at the lower left of the view and
the right finger is out on the bare table: the refined grasp went the wrong way or too far. (The reset agent's
refine_wall showed the same failure today: 4 misses after 2 cm shifts.) Cap |shift| at ~0.012, check its sign against
the image, and after one refined miss retry at the unrefined perception point. The reset now accepts the drawer within
4.5 cm and about +-15 deg of the target, so the grasp must handle that spread.
