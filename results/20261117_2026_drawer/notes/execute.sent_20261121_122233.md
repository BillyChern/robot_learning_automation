Operator note 04:50 (episode 39, attempt 104): good: the wrist footprint is in and the grasp held; and the new library
force stop worked: ins10 stopped at 66.5 N TOTAL (change only 25.4 N), which before 02:37 would have pushed on.
The jam came from the side again: "align: t_span 0.058 too small to trust, tool on the slot centre", "t err -0.073".
The side camera reads the drawer ~5-7 cm off in y (your own wrist footprint vs side camera in this episode: [0.355,
0.079] vs [0.361, 0.005]), so its in-hand t error is not usable. Use the wrist footprint instead: at grasp time you
know the footprint centre and the grasp point, so the drawer centre's offset along the front wall from the tool is
known (project grasp point - footprint centre onto the wall direction). Put the DRAWER centre, not the tool, on the
slot centre: tool t = slot centre + that offset. Keep the jam rule (stop at > 35 N with < 3 mm progress, back out, retry).
Operator note (geometry, approved by the user): the cabinet was pushed back during the night (jams up to 94 N, the
tipped-drawer recovery): two full side-camera outlines (05:06, 05:12) put it 14 mm further from the robot (along N) and
6-9 mm along the front (-t). The task yaml now follows it: front_offset +0.015 (0.5283), slot_t -0.009 ([-0.096,
-0.021], centre -0.0585, knob ~ -0.064), keep-out boxes moved by (+0.0173, -0.0028). Re-read the yaml; constants derived
from the slot follow it. Discard saved knob positions and slot-relative offsets learned before 05:20, and check the
knob with your wrist measurement on the next pinch: if it reads ~9 mm off again, the t shift was wrong (say so in your
report).
