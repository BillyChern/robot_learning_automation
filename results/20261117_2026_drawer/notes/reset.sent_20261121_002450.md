Operator note 00:25 (attempt 162 start): the arm started at the pinch pose left by attempt 160 (force stop 65.6 N on
approach_pin0, return refused), gripper 70.3 = pre-closed around the knob. The start logic did not open it (70.3 is not
> 100*PRECLOSE+2) and escape() only opens fully when the tool points down, so escape_up backed off and rose with the
pads around the knob: 61.9 N, just under the 65 N stop, pulling on the drawer and the cabinet. Fix: at start, whenever
the arm is low near the cabinet, open the gripper fully (0.0) before any escape move, whatever the tool direction, and
for a tilted pinch pose back off along the tool axis first (out of the knob), then rise.
Operator note 00:26: attempts 160 and 162 both hit the force stop on approach_pin0 (65.6 N, 67.1 N), same first try:
"drawer face median 0.013 m out", "high pose ... depth +0.000", aim t -0.0430. The pinch is planned at the flush
plane (depth clamped to 0) while the drawer face is 1.3 cm proud, so the pads drive into the face/knob. For a proud
drawer, place the pinch at the measured face (protrusion added, minus a few mm), approach the last 2 cm at slow speed
with a lower force stop (~25 N: on contact, back off and re-measure with the wrist camera), and start at the knob t
measured earlier (-0.055), not PIN_T + 0.012.
