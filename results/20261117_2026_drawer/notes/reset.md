
CORRECTION (21:15): IGNORE THE 21:00 "SLOT MOVED AGAIN" NOTE. That -35 mm shift went the wrong way and is reverted: the
task yaml is back to slot_t [-0.166, -0.091] (centre -0.1285). Evidence from your own wrist images: in attempt 140
(these numbers) the knob was at the LEFT edge of the wrist view, ~3.6 cm (280 px at 7687 px/m) from the gap between
the pads; with the shift (attempt 141) it went almost out of view at the bottom left. So the knob is ~3.6 cm on the
image-left side of where the yaml puts it.
For the reset: do not rely on the yaml PIN_T alone; steer by the wrist image (visual servoing). From the pre-pinch pose
take a wrist image, find the knob (grey stub under the label plate; your detector missed it, so a simple dark/grey blob
search or the vision judge on that image), move the tool in 1-2 cm steps toward it until it sits between the pads,
then pinch. Save the tool pose that held, for the next resets.
