
SLOT SHIFTED ANOTHER +43 mm (21:45), from the reset's own wrist measurement in attempt 143 ("tab +43.3 mm along t from
the aim", knob left of the left finger in the image, keep-out refused the measured position). task yaml now: slot_t
[-0.087, -0.012], centre -0.0495 (knob ~ -0.055). Re-read; discard saved poses from before 21:45.

CORRECTION 2 (21:58): IGNORE THE 21:45 "+43 mm" NOTE; it is reverted. The task yaml is back to slot_t [-0.130, -0.055]
(centre -0.0925), which matches the knob position the reset measured directly at 21:29 (t ~ -0.098).
Why I stopped moving the yaml: your wrist images of attempts 143 (slot centre -0.0925) and 144 (-0.0495) are nearly
IDENTICAL: knob ~2-3.4 cm left of the LEFT finger both times, and your measurement says "+30..+37 mm along t from the
aim" both times. So your pinch aim did not move although SLOT_C moved 43 mm: something pins it (the keep-out refusal
falling back to "the previous aim"? a cached pose?). Please log the aim t you actually command, and close the loop on
the image: from the pre-pinch pose step the tool along +t (image-left) by the measured offset in 1 cm steps, re-image
each step, until the knob is between the pads; if a step is refused, log which keep-out box and the gripper point
that hits it, and say so.
