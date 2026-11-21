Operator note 01:17 (attempt 170): the knob centring does not converge: measured +7.5, +6.8, +5.7 mm after each step,
so the aim walked from -0.055 to -0.035 (2 cm), and each back_for_tab move pulled 73.3, 82.2, 88.6 N (force change):
the pre-closed pads are around the knob and drag it on every back-off. Fixes:
1. Open the gripper (to ~0.4) before every back_for_tab move, and back off along the tool axis under low_force(25).
2. If a step does not shrink the measured offset by at least half, stop stepping (the measurement is biased or the
   knob moves with the pads) and pinch at the best aim so far; cap the total walk at 12 mm from the first aim.
