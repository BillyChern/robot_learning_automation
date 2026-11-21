Operator note 22:27 (attempt 147, measured from your own logs and images):

The slot geometry is right; your measure_tab was wrong in attempt 147. In wrist_tab_0_0.jpg (first pinch, aim t ~ -0.043)
the knob sits between the pads, about 85 px (full res) right of u_pad 615 = ~-11 mm, so the knob is at t ~ -0.054
(matches attempts 140-145: -0.055). measure_tab returned u_tab 1239 (13 rows, cents 1224-1260): the grey table beyond
the cabinet's right edge, a run touching the image edge, accepted by the `x > len(g) - 25` flank exception, then
clipped to -0.060. That set knob_t = -0.103, so the -60 mm step was refused, the pins aimed at -0.097..-0.109, and the
press-drag aimed at t ~ -0.10 to -0.11, inside the left cabinet_front box (t < -0.087) -> refused.

Fix now (small, in measure_tab):
1. Drop runs that touch the image edge (x0 < 5 or x > len(g) - 5); a knob cut by the edge gives a wrong centre anyway.
2. Treat a saturated or implausible result as not found: |dt| > 0.035 -> return None (fall back to the judge).
3. Never store knob_t outside the slot window (scene.SLOT_T, 1 cm inside each end); clip, or ignore it.
4. With the knob between the pads, |dt| <= ~12 mm: pinch at that aim, then pull.
