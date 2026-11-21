Operator note 22:34 (attempt 148): the aim along t is now right (tab measurement 0_1: -0.1 mm). What fails is depth/height:
- Pinch: gripper closed to 90 both times (nothing held). In wrist_pin_closed0_1.jpg the pads close at the bottom of the
  image, below/in front of the knob's end; the knob is still beyond the pads, so the pads close on air in front of it.
- Press-drag: press_tab max force 1.5 N, drag_out 5.6 N: the tips never loaded the tab (it slides at 2-6 N only once
  it is moving; it sticks at ~50 N). The drawer moved ~1 mm.
Try the pinch ~8-12 mm deeper along N (and a few mm higher), stepping until the gripper reading stops at ~70-85 (held)
instead of 90; the reset keep-out allows 4 cm into the slot and the modelled closed tips read ~1 cm deeper than the
real ones. For the press-drag, press until the force rises by 5-10 N (or 5 mm), not a fixed 3 mm, before dragging.
