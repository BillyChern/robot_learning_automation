
EXACT STALE CONSTANTS IN YOUR CODE (15:48): the slot is now 46 mm further along t (slot_t [-0.166, -0.091], centre
-0.1285; was centre -0.0825). Make these slot-relative before your next robot run:
  reset.py  PIN_T = -0.088 -> (slot_t[0] + slot_t[1]) / 2 - 0.0055   (= -0.134 now)
  reset.py  T_C   = -0.087 -> (slot_t[0] + slot_t[1]) / 2 - 0.0045   (= -0.133 now)
  insert.py T_C   = -0.087 -> same
  survey.py (-0.135, -0.026) -> slot-relative too.
A pinch aimed at the old PIN_T lands 46 mm beside the knob, on the cabinet frame.
