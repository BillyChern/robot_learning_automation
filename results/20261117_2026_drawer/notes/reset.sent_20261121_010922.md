Operator note 01:10 (attempt 168): guarded_line protects only its stepped part; its "_near" move runs at the full 65 N
and that is where tonight's high forces are: deeper_pin_8_3_near 71.1 N, deeper_pin_12_3_near 62.7 N (the stop did not
fire because the total force stayed under 65 N while the change was larger). Run the _near move under
self.low_force(fmax) too (or ~30 N), and for the deeper pinch steps go back out to the pre-pose before moving to the
next deeper/higher target, so a step never shoves the pads sideways into the knob or face.
