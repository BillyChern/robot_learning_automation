
JAMS VS YOUR HELD-DRAWER MEASUREMENT (03:08), for the tuning turn. Held-drawer side-depth measurement per episode, and
the cumulative yaw your policy applied:
  ep0 (att 23): t err -0.028, yaw err  +1.7 -> dyaw -4.9
  ep1 (att 25): t err -0.021, yaw err  +1.7 -> dyaw -2.0
  ep2 (att 27): t err -0.025, yaw err  +3.1 -> dyaw -1.8
  ep3 (att 29): t err -0.020, yaw err  +6.9 -> dyaw -5.5
  ep4 (att 31): t err -0.021, yaw err -12.4 -> dyaw +9.0   insertion reached its end, push jammed 44-52 N, 3.3 cm out
  ep5 (att 33): t err -0.017, yaw err -10.7 -> dyaw +9.8   insertion reached its end, push jammed 50 N, 1.4 cm out
  (att 20, before training: yaw err -10.3 -> dyaw +10.0)
The yaw measurement swings by ~20 deg between episodes while the grasp is the same; t err is a steady -2 cm. Before
trusting the correction, check whether these are real (e.g. the wrist camera or side image of the held drawer
against the slot edges, or the drawer's rim lines) or a partial-view bias. A drawer turned 10 deg in a slot it barely
fits would jam exactly like ep4-5.
