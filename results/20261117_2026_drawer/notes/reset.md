Operator note 23:43 (attempts 155, 156): 10 end-wall table picks in a row closed on nothing (reading 100), every one
after refine_wall shifted the grasp by -0.052..-0.057 or +0.033..+0.040 m (max_shift is 0.06, so all were accepted).
The drawer sits next to the cabinet (cabinet dist ~0.065): the strongest step in the profile is most likely the cabinet
edge or the drawer's far wall, not the near end wall. Same failure as attempt 150 (my 23:12 note). Do this now:
1. After a refined pick misses, the next pick of that wall uses the UNREFINED side-camera point (or the refine with
   |shift| <= 0.015); never refine-miss twice in a row.
2. Cap max_shift at ~0.02 and require the step to be on the drawer side (between the side-camera wall estimate and
   the drawer centre), excluding pixels that project onto the cabinet.
3. Test offline on attempts 150, 155 and 156's wrist_* images: the shift you pick should put the pads around the wall.
