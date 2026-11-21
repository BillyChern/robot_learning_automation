Operator note 15:20 (attempt 114, your v2 test): the drawer never entered the slot. push_end_wrist.jpg: the dark slot
opening is at the upper LEFT of the view, the drawer's right part reaches past the cabinet's right side post (grey
table visible behind it): the drawer is ~2-3 cm sideways of the real opening although "table align 2: t err +2.4 mm".
So the slot position you align to (yaml slot_t centre -0.0585 since my 05:20 shift, whose t part came from noisy
cabinet-outline readings) is probably off by ~2-3 cm. Also push_end_side.jpg shows the whole cabinet leaning back
under the 65-68 N pushes (it settled back afterwards, side_after.jpg).
Fix with a direct measurement instead of the yaml:
1. Before the insertion (drawer held ~4-6 cm in front of the face), take a wrist image that sees the opening, detect
   the opening's left/right edges (dark interior vs the lighter face/posts), and compare with the yaml edges projected
   into the image (rlpj.perception.wrist.project_base_to_wrist) -> the slot's t offset in the base frame, no sign
   guessing. Shift the insertion by it (log it: it tells us the true slot centre).
2. Also check the drawer's left/right walls against the opening edges in the same image before pushing in.
3. Pushes: stop at 40 N (not 65) while the drawer is not yet inside the opening; 65 N only once the back wall is in.
Report the measured slot offset in your next report so I can correct the task yaml.
