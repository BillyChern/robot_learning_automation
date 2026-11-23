Operator note 17:58 (episodes 0-3 all ended with the drawer's bottom ON the rim: bottom 0.095-0.103 m vs rim 0.095, a
force stop at every downward step, learned x/y corrections within +-5 mm, and "opening not measured (edge not found)"
so alignment used the stored rim). The clearance is only a few mm per side (drawer ~10.5 x ~7 cm vs opening 11.5 x
7.6 cm), so the stored geometry (+-1 cm) is not enough; fix the alignment before the descent, not by pushing:
1. Hold the drawer ~2-3 cm above the rim and take a wrist look-down there: the drawer's bottom edges and the opening's
   inner edges are both visible in one image (the opening's edges are sharp and high-contrast in wrist_opening.jpg);
   measure the offset and yaw error directly (drawer vs opening, same image, no calibration needed) and correct it.
   Repeat once. Fix the opening detector (it failed in all episodes) or detect the wall tops as bright strips.
2. Then descend; on contact, back off 3 mm and make a lateral/yaw search step larger than the clearance error
   (2-4 mm, 1-2 deg), not a sub-mm step. A slight tilt (2-3 deg, leading edge low) to catch one wall first and then
   straighten is the classic peg-in-hole trick.
3. Keep the RL action meaningful: residuals on top of the measured alignment with a range of +-8 mm / +-4 deg.
