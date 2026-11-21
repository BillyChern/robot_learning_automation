Operator note 04:50 (episode 39, attempt 104): good: the wrist footprint is in and the grasp held; and the new library
force stop worked: ins10 stopped at 66.5 N TOTAL (change only 25.4 N), which before 02:37 would have pushed on.
The jam came from the side again: "align: t_span 0.058 too small to trust, tool on the slot centre", "t err -0.073".
The side camera reads the drawer ~5-7 cm off in y (your own wrist footprint vs side camera in this episode: [0.355,
0.079] vs [0.361, 0.005]), so its in-hand t error is not usable. Use the wrist footprint instead: at grasp time you
know the footprint centre and the grasp point, so the drawer centre's offset along the front wall from the tool is
known (project grasp point - footprint centre onto the wall direction). Put the DRAWER centre, not the tool, on the
slot centre: tool t = slot centre + that offset. Keep the jam rule (stop at > 35 N with < 3 mm progress, back out, retry).
