
RECURRING TABLE-PICK MISS (04:20): 3 of the 4 failed resets since training began (attempts 64, 69, 72) end with three
"end wall grasp reading ~100: nothing held" and "drawer 15-18 cm from the target after 3 picks". In all three the
drawer lies at about (0.32, +0.04) after the extraction, and your table OBB there is L 0.181-0.195 x W 0.147-0.171,
while the drawer is ~0.155-0.16 square-ish: the outline is inflated (something next to it merges into the blob), so
the end-wall point ([0.27-0.31, 0.10-0.14]) is ~2 cm off and the wrist refinement only moves it 1-9 mm. Ideas: clamp
the OBB to the known drawer size (fit the rim edges, not the blob extent), refine the wall with a larger search
window, or place the extracted drawer further from the cabinet before the table pick. Each miss costs ~7 min.
