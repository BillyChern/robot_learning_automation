
KEEP-OUT ESCAPE (01:40): resets 13-14 stopped with "no escape path from the start pose": the arm, left at the cabinet by
RL attempt 17, had the gripper 0.5 mm inside the cabinet's front keep-out layer, and the model check refused every
move that touched the box, including moves out of it. The check now lets a move that STARTS inside a keep-out box
leave it, as long as no gripper point goes deeper than at the start (+1 mm). Verified from that pose: "back 3 cm"
passes, "2 cm deeper" is refused. Your escape planner should now find a path; plan it straight back along -normal first.
