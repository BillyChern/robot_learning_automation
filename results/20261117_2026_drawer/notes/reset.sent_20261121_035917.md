Operator note 04:00 (attempt 194: "no escape path from the start pose"): the arm is at tool [0.432, 0.208, 0.074] with
the gripper fully open (reading 0.9). With open fingers the model puts a finger 7 mm inside cabinet_front (the side
box), and every escape (back 4 cm, up) was refused for keep_out cabinet_front. I replayed 01_escape_back and
02_escape_up offline: with the gripper half closed or more (frac 0.5-0.9) no finger is in any box and both escapes
pass. So at a low start near the cabinet, pick the gripper opening per escape candidate: try the escape with the
current opening, then with the gripper closed to ~0.5-0.9 (nothing is held: reading < 3 means empty), and command
that opening before moving. Opening fully is only needed when the pads are around the knob or a wall (reading 3-97).
