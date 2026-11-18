
Attempt 16 (01:17): after a reset that passed the orientation check, your insertion got the drawer about two thirds
in, the right way round. The release failed: during `up_from_front` the gripper still read 86.9, i.e. it was still
pinching the drawer wall, so the lift raised the drawer front into the rail above the slot (58.2 N, force stop). Check
the gripper reading after opening and before lifting: open until it reads well below the wall width (or fully), and
back away from the cabinet along -normal before going up. The arm is left at the cabinet; the next reset starts there.
