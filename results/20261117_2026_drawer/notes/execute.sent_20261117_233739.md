Contact-force stop is back at 50 N (user, 23:10): task yaml `rl.force_limit_n: 50`. A segment stops when the estimated
tool force changes by more than 50 N. Without a stop, RL attempt 5's lift caught the rail above the slot, reached 99.6 N
and moved the cabinet 6 cm; drawer pulls and pushes so far needed 20-40 N, so 50 N leaves room for them. Do not add a
lower stop of your own. The cabinet was put back by hand and re-measured before this restart (see the operator log).

Scene after RL attempt 5 (restored 23:28): the arm was freed by straight-line moves (back from the cabinet, up, Home)
and is at Home. The cabinet did not move: its front face (above the drawer) measures 10 mm behind the yaml plane,
the same as the reset agent's triangulation before (11 mm), top 0.271 m. The orchestrator's "cabinet moved 6.0 cm"
came from the cabinet blob's centre shifting with the gripper beside it and the drawer sticking out. The drawer now
sticks out of the bottom slot by ~10 cm; the reset takes it from there.
