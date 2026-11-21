Operator note (sim policy ready, please integrate now): /home/r2r/robot_learning_final_pj/sim/drawer_insert/ has a
trained insertion policy (NVIDIA Newton GPU training, MuJoCo CPU twin for eval). Sim results, 500 randomized episodes
each (clearance 3-10 mm, in-hand offset +-30 mm / yaw +-10 deg with noisy estimates, slot pose +-10 mm, friction,
force noise): carry-and-insert like yours: policy 96.6% vs wiggle 64.0% vs straight 36.2%; aligned-on-table start
(the user's strategy): 98.4%; starting jammed: 96.2%. Read sim/drawer_insert/README.md first (frames, tool point,
yaw_straight, force sign, how to compute the step peak force).
API (numpy only, runs in the robot .venv):
    from sim.drawer_insert.policy import Policy, Insertion
    pol = Policy(".../sim/drawer_insert/weights/policy.npz")
    ins = Insertion(pol, normal=N, offset=OFF, t_centre=mean(SLOT_T), yaw_straight=..., off_est=..., yaw_est=...,
                    tool_xy=T_now[:2, 3], tool_yaw=..., f0=force())
    loop: xy, yaw = ins.next(force()); run a short model-checked line there (s.executor.force_limit ~40 N, the README
    recommends 40 N for these steps); peak = result["max_abs_force_n"], stopped = contact stop; done = ins.done(...)
Known gaps: (1) your gripper cannot reach flush while holding (the keep-out stops a vertical gripper ~3.6 cm in front
of the face): run the policy until that limit or until the back half is in, then release and use your scripted push;
(2) it acts in the horizontal plane only (lift over the ~1.5 cm floor lip as you do now); (3) the force estimate has
pose-dependent bias: use force minus its value at the start of the insertion (f0), as the policy expects.
Run it as a strategy arm next to your current one (e.g. alternate episodes), log which arm ran, and report results.
