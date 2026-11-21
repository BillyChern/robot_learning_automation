Operator note (sim policy v2 with height correction is ready): sim/drawer_insert/weights/policy_v2.npz (numpy, loads in
the robot .venv; README v2 section). The sim now has the 76-82 mm opening, the floor/lip 13-17 mm above the table, the
rail, grasp pitch/roll slip, and a 4th action dz (+-3 mm/step, +-12 mm total). Sim results (500 eps): wedge starts
(too high / too low) v2 85% vs straight 17%, wiggle 8%, v1 19%; mixed heights 86-90% vs v1 42-45%. It fixes height
first (lip catch: +3 mm in 3 steps; rail: -1 to -2 mm), sideways search within 11 mm. To use it:
1. Insertion(..., tool_z=<start tool z m>, gap_est=<drawer bottom above the slot lip m>), gap_est = start tool z -
   drawer height - lip top (table + ~0.015), with the tool point at the top of the front wall (see README).
2. After each xy, yaw = ins.next(force()), command the tool height ins.z_target_abs in the same step (tool vertical).
3. ins.done(peak, stopped, xy, yaw, tool_z) with the measured tool z; jam_rep is counted inside Insertion.
4. Keep the guards: hand over after repeated same-depth jams, tighter t/yaw bounds for the aligned start, Z_BOUND 12 mm;
   log ins.obs and ins.step every decision.
Run v2 as the sim arm (keep v1 out), alternating with your own steps; report heights and results.
