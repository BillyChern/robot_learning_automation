Operator (10:14): full_v6 grasps (421, 422, 423 all closed on the bee, 0-1.2 cm), but two grasps went about
15 mm deeper than the demos (tool 37 mm above the table vs 52 mm). With the gripper closed, the model then puts the
tips below -2 mm, so every lift was refused: 421 held 290 chunks until the time limit; 422 was aborted by the session,
because your pre-check used the reading 0.56 while the session checks the commanded 1.0. rollout_floor now (1) checks
with the session's own grip_frac and table z, and (2) with the gripper closed allows any chunk whose tips stay at or
above where they are now (0.5 mm tolerance) and >= -2 mm, whichever is lower. Lifting out of a deep grasp passes,
pushing down never does (tested on 422's pose). 423 lifted, carried to 1 cm over the bowl and released, then a 1.08
deg tracking deviation stopped it; verdict unsure. Please check its images. Rollouts from 10:14 on use this.
Operator (10:15): the full_v6 evaluation has run under different lighting. Side-camera brightness (median V of
before_side.jpg) is 94-95 for every full_v6 rollout since 09:00, versus 136-139 for the full_v5 eval (23:00-01:30)
and a median of 136 in bee_v6 (darkest 10 % about 93). The room lights are off or dimmer this morning; the operator
has asked the user to switch them on. Report full_v6's dark rollouts separately; they are not comparable with
full_v5's 7/20. ep 423 is a success: scene_after.jpg shows the bee in the bowl, and the VLM's "no" is from the dark
image. Count it after you check it yourself.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 76.3): object colour 0.16 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000434/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
Operator (11:43): full_v6 stands at 2/18 (3/18 with ep 423, which is a success) against full_v5's 7/20, but
under different conditions: half of full_v6's rollouts were in the dark (brightness 94 vs 137), it is served from the
5090 (inference about 260 ms vs 135 ms locally), and the rollout now uses eased chunks. Successes now release at
103-115 s of the roughly 128 s of policy time, and ep 434 timed out over the bowl. Before choosing the base of the next
full fine-tune, run an A/B check while the 4090 is busy (another user's jobs hold about 17.5 GB, so no training can start
there now): when eval 14 ends, evaluate full_v5/30000 again with the current programs/rollout_look.py, 10 rollouts,
DAgger on, served the same way as full_v6. If full_v5 still clearly beats full_v6, start the next run (full_v7) from
full_v5 and look at what bee_v6 added: the eval-12 corrections came from full_v1's failure states, and 60k total
steps on about 100 demos may overfit. Keep full fine-tuning, >= 30k steps, batch 4, on the 4090 once it is free.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 77.6): object colour 0.13 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000442/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 77.2): object colour 0.15 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000443/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
