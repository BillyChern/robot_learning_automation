The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 73.2): object colour 0.22 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000388/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
Operator (23:14): eval 13, ep 388: full_v5 went from the look pose straight to the bee (0.5 cm), descended and closed
on it (reading 73.2), then sat at the grasp height until the time limit: 664 chunks refused by the 8 mm tip floor.
With the gripper closed the model puts the tips 47 mm below the tool (real about 38 mm), so at grasp height every
move, even straight up, starts "below" the floor. execute.py lifts with LIFT_TIP_MARGIN -2 mm for that reason.
rollout_floor now uses -2 mm while the gripper reading is above 50, for its own check and the session's
(s.run_segment(..., tip_margin=margin)); open-gripper chunks keep 8 mm. Rollouts started after 23:14 use it; ep 388
did not. Report it as "grasped, held at the floor by the rollout margin".
Operator (23:18): two more eval 13 findings.
- ep 390 stopped after 3 chunks: "measured joints differ from plan start by 0.75 deg". The session requires the
  plan start within 0.5 deg; rollout_floor planned from joints read before inference (~0.2 s) while the arm was
  still settling (full_v5 moves more decisively than full_v1). It now re-reads the joints after inference and plans
  from them; the observation is unchanged. (Workspace commit on the 5090, synced to the 4090.)
- ep 389: jaw yaw 165 puts joint 6 at 185.6 deg, -174 in the wrapped state, where bee_v5 has few frames (about 4 %);
  the policy crept 2 mm per chunk for 2 minutes. If yaw > 150 deg recurs among the failures, the next dataset or
  recipe should handle the joint-6 wrap: e.g. sin/cos for q6 as in your recipe_v3, or a look-pose yaw choice that
  keeps joint 6 inside -150..150. Count it over the 20 rollouts before deciding.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 74.6): object colour 0.16 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000394/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
Operator (23:34): ep 394 proved the closed-gripper margin works: full_v5 grasped (74.6), lifted to 0.22 m and
moved toward the bowl (19.9 -> 14.8 cm), then "tracking deviation 1.04 deg / 9.4 mm at t=0.21 s" stopped it. Each
stock chunk starts from rest, and full_v5's carry chunks are fast. rollout_floor now uses your rollout_smooth
EasedSegment with a 0.4 s ramp (as rollout_v3): chunks take about 1.07 s instead of 0.67 s, checked offline. ep 392
closed on empty air 5 times (99.1): a miss, policy precision. Rollouts from now on run with both fixes. Results
before about 23:40 were affected by the rollout problems (388: margin, 390: plan start, 394: tracking). Report them
separately when you summarize eval 13.
Operator (00:13): eval 13 pattern so far (rollouts after the three rollout fixes plus the earlier grasps):
jaw yaw 68-85 deg -> 3/3 grasped within about 1 cm (388, 394, 395; 395 completed the task). Jaw yaw >= 105 deg -> 0/5
(389: 165, 392: 145, 396: 108, 398: 126, 399: 137): the gripper stops 1.6-21 cm from the bee, never below z 0.05,
and closes on air. bee_v5 has 27 of 94 demos with look yaw > 105 deg, so it is not a coverage gap; the policy is
weaker there. If eval 13 ends below 70 %: bee_v6 = bee_v5's recipe with the round-3 demos plus the normal-wrist
DAgger corrections from evals 12 and 13 (--modes collect,dagger --wrist normal --start-phase approach --drop-idle).
The corrections of these high-yaw misses are exactly the missing data. Train full_v6 from full_v5/30000 (full
fine-tune, >= 30k steps, batch 4, on the 4090). Before training, check the yaw breakdown of the corrections.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 69.3): object colour 0.19 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000411/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
