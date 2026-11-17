The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 75.0): object colour 0.19 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000540/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 75.0): object colour 0.18 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000546/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
Operator (00:58): full_v7 now trains on the 5090, with the user's approval after the user installed the power
mitigation there: 400 W, GPU clock locked at <= 2200 MHz, CPU 125/200 W. It runs at about 280 W instead of 400 W with
spikes to 460 W. Your recipe is unchanged: bee_v8, init full_v5/30000, 30k steps, batch 4, constant LR 1e-5, checkpoints
every 5000 steps. It started at 00:55 in tmux session train5090 on the 5090 (scripts/train_guard_5090.sh, which resumes
after a crash) and should finish about 09:20. The 4090 waiter (session learn_full_v7) was stopped. Do not start
another training run. When eval 19 ends, the run switches to collection (more clean demos), so the 5090's GPU is free
for training. scripts/after_training_remote.sh copies full_v7's weights back and sends you a note when it is done.
