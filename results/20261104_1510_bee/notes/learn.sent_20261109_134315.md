Operator (12:13): the user's decision, which I failed to pass on at 01:45. Full fine-tuning runs on the 4090
only, never on the 5090. The 5090 lost power three times under sustained training load on 11-07. full_v6 trained on
the 5090 (host ishitas-workstation, 01:47-08:54) and it held this time, but do not do that again. If the 4090's GPU
is taken (another user's LEO jobs use about 17.5 GB of its 24 GB; a full fine-tune needs about 22 GB), wait and tell the
operator (need_operator), and run evaluations or collection on the robot meanwhile. Everything else stands: full
fine-tuning only, >= 30k steps, batch 4, checkpoints every 5000 steps at most. Serving from the 5090 is fine.
The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 70.2): object colour 0.24 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000450/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
