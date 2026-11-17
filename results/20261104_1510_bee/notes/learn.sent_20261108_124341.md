
Update (11-08 10:20, operator): full_v4 (your attempt 27) did not fail because of your recipe. The 4090's disk filled
at 08:12 while it saved step 10000, which killed the training and the orchestrator. The disk is fixed (the run is
back on /home, about 200 GB free, and unrelated projects were removed). The operator resumed full_v4 from its step
5000 checkpoint with the same recipe, config and settings (scripts/train_guard.sh in tmux window train4090: it
resumes by itself after a crash). Do NOT start full_v4 again or start another training run while it runs (one GPU).
It saves every 5000 steps (the newest kept whole, 10000-step multiples as weights only), so checkpoints stay small.
Meanwhile agents 1 and 2 finish collection round 2 (PHASE collect). When full_v4 finishes, after_training.sh switches
the run back to the learn phase: evaluate the final full_v4 checkpoint with your v4 rollout (joint deltas, gripper-only
state, H=30). Full fine-tuning only, >= 30k steps.
