Operator (21:14): full_v7 is ON HOLD, and the choice of its recipe is yours, after eval 18. The evidence so far,
all under the same conditions (240 s, 5090 serving, eased chunks): full_v5 4/6 in eval 18 vs full_v6 3/20 in eval 17.
full_v6 = full_v5 + bee_v6 (more DAgger corrections, a few round-3 demos) + 30k more steps, and it got worse.
The queued full_v7 (init full_v6, bee_v7 = even more corrections) repeats that step, so the operator stopped
scripts/train_when_free.sh before it could start (it had not started; the GPU was never free). Things to weigh:
- Base: full_v5 is the better model now. If you stack another 30k steps on it, consider fewer steps or a lower LR.
- Corrections: 44 normal-wrist DAgger successes exist; 105 clean collect demos (bee_v5 had 94). Check what the
  corrections from eval 12 (full_v1 failure states) look like, e.g. recover phases or odd poses, before using them. The
  ones from evals 15/18 (full_v5's own failures: hovering 3-4 cm beside the bee) are the most relevant.
- More clean demos: the robot can collect while the 4090 is blocked (aaron's jobs, ~17 GB).
When you have decided, re-arm it yourself (run on the 4090 machine, in tmux):
  bash scripts/train_when_free.sh runs/20261104_1510_bee <dataset> full_v7 --num-train-steps <N> --batch-size 4 --pytorch-weight-path <init>
(compute the dataset's norm stats first; after_training.sh full_v7 is already waiting in window after_train).
Full fine-tuning only, on the 4090 only, checkpoints every 5000 steps at most.
