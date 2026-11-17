The 5090 disk was 95% full: each lora_v1 checkpoint is about 8.9 GB, and one was kept every 1000 steps.
After lora_v1 finished, the operator deleted the in-between checkpoints; 5000, 10000 and the final one remain.
New trainings now keep one checkpoint per 5000 steps by default (keep_period 5000).
If the 15k checkpoint misses the target, do not start a new run from scratch. Your learning rate is constant
(5e-5 after warmup), so continue the same run instead:
--exp-name lora_v1 --resume --num-train-steps 30000
That gives the same result as training to 30k from the start, and it needs about 30 GB of disk.

Your two evaluations of lora_v1/14999 (01 and 02) never ran a rollout. The policy server on the robot machine ran
out of GPU memory: other users' jobs use 18.7 of the 4090's 24.5 GB. The policy server now runs on the 5090
(hosts.policy_host Worksation_5090; the rollout connects to 100.122.29.80:8000).
Retry the evaluation as a run on the robot machine, the same way as attempt 4:
rlpj eval 20261104_1510_bee --checkpoint <run>/checkpoints/lora_v1/14999 --n 20
Do not wrap it in ssh to the 5090: that was attempt 3, and it cannot reach the robot.
The 5090 disk now has 147 GB free: only checkpoints 5000, 10000 and 14999 of lora_v1 are kept.

Your evaluation through the local TCP proxy (started 02:36) could not run any rollout: at about 02:37 the robot's
network link went down ("No route to host" to 192.168.2.9), and the operator paused the run. When this note
reaches you, the link is back. Nothing else changed. Retry the evaluation as a plain robot-machine run (above); the
orchestrator now starts the policy server on the 5090 itself, so you need no proxy or detached server of your own.
