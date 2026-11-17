Two orchestrator bugs, both fixed now, cost you last night:
1. The 20-minute time limit meant for robot runs also applied to your GPU jobs. It killed your lora_h30 training at
   22:30. The limit now applies to robot runs only.
2. Your wait_train.sh polling runs were counted as training runs, because the command contains "train.sh". Four of
   them used up the budget of 6, the round ended as "max_runs", and the pipeline went back to collecting
   (collection round 2: 11/100). Now only a step that actually starts a training counts (rlpj vla train, scripts/train.sh,
   openpi_entry train). You have a fresh budget of 6 training runs this round.
lora_h30 (bee_v2, horizon 30, from lora_v1/14999, 12k steps) is still training detached on the 5090. It was at step
4.8k at 01:16, about 1.9 s/step, so it should finish around 05:05. Log: runs/20261104_1510_bee/logs/train_lora_h30.log.
A blocking wait for it is fine now: it is not counted, and there is no time limit on non-robot runs.
When it finishes, evaluate it with DAgger on (20 rollouts). The extra collection episodes from round 2 can go into
your next dataset.
For reference, evaluation 04 of lora_v1/14999: 0/20, with 18 DAgger corrections, 14 of them successful.
