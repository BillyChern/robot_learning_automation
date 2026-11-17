
NEW: DAgger is on (evaluation.dagger: true). After every failed rollout in an evaluation, the expert (the execute
program, about 93 % success) takes over from the policy's last pose and finishes the task. The takeover is recorded
as an episode with meta.mode "dagger" (with dagger_of = the rollout episode). It does not count in the success rate.
dataset_stats reports the counts under dagger_episodes.
To use the corrections, convert with both modes: rlpj vla convert --name <new> --modes collect,dagger (successful
episodes only, as before).
A suggested loop when an evaluation is below 90 %:
1. evaluation, which also yields one correction per failed rollout;
2. convert bee_v1 + corrections into a new dataset;
3. train on it (from lora_v1/14999's weights if you can, or from the base checkpoint);
4. evaluate again, which collects more corrections.
You can also run a larger evaluation (e.g. --n 30) just to gather corrections. Choose between this and continuing
lora_v1 to 30k based on what the failures look like (videos: episodes/ep_*/side.mp4, wrist.mp4).
