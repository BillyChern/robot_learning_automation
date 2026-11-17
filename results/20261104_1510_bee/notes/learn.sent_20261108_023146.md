
Safety change (2026-11-07): a policy rollout (episode 209, tool y 0.295 at 0.44 m height, gripper reading 65)
grasped the curtain beside the table (y +0.31..+0.37). The arm faulted and the side camera was dragged out of place.
The robot library now refuses any segment, including each policy chunk, where a gripper point leaves
y -0.45..+0.30 (config robot.keep_in). A rollout that tries is stopped there.
The policy learned that move from the demonstrations: in 46 of 119 collection episodes the execute program's joint
moves between Home and its bowl-look pose swing the tool to y 0.29-0.30. The execute agent will reroute them.
Consider leaving those segments out, or down-weighting them, in your next dataset, and expect lora_v4 to be stopped
when it imitates the swing.

The robot machine (4090) rebooted at about 16:20. Your blocking wait on lora_v4 ended with it, but the training
itself was not affected: it runs detached on the 5090 (step 6.2k of 12k at 16:22, about 1.9 s/step, done around
19:25; log runs/20261104_1510_bee/logs/train_lora_v4.log). Wait for it again, then evaluate.
The side camera was recalibrated at about 16:40 after the curtain incident: it is now at (1179, 44, 605) mm, was
(1178, 56, 603) mm.

The operator asks for a policy test now, without waiting for lora_v4 to finish. Screen its latest saved checkpoint,
runs/20261104_1510_bee/checkpoints/lora_v4/7000, with about 5 rollouts on the robot. DAgger is on, so failures also
give corrections. Use the serve config and rollout program that lora_v4 needs (gripper-only state, wrap-safe
deltas). The policy server now runs on this 4090 (hosts.policy_host local, 127.0.0.1:8000): its GPU has been free
since the reboot, so it does not compete with lora_v4's training on the 5090.
The robot is ready: the side camera is recalibrated and checked (sideways +4.2 mm, table -17.4 mm, PASS), the arm is at
Home, and rlpj check passes. The new curtain keep-in box (gripper y <= 0.30) is active: a chunk that goes past it
ends the rollout.
After the screen, wait for lora_v4 to finish (12k, around 19:25) and evaluate the final checkpoint in full.

OPERATOR DECISIONS (2026-11-07 ~17:10), these override your earlier plans:
1. Target success rate is now 70 % (was 90 %).
2. Full fine-tuning only, no LoRA. lora_v4 was stopped at about 7.3k steps; checkpoints 5000 and 7000 are kept.
3. Every training run gets at least 30,000 steps.
The orchestrator now refuses any training command that names a LoRA config (or none, since the default is LoRA)
or has fewer than 30,000 steps. A refused command is not run and not counted.
Memory: openpi's JAX full fine-tuning needs more than 70 GB of GPU memory, and the 5090 has 32 GB. The operator is
setting up full fine-tuning with openpi's PyTorch trainer (bf16 weights and optimizer state, gradient
checkpointing), which may fit at a small batch size. Do not start a training run until a note says that path is
ready and gives the command. Until then: finish the lora_v4/7000 screen and analyse the data. The DAgger
corrections and every collection episode stay available for the next dataset.

Operator (17:20): the pipeline switches to collecting more demonstrations (round 2) while the operator runs the
first full fine-tuning (PyTorch, 30k steps) on the 5090 outside the agent loop. The lora_v4/7000 screen was
dropped: no LoRA any more. When the full fine-tuning has finished you will be resumed to evaluate it with DAgger.
Full fine-tuning now works on the 5090. Use: rlpj vla train --run-dir <run> --dataset <name> --exp-name <exp>
--num-train-steps 30000 --batch-size 8. The default config pi05_kinova_full goes to the PyTorch trainer: bf16,
gradient checkpointing, 8-bit AdamW, 23 GB peak at batch 8, 1.37 s/step, so about 11.5 h per 30k steps. The
operator's first run, full_v1 (bee_v3, 30k steps, batch 8), started at 17:40 and finishes around 05:10.
Serving its checkpoints needs the PyTorch policy path: the operator will check that before the evaluation.
Update (18:15): training now runs on the 4090. The 5090 kept losing power under sustained GPU load; its agents and
the judge are unaffected. hosts.train_host local: your convert and train commands run on the 4090 automatically.
full_v1 started at 18:13 on the 4090 (bee_v3, batch 4, 30k steps, about 1 step/s, done around 02:20), with
checkpoints every 5000 steps. Batch 4 is the most the 4090's 24 GB allows for full fine-tuning (22 GB peak); keep
--batch-size <= 4 on this machine. Do not start another training run while full_v1 is running.

Disk (01:35): the operator approved removing full_v1/10000 and full_v1/20000 once the first evaluation of full_v1
finishes (only ~46 GB free on this machine). Evaluate the final checkpoint; do not plan on the intermediate ones.
lora_v4 was deleted (LoRA is not used any more).

full_v1 finished training at 2026-11-08 02:28:45 on the 4090: final checkpoint /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000.
Evaluate it now: rlpj eval 20261104_1510_bee --checkpoint /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000 --n 20 (DAgger on; the default rollout fits it, since it uses the
standard state and action representation, so do not pass a --rollout). The policy server runs on the 4090
(hosts.policy_host local). If it is short of 70 %, convert a new dataset (all collection episodes plus DAgger
corrections) and train the next full fine-tuning run on the 4090: >= 30k steps, --batch-size 4.
