
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
