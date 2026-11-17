Operator (13:05): why full_v1 scored 0/20, and what to do now (the robot is idle, the 5090 GPU is free).

Diagnosis (eval 08, episodes 284-319). 16 rollouts were stopped by the keep-in check at the same spot every time
(tool about (0.57, 0.23, 0.50), wherever the bee was), 3 ran out the time without closing the gripper, and 1 (ep 307)
got low near the bee, then hit a tracking deviation. The gripper never closed. The policy replays the demonstrations'
opening look_bowl move: home -> a camera view above the bowl, about 20 % of each episode and almost the same every
time. Older episodes did it as one joint move that swung out to y 0.29. 50 % of collect+DAgger episodes go past
y 0.15; bee_v4 still has 40 % past 0.15 and 14 % past 0.20, so the y <= 0.26 filter does not remove it.

Fix: the scripted move to the look pose is not part of the policy's job. Start every rollout at the look pose and
train on episodes that start there.

1. Now: write programs/rollout_look.py. Do execute.py's perceive (bowl and bee), pick_yaw, to_ready, then
   travel(look_bowl) to tool-down above the bowl at table z + LOOK_Z with that yaw. Use the same functions, checked,
   inside the keep-in box (import them from the execute workspace or copy them), no bowl refine. Then run the policy
   loop from there. Wrap the rollout module the checkpoint needs: rlpj.vla.rollout for full_v1, your rollout_v4 for
   full_v4. Dry-run it, then evaluate full_v1/30000 with it: 20 rollouts, DAgger on. Serve on the 5090
   (hosts.policy_host). The operator copied full_v1/30000 (weights, assets, train_info) to the 5090 at the same
   path; your ckpt/full_v1 shim there pointed at files that only existed on the 4090. full_v1 was trained on the
   frames after the look pose, so this is in distribution.
2. Dataset bee_v5: all successful collect episodes, each trimmed to start at its "approach" phase (phases.json i0,
   i.e. after look_bowl and the bowl refine), plus the DAgger corrections as they are (they have no look move).
   Trim the steps and the videos together. Check that the first frame of a converted episode is at the look pose.
   Rebuild your sel/ links first: sel/bee_v4 still points to /data/sonic/rlpj/..., which was deleted; the run is
   back at runs/20261104_1510_bee on /home.
3. When full_v4 finishes (about 17:05; do not start another training run on the 4090 before then): evaluate it with
   rollout_look.py wrapping rollout_v4, 20 rollouts. This supersedes the plain rollout in the after-training note.
4. If neither reaches 70 %: full fine-tune full_v5 on bee_v5 on the 4090, >= 30k steps, batch 4, from the better of
   full_v1/30000 and full_v4/30000. Then evaluate it with rollout_look.py.
Report results as "policy from the look pose". The scripted part is only the camera move; grasp, lift and place
are the policy's.
