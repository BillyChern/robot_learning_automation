Operator (14:00): your ep 341 check worked. The look move went fine, and from the look pose full_v1 went straight
down to the bee: tool at (0.495, -0.156), bee at (0.522, -0.161) along its long axis, jaw yaw 90, gripper still
open. Then the session refused the next chunk: "model check failed for 26_policy_23 ... min_tip_above_table 0.0063".
The session checks every segment with an 8 mm fingertip margin; the scripted grasp stops at table z + GRASP_Z
(0.052 m), above it. The rollout's own pre-check used 3 mm, so it let that chunk through; its default is now 8 mm, to
match. The 8 mm floor stays (do not lower it). full_v1 went about 7 mm below any demonstration's grasp height.
Next, in rollout_look.py (and _v4): when a chunk fails only on tip clearance, run its longest prefix that passes
the check (the arm holds at the floor), and take the gripper command from any action in the chunk, not only the
last executed one. In the demonstrations the close comes right at the bottom of the descent. Stop with "fail" after
3 chunks in a row held at the floor without a close. Do not close for the policy. Dry-run it, then run the
20-rollout evaluation of full_v1/30000 from the look pose. At about 17:05 full_v4 finishes, and after_training.sh
switches policy_host to local and asks for a restart; start the 20 rollouts soon enough to finish before then.
Serving: scripts/serve.sh now accepts a checkpoint with model.safetensors and no params/, on both machines, so pass
checkpoints/full_v1/30000 directly. Your shim crashed at 13:06 because its model.safetensors link pointed at a
4090-only path then, and openpi falls back to JAX when model.safetensors does not exist. The 5090's openpi venv is
back to torch cu126 (no sm_120), because uv run re-syncs it to the lockfile. Keep using your own serve program
there, and set UV_NO_SYNC=1 for every uv run on the 5090.
