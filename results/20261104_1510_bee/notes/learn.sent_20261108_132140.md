Operator (13:25): eval 09 never started a rollout. The policy server on the 5090 crashed at 13:06:40 (log:
runs/20261104_1510_bee/evals/server_8000.log on the 5090). Your shim agents/learn/ckpt/full_v1/30000 has an empty
params/ folder, so openpi took it for a JAX checkpoint: "Metadata file (named _METADATA) does not exist at
.../ckpt/full_v1/30000/params". Pass the real checkpoint instead:
--checkpoint /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000 (the same path now
exists on both machines; the orchestrator serves PyTorch checkpoints trained here directly), or delete the empty
params/ from the shim. The orchestrator now notices a crashed remote server within seconds instead of waiting 15 min.
Also: your 13:06 look-only dry run failed ("look: no bee-sized blob"; the bee was probably still in the bowl after
collection), but the attempt reported success because the pipe through grep/tail hid exit code 2. Add
"; exit ${PIPESTATUS[0]}" or check summary.json. The first real rollout follows a reset, so perception should find
the bee then; check the first rollout's log for the look move before you leave the evaluation running.
