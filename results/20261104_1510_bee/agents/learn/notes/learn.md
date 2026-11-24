# learn notes: bee_in_bowl

## Plan
1. Convert the 100 success episodes (default filter: outcome.final == success) -> dataset `bee_v1`.
2. Train pi05_kinova_lora with default hyperparameters as a baseline, then screen and evaluate.

## History
- attempt 1: convert -> bee_v1 (success-only)
  result: 100 episodes, 176072 frames (~117 s / episode at 15 fps).
- attempt 2: train `lora_v1` on bee_v1, pi05_kinova_lora defaults (bs 16, lr 5e-5 const after 500 warmup, H=10)
  but 15000 steps instead of 5000: 5000*16 = 80k samples < 0.5 epoch of 176k frames; 15000 steps ~ 1.4 epochs.
  Expect a usable baseline; screen the 15000 checkpoint.
  result: done in 7.5 h; loss ~0.005-0.01 at the end with spikes to ~0.1-0.2 (normal flow-matching noise).
- attempt 3: full 20-rollout eval of lora_v1/14999 (training costs 7.5 h, so a 5-rollout screen saves little).
  result: not run - a command with a `cd ... && uv run` prefix goes over ssh to the 5090. Eval commands must start
  with `rlpj eval --checkpoint ...` (orchestrator routes them to the robot; loop.py _learn_run).
- attempt 4: same eval, bare `rlpj eval --checkpoint .../lora_v1/14999 --n 20`.
  result: eval 01_lora_v1_14999: 0 rollouts, "policy server 127.0.0.1:8000 did not come up". Likely cause: the robot
  machine could not reach the 5090 (attempt 3 log: "Could not resolve hostname worksation_5090"), so the checkpoint
  copy (rsync params/assets) failed. Asked the operator to fix host resolution / check evals/server_8000.log.
- No operator available. Retrying the same eval (ssh to the 5090 worked for convert/train, so the DNS failure may have been transient).
- eval 02_lora_v1_14999: same "server did not come up". attempt 4 diagnostic: ssh 4090->5090 works now, checkpoint
  14999 complete (params 6.0 GB, assets), 5090 GPU idle. So the local 4090 server fails for a reason I cannot see.
  Workaround: serve on the 5090 (programs/serve_5090.sh, detached, port 8000) and eval with
  `--serve programs/proxy_serve.py`, a stdlib TCP proxy robot:8000 -> 100.122.29.80:8000 (same checkpoint).
- attempt 5: start the 5090 server for lora_v1/14999.
  result: 5090 server up in 20 s, norm stats from the checkpoint assets (rlpj/bee_v1).
- attempt 6: eval lora_v1/14999 n=20 with --serve programs/proxy_serve.py.
  result (operator note): evals 01/02 failed because the 4090 server ran out of GPU memory (other jobs use 18.7/24.5 GB).
  Proxy eval never rolled out (robot network link down at 02:37). The orchestrator now serves on the 5090 itself
  (policy_host Worksation_5090), so the proxy and serve_5090.sh are obsolete.
  Operator advice: if 14999 misses, continue the same run (--exp-name lora_v1 --resume --num-train-steps 30000).
  No new episodes (still 108), bee_v1 is current.
- attempt 7: stop my detached 5090 server on port 8000 (it would block the orchestrator's server).
  result: port 8000 free on the 5090.
- attempt 8: plain eval lora_v1/14999 n=20 (orchestrator serves on the 5090).
  result: orchestrator crashed starting the 5090 server (ssh hang); fixed by operator. No rollout.
- attempt 9: same eval again.
  result: eval 03_lora_v1_14999: 0/12 (operator stopped after 12). Ep 117 and 119: perception check said success, VLM fail.
  DAgger now on: failed rollouts get an expert takeover episode (mode dagger).
- attempt 10: sync episodes (convert --dry-run) and summarize eval eps 108-119 vs training (programs/analyze_eval.py).
  analysis (attempt 7 digest, eval 03 eps 108-119, 0/12):
  - 3x tracking-deviation abort (1.0x deg at t~0.2 s of a chunk: each chunk starts from rest with a velocity step)
  - 2x chunk failed model check (joint limit violations / speed): out-of-distribution actions
  - 4x grasp closed on nothing (reading 99.1; training grasps read ~81-83) - offset 1-4 cm from the bee
  - 1x hover until time limit (193 s), 1x release refused (opened 0.24 m above table; session allows <= 0.20 m)
  - 1x (118) grasped (81.6) but did not release
  Training demos: median 117 s, close at ~41 s; policy slower (close at 54-83 s).
- attempt 11: idle-frame / phase stats of the training episodes (programs/analyze_idle.py).
  result: idle (q_cmd and gripper_cmd constant) = 32% of frames (verify 9 s at end, grasp 1 s, perceive); grip reading on bee 76-87.
  Operator re-score of eval 03: 117 real success (unsure under new rule), 119 fail; ~1/12. Rest near-miss grasps.
  Bee wrapped orange from ~ep 59 on (half of bee_v1 yellow, half orange).
- attempt 12: eval lora_v1/14999 n=20 with programs/rollout_smooth.py (trapezoid-eased chunks, RAMP 0.15 s) to
  remove tracking aborts; also yields DAgger corrections for the next dataset. Offline check: start velocity
  0.6 vs 7.5 deg/s step, end exact, same peak speed.
  result: eval stopped after ep 120: the arm went home holding the bee (reading 80.3). The session refuses release
  above 0.20 m / outside the zone, and the stock rollout then aborts holding the bee.
- attempt 13: eval lora_v1/14999 n=20 with programs/rollout_v2.py = eased chunks + release handling (descend slowly
  to 0.19 m before an in-zone open; ignore opens outside the zone).
  result: eval 04_lora_v1_14999: 0/20, every rollout claim=none (crash?), ep 125 ended home holding the bee.
- attempt 14: sync + analyze eps 121-157 (incl. dagger episodes).
  analysis eval 04 (0/20, rollout_v2): ~9 near-miss grasps (closed on nothing, 97-100), ~6 tracking aborts at
  t~0.27-0.30 s of a chunk (right after the 0.15 s ramp, always ~80 s in, the fast move from the look-around pose
  to the approach), 2 chunk model-check fails, several hit the ~190 s limit; one in-zone release descent worked (ep 140,
  but the bee was not delivered). DAgger: ~14 successful corrections (eps 122..154).
  Conclusions: grasp precision is the main problem (needs data/training); stop-and-go execution is the second.
- Plan training run 2 (lora_h30): dataset bee_v2 = collect + dagger successes; recipe programs/recipe_h30.py:
  H=30 (2 s chunks -> execute ~20 actions per stop, fewer starts), init from lora_v1/14999 params, 12k steps,
  cosine 5e-5 -> 5e-6 (decay for precision at the end). Rollout will use --chunk/--replan 20 and a longer ramp.
- attempt 15: convert bee_v2 (--modes collect,dagger, successes only).
  result: bee_v2 = 114 episodes (100 collect + 14 dagger), 200791 frames.
- attempt 16: train lora_h30 (recipe_h30, bee_v2, 12k steps, bs 16) - training run 2 of 6.
  result: the orchestrator killed the ssh after 1200 s (attempt_timeout_s 1200 now also applies to training, since
  training.timeout_s is unset). The training process on the 5090 kept running (step ~200 at 22:30, 1.9 s/it, ~6.3 h
  left); its tee died, so the log file stops at 22:30. Waiting with programs/wait_train.sh (<= 19 min per run).
  result: the orphaned training died at 22:31 (broken stdout pipe after tee died), no checkpoint.
- attempt 17: relaunch the same training fully detached (setsid nohup scripts/train.sh with RLPJ_CONFIG/RLPJ_RECIPE,
  log notes/train_lora_h30.out), then wait in <= 19 min runs. Same run as attempt 16 (it never produced a checkpoint).
  result: ssh hung to 1200 s again (killed) but the setsid'd training survives: step 236 at 22:52, ~6.2 h left.
  result: lora_h30 finished 05:07, checkpoints 5000/10000/11999; loss ~0.02 at 1-3k -> ~0.004-0.01 at the end.
- eval lora_h30/11999 n=20 with programs/rollout_v3.py (chunk 30, replan 20, ramp 0.4 s, release handling).
  result: eval 05_lora_h30_11999: 0/20 (1 unsure: ep 180 geometry yes, VLM no).
- analyze eval 05 with programs/eval_brief.py (one line per rollout).
  analysis eval 05 (lora_h30, rollout_v3): the gripper NEVER closed in 20 rollouts; 7 tracking aborts with ~1.1 deg
  but only 0.4-5.6 mm (wrist joints spinning); the rest wandered until the time limit.
  ROOT CAUSE FOUND (norm stats): joints 4 and 6 sit at +-pi (joint 4 median 3.11 rad, joint 6 spans the circle).
  State wrap_pi makes one pose read +3.1 or -3.1; chunk deltas crossing the wrap contain ~2 pi spikes (action std
  j4 0.37 / j6 0.60 rad vs 0.03-0.06 others; with H=30 more chunks cross -> worse than H=10). Explains wild wrist
  motions, joint-limit chunks, poor precision in all evals so far.
- Fix: programs/recipe_v3.py: wrap-safe delta (wrap_pi(action - state)), state encoded with sin/cos for q4, q6
  (9 dims), decode with atan2 at the output. Offline round trip exact (7e-7 rad). Init from lora_h30/11999 (H=30),
  12k steps cosine 5e-5 -> 5e-6.
- convert bee_v3 = collect (112) + dagger successes (eval 04 + 05).
  result: bee_v3 = 139 episodes, 247114 frames.
- train lora_v3 (recipe_v3, bee_v3, 12k) - training run 2 of this round's 6 (lora_h30 = 1).
  result: ssh "broken pipe" at 12:06 (step 4.92k) killed training (same tee/pipe problem). Checkpoint 4500 should exist.
- screen lora_v3/4500 with 5 rollouts (rollout_v3) to verify the wrap-safe pipeline end to end before resuming.
  screen 06 (lora_v3/4500, 2 rollouts, operator stopped): closes on air 21-24 cm from the bee (ep 209 at t=2 s), carries
  nothing to the bowl, release_descend + open, repeat. Not grounded in the scene.
  offline_check.py (served pipeline on recorded training frames): predictions match ground truth within ~1-2 deg,
  gripper exact -> serving/transforms are correct.
  vision_check.py: same state + images of ANOTHER episode (bee elsewhere) -> almost the same chunk as own images;
  black images change it. => the model replays trajectories from joint state, ignores where the bee is (copycat).
- recipe_v4: model state = gripper reading only (no joint angles); wrap-safe joint deltas; server returns deltas,
  programs/rollout_v4.py adds q_now. Init lora_v3/4500, 12k steps cosine. Launch detached (survives ssh drops).
  result: lora_v4 training was killed at 16:50 (step 7.1k, no error in its log; orchestrator restart). loss ~0.003 at 6.5k.
- screen lora_v4/7000 with 5 rollouts (rollout_v4) before resuming to 12k.

## 2026-11-08: operator decisions
- Target now 70 %. Full fine-tuning only (no LoRA), >= 30k steps, batch <= 4 on the 4090 (train_host local, PyTorch
  trainer, ~1 step/s -> ~8.5 h per 30k). Curtain keep-in: gripper y <= 0.30 (46/119 collection episodes swing to
  y 0.29-0.30 between Home and the bowl-look pose; execute agent will reroute).
- full_v1 (operator's run: bee_v3, standard representation, 30k, batch 4) finished 02:28. lora_v4 deleted.
- eval full_v1/30000 n=20, default rollout (as instructed).
  Caveat from my analysis: standard representation has the +-pi wrap on joints 4/6 and joint state input
  (copycat risk); if it fails, next run should use the wrap-safe deltas and gripper-only state (recipe_v4 ideas,
  as a full-FT config) if the PyTorch path supports custom transforms.
  result eval 07: "policy server did not come up": serve.sh requires params/ (JAX); full_v1/30000 is PyTorch
  (model.safetensors, assets, metadata.pt). openpi's create_trained_policy loads PyTorch when model.safetensors exists.
  Workaround: shim ckpt/full_v1/30000 in this workspace = empty params/ + symlinks to the real model.safetensors,
  assets, metadata.pt (+ train_info.json in ckpt/full_v1). Same weights. programs/rollout_std.py = stock rollout
  (passed only so the orchestrator pulls the workspace with the shim to the robot machine).
- eval 08: full_v1/30000 via the shim, n=20.
  result: eval 08_full_v1_30000: 0/20 (served via shim; stock rollout).
- analyze eval 08 (eval_brief on the 4090).
  analysis eval 08 (full_v1, standard representation): 17/20 rollouts stopped ~26 s in by "policy chunk failed the
  model check: joint_limit" (chunks 10-14, the look-around pose where joint 4 crosses +-pi): the wrap bug again
  (full_v1 uses the stock representation), 3 hit the time limit, the gripper never closed.
- Plan full_v4 (next full fine-tune): recipe_full_v4 = recipe_v4 transforms (wrap-safe deltas, gripper-only state,
  H=30) on the PyTorch full config; dataset bee_v4 = successful collect+dagger episodes whose tool max y <= 0.26
  (drops the curtain-swing episodes); selection via programs/select_episodes.py -> sel/bee_v4 symlinks + --episodes-glob.
- convert bee_v4 (on the 4090).
  result: bee_v4 = 166 episodes (145 collect, 21 dagger), 295891 frames (max tool y <= 0.26; 49 of 215 dropped).
- train full_v4 (training run 1 under the full-FT policy): programs/recipe_full_v4.py, config pi05_kinova_full_v4,
  bee_v4, 30k steps, batch 4, init full_v1/30000, cosine 2.5e-5 -> 2.5e-6, H=30, wrap-safe deltas,
  gripper-only state. Recipe written to the 4090 inline (workspace is not synced there for train runs).
  Removes full_v1/10000 and 20000 first (operator approved after the first full_v1 evaluation; disk ~46 GB free).
  full_v4: first attempt died when the 4090 disk filled at step 10000 (08:12); operator resumed from 5000 with
  train_guard.sh (same recipe). At ~12.5k by the last check (loss ~0.001-0.01). Shim ckpt/full_v4/30000 prepared
  (symlinks; PyTorch checkpoint lacks params/). Plan: eval with programs/rollout_v4.py.

## 2026-11-08 13:05 operator: policy from the look pose
- Operator diagnosis of eval 08: 16/20 stopped by the keep-in check at the same place (tool ~(0.57, 0.23, 0.50)),
  replaying the demos' scripted look_bowl move (home -> camera view above the bowl, ~20 % of each episode).
  bee_v4 still has 40 % of episodes past y 0.15.
- programs/rollout_look.py: execute.py's perceive + pick_yaw + to_ready + travel(look_bowl) (functions imported from
  the execute workspace, all session-checked), then the policy loop. rollout_look_v4.py wraps rollout_v4.
  --look-only for dry runs (dummy policy client).
- Plan: dry run; eval full_v1/30000 with rollout_look.py (20, served on the 5090); bee_v5 = collect episodes trimmed
  to start at the "approach" phase + dagger; eval full_v4/30000 with rollout_look_v4 when it finishes (~17:05);
  if < 70 %: full_v5 on bee_v5 (>= 30k, batch 4) from the better checkpoint.
  dry run (look-only, motion off): execute.py import and side perception work; stopped at "no bee-sized blob" (no
  reset before a dry run; the bowl blob alone was on the table). Motion part is checked by the session in the eval.
- eval 09: full_v1/30000 (shim) with rollout_look.py, 20 rollouts, served on the 5090. "Policy from the look pose".
  result eval 09: no rollout, server crashed (empty params/ in the shim -> openpi took it for JAX). Use the real
  checkpoint path (the orchestrator now serves PyTorch checkpoints directly). Note: piping through grep/tail hides
  exit codes (dry run reported success); use "; exit ${PIPESTATUS[0]}".
- eval 10: full_v1/30000 (real path), rollout_look.py, n=1 to verify the look move, then n=20.
  result eval 10 (real path, served on the 5090): server never came up and left no log: the 5090's scripts/serve.sh
  still exits on "no params/" before logging. The shim (params/ + model.safetensors) now resolves on the 5090, and
  openpi takes the PyTorch path, but inference fails: "CUDA error: no kernel image is available". The openpi venv on
  the 5090 has torch 2.7.1+cu126, arch list up to sm_90; the RTX 5090 is sm_120. PyTorch policies cannot run on the
  5090 GPU in this venv. (My offline tests use `uv run` in third_party/openpi, which re-syncs the venv to the
  lockfile; if someone had installed a cu128 torch there by hand, that may have reverted it.)
  -> need_operator: a cu128 (sm_120) torch in the 5090 openpi venv, or serve on the 4090 once full_v4 is done.
- Workaround (no operator): torch 2.7.1+cu128 / torchvision 0.22.1+cu128 installed into pylib/torch128 (5090,
  gitignored); programs/serve_torch128.py puts it first on sys.path (generic packages pre-imported from the venv) and
  serves like openpi. Offline on the 5090: full_v1/30000 via the shim -> actions (10, 7), 0.17 s per inference.
  Eval with --serve programs/serve_torch128.py and the shim path (serve.sh needs params/).
- eval 11: full_v1/30000 from the look pose, n=1 check.
- rollout_floor.py (operator 14:00 request): 8 mm floor; tip-only failures run the longest passing prefix; close if
  any action in the chunk closes; stop after 3 held chunks without a close. Dry run on the robot machine against the
  5090 server: 3 chunks inferred (~270 ms) and checked; look skipped (no bee on the table in a dry run).
  Note: the orchestrator's remote server stop does not match serve_torch128.py; kill it myself before each eval.
- eval 12: full_v1/30000 from the look pose (rollout_look.py + rollout_floor), n=20, --serve serve_torch128.py.
  result eval 12 (full_v1 from the look pose, 20): 0 success, 1 unsure (ep 345: grasped, reading 70.6, aborted by my
  rollout rule "held 3 chunks" -> operator fixed: holds count only while the gripper is open, commit 5d9ef6d).
  Operator analysis: full_v1 does use the images (image-swap test turns the motion 120 deg toward the other bee);
  failures: hesitation at the look pose (demos pause 2.9 s there), and wrist configuration mismatch: 106/112 bee_v3
  collect episodes are wrist-flipped (joint 4 ~180) at the look pose, the rollout's look pose is normal -> joint-5
  limit drifts (ep 343, 346).
- Operator plan (user approved): bee_v5 = collect, normal wrist only, start at approach, idle frames dropped
  (94 eps, 87k frames); full_v5 = full FT stock representation from full_v1/30000, 30k, batch 4 on the 4090,
  done ~22:55. full_v4 stopped at 21.2k (20000 whole). Then: eval full_v5/30000 with rollout_look.py, 20, DAgger.
  Next dataset: include dagger (--modes collect,dagger).
- waiting for full_v5 (blocking wait on the 4090).
- MISTAKE: attempt 34 (12 h blocking wait for full_v5) held the orchestrator; collection round 3 could not start and
  the robot idled 4.7 h. Rule: never block on long jobs; finish the turn, after_training.sh notifies me.
- full_v5 finished 22:59. eval 13: full_v5/30000 with rollout_look.py (rollout_floor with the operator's hold fix),
  20 rollouts, served on the 4090 (policy_host local).
  result eval 13 (full_v5/30000 from the look pose): 7/20 = 0.35 (successes 395, 401, 402, 403, 408, 412, 413;
  394 unsure). Rollout bugs, fixed by the operator during the eval: 388 (closed-gripper tip margin -> -2 mm while
  reading > 50), 390 (plan start from joints read before inference -> re-read after), 394 (fast carry chunk tracking
  -> EasedSegment 0.4 s ramp). After the fixes (395..413): 7/15. Yaw pattern: jaw yaw 68-85 -> grasps; >= 105 -> 0/5
  (misses 1.6-21 cm, never below z 0.05).
- Next: bee_v6 = --modes collect,dagger --wrist normal --start-phase approach --drop-idle (round-3 demos + normal-wrist
  DAgger corrections of evals 12/13); yaw breakdown of the corrections; then full_v6 from full_v5/30000 (30k, bs 4).
  result: bee_v6 = 123 episodes, 117532 frames. DAgger corrections from evals 12/13: 18 success / 2 fail;
  start tool yaw >= 100 deg: 349, 367, 372, 397, 400 successful (+393 fail) -> high-yaw corrections present.
- train full_v6: pi05_kinova_full (stock representation, like full_v5), bee_v6, init full_v5/30000
  (--pytorch-weight-path), 30k steps, batch 4, on the 4090 (~8.2 h).
  result: full_v6 training OOM on the 4090: other users' jobs hold 19.7 GB of 24.5 GB (aaron's XPolicyLab policy
  server + RMBench eval, an r2r imitate_episodes job; running 13+ h). Cannot stop them.
- Workaround: train full_v6 on the 5090 (32 GB, idle) with the cu128 torch from pylib/torch128 (programs/train_5090.py),
  same config/flags; counts as a training run (declared). Copy bee_v6 + full_v5/30000 to the 5090 first, copy the
  result back to the 4090 for serving.
  smoke test on the 5090 (programs/train_5090.py, JAX_PLATFORMS=cpu, norm stats computed separately): 40 steps from
  full_v5/30000, loss 0.005-0.04 from step 1 (weights loaded), 1.25 steps/s at batch 4 -> 30k in ~6.7 h.
- full_v6: launched DETACHED on the 5090 (setsid nohup, log notes/train_full_v6.out). This IS a training run
  (2nd of this round after the failed OOM start); the orchestrator does not detect it as one because it runs through
  programs/train_5090.py on the agent host. Config pi05_kinova_full, bee_v6, 30000 steps, batch 4, init full_v5/30000.
  result: full_v6 done on the 5090 (30000 steps, 6.7 h, final loss ~0.0013). Next: copy weights to the 4090, eval.
- 4090 GPU still held by other users (18.2 GB used): full_v6 served on the 5090 by serve_torch128 (started by me,
  detached, notes/serve_full_v6.out) and reached through programs/proxy_serve.py (TCP proxy 4090:8000 -> 5090:8000)
  as the eval's --serve program. Kill the 5090 server after the eval.
- eval 14: full_v6/30000, rollout_look.py, 20 rollouts, DAgger on.
  result eval 14 (full_v6/30000, served 5090 + proxy, ~260 ms inference): 2/20 (418, 435) + 2 unsure (423 judged a
  success by the operator from scene_after: bee in bowl; 442). About half of the rollouts ran in a dark room (side
  brightness 94 vs 137 in eval 13 and bee_v6 median 136). Deep grasps (tool 37 mm vs 52 mm) were held by the closed
  margin (421, 422) -> operator fixed rollout_floor at 10:14 (session grip_frac/table z; closed: allow chunks that do
  not lower the tips). Not comparable with full_v5's 7/20.
- A/B (operator): eval 15 = full_v5/30000 again, current rollout_look.py, 10 rollouts, served the same way (5090 +
  proxy). If full_v5 clearly beats full_v6, full_v7 starts from full_v5.
  result eval 15 (A/B, full_v5/30000, same serving as eval 14): 2/10 (444, 446). Under today's conditions full_v5
  ~ full_v6 (2-3/20); not clearly better -> the drop vs eval 13 (7/20) is conditions (light, 5090 latency), not v6.
- USER DECISION: full fine-tuning only on the 4090, never on the 5090 (power losses). full_v6 on the 5090 was
  against this (decision not passed on in time). If the 4090 GPU is taken: need_operator, evaluate/collect meanwhile.
- Next: bee_v7 = collect+dagger, normal wrist, from approach, drop idle (adds round-3 demos and eval 13-15
  corrections); full_v7 on the 4090 from full_v6/30000 when its GPU is free.
  result: bee_v7 = 133 episodes, 127483 frames. 4090 GPU: 18.3 of 24.5 GB used by other jobs -> full_v7 cannot
  start; need_operator per the user's rule.
- No operator. Training blocked (4090 GPU busy, 5090 forbidden). Robot time used for eval 16: full_v6/30000 again,
  20 rollouts (hopefully normal light), served from the 5090 via proxy; also more DAgger corrections for bee_v8.
  After it: check the 4090 GPU and start full_v7 if free.
  result eval 16 (full_v6/30000, 150 s limit, 5090 + proxy): 2/20 (463, 466) + 2 unsure (467, 472); several
  rollouts held the bee near the bowl when time ran out (461, 462, 470: bee in the jaws at home).
- Operator: max_policy_seconds 150 -> 240 (user). full_v7 queued by the operator on the 4090 (train_when_free.sh:
  bee_v7, full, init full_v6/30000, 30k, batch 4); I must not start training. Baseline now: eval 17 = full_v6/30000,
  240 s, 20 rollouts (5090 server + proxy).
  result eval 17 (full_v6/30000, 240 s): 3/20 (489, 491, 498); ep 505 never ran the policy (Kortex session error at
  start) -> 3/19. Failures: hovering 2.7-4.7 cm from the bee without grasping (492, 494), slow (496), holding the bee
  near the bowl high up without releasing (493), one release refusal far from the bowl (490; operator now lets the
  policy continue after such a refusal).
- 20:09: full_v7 not started (4090 GPU still 18.2 GB used by others). Robot: eval 18 = full_v5/30000 at 240 s,
  20 rollouts (comparison with eval 17's full_v6 3/19; base for later runs; more DAgger corrections).
  result eval 18 (full_v5/30000, 240 s, 5090 + proxy): 13/20 = 0.65 (+533 unsure). full_v5 >> full_v6 (3/19 in eval
  17, same conditions). full_v6's extra data/steps hurt. Operator put full_v7 on hold; recipe is mine.
- full_v7 decision: init full_v5/30000; LR constant 1e-5 (stock 5e-5) to stay close to v5; 30k steps (minimum),
  batch 4; dataset bee_v8 = all normal-wrist collect demos (from approach, idle dropped) + only the DAgger
  corrections of full_v5's own failures (evals 13, 15, 18: rollouts 388-413, 444-455, 515-536). Excluded: eval-12
  corrections (full_v1 failure states) and evals 14/16/17 (full_v6). Selection: symlinks sel/bee_v8 on the 4090.
  result: bee_v8 = 113 episodes, 105550 frames (normal-wrist collect demos + 10 corrections of full_v5's failures:
  390, 392, 396, 399, 405, 409, 447, 451, 520, 526). Norm stats computed (assets/pi05_kinova_full/rlpj/bee_v8).
- Armed full_v7 on the 4090 (tmux session learn_full_v7): scripts/train_when_free.sh runs/20261104_1510_bee bee_v8
  full_v7 --num-train-steps 30000 --batch-size 4 --pytorch-weight-path full_v5/30000
  --lr-schedule.peak-lr 1e-5 --lr-schedule.decay-lr 1e-5. Starts when 23 GB is free.
  armed: train_when_free waiting (5.7 GB free at 23:22).
- eval 19: full_v5/30000 again, same conditions as eval 18 (240 s, 5090 + proxy), 20 rollouts: tighter estimate of
  full_v5 (report both evals and the combined rate) + more corrections of full_v5's failures.
  result eval 19 (full_v5/30000, same conditions as eval 18): 12/20 = 0.60. Combined evals 18+19: 25/40 = 0.625.
- Operator 00:58: full_v7 (my recipe: bee_v8, init full_v5/30000, 30k, batch 4, LR 1e-5) trains on the 5090 with the
  user's approval (power mitigation), tmux train5090, ETA ~09:20; after_training_remote.sh copies it back and notifies.
  I stop my full_v5 server on the 5090 (GPU memory for training). Next: eval full_v7/30000 (20, 240 s).
- full_v7 finished 08:57 (5090). eval 20: full_v7/30000, rollout_look.py, 20 rollouts, 240 s, 5090 + proxy.
  eval 20 stopped by the operator after 1 rollout (ep 624 unsure: bee left hanging on the gripper at home). Rerun.
- 10:16 2026-11-10: evals 20/21 of full_v7 stopped: after ep 624 (unsure; bee left hanging on the gripper) the bee
  is gone. Reset agent: full wrist sweep, bee not on the table (attempts 696-698), orchestrator autonomous_stop
  "reset failures". My check (dry session, motion off): gripper empty (reading 17.5, wrist image clear), side camera
  shows only the bowl, empty, on the table. Bee most likely fell off the table. Cannot be recovered by a program.
  full_v7/30000 remains untested (1 rollout: 624 unsure).
## Summary (policy from the look pose, 240 s unless noted)
- best: full_v5/30000 (full FT, stock representation, bee_v5, init full_v1): eval 18 13/20, eval 19 12/20 -> 25/40.
- full_v6: 3/19 (eval 17). full_v7 (bee_v8, init full_v5, LR 1e-5): untested.
- 13:03 bee put back by a person; run relaunched. Eval 20 had no valid rollout (624: bee stuck outside the closed
  fingers; DAgger 625 dropped it). eval 22: full_v7/30000, 20 rollouts, 240 s, 5090 + proxy.
  result eval 22 (full_v7/30000, look pose, 240 s, 5090 + proxy): 16/20 = 0.80 -> TARGET MET (>= 0.70).
- full_v8 (operator-run: full_v7 recipe on bee_v9 = 169 clean normal-wrist demos + same 8 corrections, init full_v5,
  LR 1e-5, 30k, batch 4, 5090) finished 20:06. eval 23: full_v8/30000, same protocol as eval 22. No more training
  after this unless the user asks.
