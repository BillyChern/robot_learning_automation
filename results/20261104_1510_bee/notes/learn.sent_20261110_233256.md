
full_v8 finished training on Worksation_5090 at 2026-11-10 20:14:19: final checkpoint /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v8/30000 (weights also copied to the 4090).
Evaluate it now: 20 rollouts from the look pose (programs/rollout_look.py), DAgger on, 240 s, served as for evals
18/19 (5090 through your proxy unless the 4090's GPU is free). Compare with full_v5 (eval 18: 13/20, eval 19).
Operator (23:32): the user asks for full_v8's final success rate. full_v8 = full_v7's recipe on bee_v9 (177
episodes: 169 clean normal-wrist demos + the same 8 DAgger corrections), init full_v5/30000, constant LR 1e-5, 30k steps,
batch 4, full fine-tune on the 5090 (finished 20:06). Evaluate full_v8/30000 exactly as eval 22 (full_v7: 16/20 = 80%): 20
rollouts from the look pose (programs/rollout_look.py), 240 s, DAgger on, served from the 5090 through your proxy
(stop full_v7's server there first; full_v8's weights are on both machines). Report both rates. The 70 % target is
already met by full_v7, so do not start another training run after this evaluation unless the user asks.
