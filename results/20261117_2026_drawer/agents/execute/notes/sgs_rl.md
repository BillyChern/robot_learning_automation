# SGS-RL (Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots: "A Balanced Data Diet: Addressing the
# Exploration Bottleneck in Mega-Scale RL for Robot Control", arXiv 2610.12465; reported for CoRL 2026)
Read 2026-11-18 (arXiv PDF not text-extractable here; method details from the emergentmind summary of the paper).

What it is: Success Guided Sampling, an adaptive reset sampler on top of unchanged PPO. A fixed, finite set of task
configurations / reset states (for manipulation: reaching, stable-grasp and near-goal states, 32k of them); per
configuration the success rate p_i over its last H episodes; weight w_i = (p_i+eps)^(k t) (1-p_i+eps)^(k(1-t))
(Beta-shaped, peak at the target success t = 0.5 for manipulation), softmax with temperature -> sampling
probabilities. Experience goes to configurations of intermediate difficulty, so one simple shared reward suffices.
No demonstrations. Scale: up to 2^20 parallel simulated environments; manipulation teacher (state + point cloud)
distilled into an RGB student with DAgger; zero-shot sim-to-real (UR5e, NIST board assembly); no real-world RL.

What I take for drawer_rl:
- The core lesson: with a simple reward, exploration is solved by WHERE episodes start (near-goal resets, a
  curriculum by success rate), not by reward shaping. My learned part already starts "near goal" (scripted grasp
  and carry to the slot front), and the terminal reward is the orchestrator's own check.
- Not adopted now: SGS needs (a) thousands of resettable configurations and (b) millions of episodes; real-world
  episodes cost ~3 min each with the reset owned by the reset agent (I cannot choose reset states), and a faithful
  contact sim of a wooden drawer in a slot (unknown clearances, friction, the 1 cm side-camera/wrist offset of the
  front plane) would be its own project; the 5090 here also has no kernels in the installed torch.
- Possible later adoption (tuning turn, if the curve stalls): an internal SGS-style curriculum over the learned
  stage's start configuration (e.g. insertion start depth / push start offset as a small set of configurations,
  sampled with the Beta-kernel weight on their recent success), keeping the reward simple (check +5, small
  step/force costs). Logged here so the choice is explicit.
