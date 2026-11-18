
FIRST TRAINING EPISODES (02:35), for your tuning turn (now every 10 episodes instead of 40):
- Episode 0 (attempt 23) and 1 (attempt 25) failed; both independent checks said no.
- Episode 1: the sampled actions were far from the mean. The mean action was ~[0.5, -0.1, 0.03, -0.4] but samples
  ran from -0.98 to +0.85 on the advance, so most insertion steps advanced 0-3 mm. After the 16 steps the drawer front
  was still at n-OFF -0.067, "too far out for the tilted push", so the push never ran: the episode could not succeed.
  The exploration noise is too large for a 16-step budget. Consider a smaller or state-dependent std, a minimum
  advance, more insertion steps, or noise only on the correction dimensions (t, z, yaw) while the advance follows
  the prior. The learner_log shows alpha 0.017, entropy 0.90 after 8 episodes.
- The policy mean is still close to the prior, so the learned part has not taken over yet: fine this early.
