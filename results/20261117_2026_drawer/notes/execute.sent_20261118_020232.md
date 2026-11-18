
REWARD FIX (user, 01:50): the reward must consider the drawer's ORIENTATION and agree with the real verdict.
Problems in execute.py `finish()` now:
1. The terminal bonus comes only from `checks.drawer_rl` (side-depth geometry). That check cannot tell which way round
   the drawer is: a drawer pushed in flush but BACKWARDS (plain wall out, pull inside) gets +2..+5 + 2*flush from
   you, while the orchestrator's verdict (geometry AND vision judge, whose success text says backwards = failure)
   says fail. Your reward then teaches the wrong thing.
2. The +5 needs `orient_state != "unknown"`; with the turning step removed, a real success only gets +2.
3. F_SOFT = 20 N penalises normal sliding (22-28 N measured): at 28 N a 12 mm insertion step nets -0.06.
Do this:
- Orientation in the reward: success counts only if the pull is outside (facing the robot). In finish(), judge the
  after images with the vision judge the orchestrator uses (from rlpj.success import vlm_judge;
  vlm_judge.judge({"before_side": ..., "after_side": ..., "after_wrist": ...}, TASK) with the task config, whose
  success text includes the orientation), and give a clear negative terminal reward (e.g. -5) for a drawer inserted
  backwards, so the policy is never paid for it.
- Better still, use the orchestrator's own verdict: from now on, after every RL episode it writes
  $RLPJ_RL_DIR/verdicts/<episode>.json {episode, mode, final ("success"/"fail"/...), reason, verdict{...}} before
  your learner runs. Relabel the terminal reward in learner.py from that file (final == "success" -> bonus; a
  backwards or wrong result -> penalty), keeping your own shaping terms for progress and flushness.
- Drop the orient_state condition on the bonus.
- Move F_SOFT above normal sliding (e.g. 35 N) or penalise only force without progress.
Say what you changed in notes/design.md.
