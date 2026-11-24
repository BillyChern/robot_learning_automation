# X thread: real-world RL with Claude Code agents (drawer into a lying cabinet)


Counts are X's (arrows, ≥ and emoji count 2). Short versions fit 280; long versions need a Premium account.
Sim credit: SGS-RL (Success Guided Sampling): Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots, "A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control", CoRL 2026.

## 1/4 · The main video

Attach: `00_full_story.mp4`

Short (275/280):

```
Round 2: Claude Code agents (Opus 5.5) vs. a drawer. Real-world RL on our arm, cabinet on its back, 1-2 mm of clearance per side.

They wrote the resets, the episode code, the reward and the learner. First 5 episodes: stuck on the rim. Later 37 in a row, final test 7/10 🗄️
```

Long (489):

```
Round 2: Claude Code agents (Opus 5.5) vs. a drawer. Real-world RL on our arm, cabinet on its back, 1-2 mm of clearance per side.

They wrote the resets, the episode code, the reward and the learner. First 5 episodes: stuck on the rim. Later 37 in a row, final test 7/10 🗄️

A third agent built a sim of the drawer (MuJoCo + NVIDIA Newton) and trained a PPO policy. We put it on the real robot without changing a thing: 5/5.

We stepped in a few times. Every time is marked in the video.
```

## 2/4 · The numbers

Attach: `thread/episodes.png`, `thread/learned_vs_scripted.png`, `thread/sim_results.png`, `thread/return.png` (X: up to 4 images per post)

Short (237/280):

```
Numbers (50 training episodes, one night, 0 manual resets):
- 41/50 in, 37 in a row
- ep ≥17, straight in w/o rim touch: learned 6/17 vs scripted baseline 0/11 (p=0.055)
- final test, deterministic: 7/10
- sim PPO policy, zero-shot: 5/5
```

Long (1372):

```
The numbers. One night (16:16 to 02:42), 50 training episodes, 77 robot resets, none by hand.

Success = two independent checks agree: side-camera depth geometry and a Claude Opus 5.5 judge on before/after images.

- Episodes 0-4: 0/5, drawer on the rim. The agent rebuilt its program (measure the opening before the grasp, then a touch search; both ideas came from a note we sent, the start offset it found itself). Since episode 9: 40/41 counted, 37 of them in a row; 46 put the drawer in but our check said "unsure" (a bug).
- Learned (AWAC on the robot's own episodes) vs a scripted baseline (same program, learned part off, every 3rd episode), episodes ≥17: straight in without touching the rim 6/17 vs 0/11 (Fisher's exact p = 0.055), 16.8 vs 18.4 steps, counted successes 16/17 vs 11/11 (episode 46: in, but scored "unsure" by a bug in our check).
- Final test after 50 training episodes: deterministic policy, 10 episodes: 7/10 (misses: episodes 52, 54, 57, all the same way: jammed tilted 4-6 mm into the opening, no lift-and-retry in the program).
- Sim: PPO with an SGS-style reset curriculum, 4,096 worlds in NVIDIA Newton on one RTX 4090 (about 1.5 h). 500 MuJoCo episodes, same starts: PPO 74.8% vs 21.0% straight down, 14.0% spiral search, 59.6% force-guided script. On the real robot, unchanged: 5/5.

SGS: Zhang et al., "A Balanced Data Diet", CoRL 2026.
```

## 3/4 · Bloopers

Attach: `03_human_help.mp4`

Short (274/280):

```
Bloopers. We put the first reset spot right on the table's edge (our bad): we hit Ctrl+C twice, and I moved the drawer back by hand once. Bugs in our own check scored 3 real insertions "unsure".

Mid-test the drawer tipped over. The reset agent stood it back up by itself 😅
```

Long (619):

```
Bloopers. We put the first reset spot right on the table's edge (our bad): we hit Ctrl+C twice, and I moved the drawer back by hand once. Bugs in our own check scored 3 real insertions "unsure".

Mid-test the drawer tipped over. The reset agent stood it back up by itself 😅

And the fix for the rim? Two ideas from a note we sent. "We" = me plus a Claude Code session watching the run.

All in all: 1 hand on the scene, 2 tests stopped, 18 notes to the agents, 6 restarts to load our changes. The big calls were ours too: a scripted baseline, the freeze, one return formula, the 10-episode test, trying the sim policy.
```

## 4/4 · Takeaway

Attach: nothing (text only)

Short (272/280):

```
Takeaway: agents can take over the chores of real-world RL: resets, rewards, logging, plots, even a sim.

A scripted baseline kept us honest. Most of the gain came from the agent rewriting its program; the learner only cut rim touches. And humans still made the key calls.
```

Long (310):

```
Takeaway: agents can take over the chores of real-world RL: resets, rewards, logging, plots, even a sim.

A scripted baseline kept us honest. Most of the gain came from the agent rewriting its program; the learner only cut rim touches. And humans still made the key calls.

P.S. Opus 5.5 made these videos too.
```

Extra videos (replies, a project page): `01_sim_to_real.mp4`, `02_learning_curve_and_arms.mp4`, `04_final_evaluation.mp4`
