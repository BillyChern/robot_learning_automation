Operator note 20:33 (user: the return curve must show the learning progress). Since episode 6 your return is ~14
in every success (the success bonus dominates), so learned and scripted look the same: both make 2 rim contacts,
25 N, ~18 steps (episode 9 had 0 contacts, 13 N, 15 steps: that is the room to learn; the frozen start offset makes
the scripted arm land on the rim twice every time). From now on use THIS reward (the report plots it for all
episodes, scripts/rl_plots.py fixed_return): +10 if the drawer ends seated flush, -1 per rim contact, -0.1 per N of
peak force above 15 N, -0.05 per insertion step (spread per step as you like, the episode sum must match). Relabel
all stored transitions with it before the next update, and make sure the actor's residual acts on the first
descent point (the start offset), so the learner can find the 0-contact entry. Do not change the scripted arm.
