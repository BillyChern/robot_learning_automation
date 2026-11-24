Operator note 19:10 (user decision): add a SCRIPTED BASELINE arm so the curves show what RL adds. In every training
episode with episode % 3 == 2 (8, 11, 14, ...), run the same current program with the learned actor OFF: zero
residuals (rlcommon.prior), no exploration noise; everything else identical (grasp, measured alignment, touch search,
start offset). Write s.summary(arm="scripted") there and arm="learned" in all other episodes (eval episodes:
arm="learned_eval"). Keep logging the transitions of scripted episodes (valid off-policy data for AWAC). Do not
change the scripted part between paired episodes more than needed; when you do change it, note the commit in the
summary (both arms then use it). Fixed eval rounds stay as they are.
