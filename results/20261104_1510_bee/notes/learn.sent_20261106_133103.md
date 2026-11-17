Evaluation 03 (lora_v1/14999, 12 rollouts before the bee was lost) was scored with a rule that could never give a
policy rollout a success. The rollout always claims "fail" and the verdict needed a "success" claim. This is fixed:
for policy rollouts, success now needs both independent checks (geometry and VLM) to say yes, one yes is "unsure",
and no yes is "fail".
Re-reading the 12 rollouts with the operator's eyes:
- 117: the bee ended in the bowl (scene_after.jpg). The geometry check said yes; the VLM said no at confidence 0.5,
  from the wrist view. That is a real success, "unsure" under the new rule.
- 119: the policy grasped the bee by its tip and went home still holding it. The geometry check gave a false yes;
  the VLM was right. Fail.
- The rest are fails: mostly near-miss grasps with the jaws closing beside the bee.
So the policy is at about 1 in 12. DAgger is on from the next evaluation: each failed rollout gets an expert
correction from the policy's last pose. The bee is now wrapped in orange (since 2026-11-05, about from episode 59 on),
so roughly half of bee_v1 shows the yellow bee and half the orange one.
