# Zero-shot test of full_v7 on new objects (2026-11-14, a lab member at the robot)

Policy: `runs/20261104_1510_bee/checkpoints/full_v7/30000` (fine-tuned only on the toy bee, one instruction,
"pick up the toy bee and put it into the bowl"; 80 % on the bee in eval 22). Rollout: scripted look move above the bowl
(side depth; bowl only, object optional), then the policy for up to 240 s. Objects placed by hand; success judged by the
person, the VLM judge agreed on every rollout.

Tests: `runs/zeroshot_blind` (prompt "pick up the toy and put it into the bowl") and `runs/zeroshot_named`
(prompt "take the black adapter and put it in the bowl").

## Valid rollouts (the policy ran)

| test | rollout | object | result | closest approach (xy) | lowest tool z | gripper |
|---|---|---|---|---|---|---|
| blind | r02 | black adapter | fail | 3.7 cm | 0.125 m | never closed |
| blind | r03 | black adapter | fail | 1.5 cm | 0.069 m | never closed |
| named | r01 | black adapter | fail (pushed the bowl) | 0.7 cm | 0.075 m | never closed |
| named | r07 | orange-black block | fail | 1.4 cm | 0.072 m | closed in the air |
| named | r03 | blue ball | fail | (not seen in depth) | 0.042 m | **grasped the ball**, never carried it to the bowl |
| named | r02 | bee | **success** | 3.7 cm | 0.027 m | grasped, placed in the bowl |
| named | r04 | bee | fail | 12.9 cm | 0.123 m | never closed |
| named | r05 | bee | fail | 2.2 cm | 0.085 m | closed above the bee |

Grasp height for the bee is about 0.036 m (tool z). Novel objects: 0/5. Bee: 1/3.

Not valid (the policy never ran): blind r01 and named r06 (bowl outside the look move's area), blind r04-r06 (the
arm was in fault after being stopped between rollouts), blind r07 (interrupted, no result). Earlier aborted attempts
are in `runs/zeroshot_blind/invalid_policy_never_ran/`.

## Conclusions

1. Perception was not the cause in the valid rollouts: the scripted look move found the bowl and handed over each
   time. The invalid ones were set-up problems (bowl placed outside the area the bee runs used; an arm fault).
2. The policy finds new objects: from above the bowl it moved to within 0.7-3.7 cm (horizontally) of the adapter and
   the block, using the camera images only.
3. It does not manipulate them: it stops 7-13 cm above small flat objects, never descends to grasp height, and closes
   in the air or not at all. The ball is the exception (grasped, not carried to the bowl).
4. The instruction is ignored: with "take the black adapter..." it picked up the bee and put it in the bowl (named
   r02). Fine-tuning on a single sentence removed the language conditioning; naming the object cannot help.
5. The bee result does not transfer to other layouts: 1/3 here, with failures where the bowl was on the other side of
   the table (y +0.07; training -0.17..+0.04) or the bee was far from the usual area. The 80 % holds for the layouts the
   resets produced.

## What it would take

- Several objects in training (with the instruction naming each), so grasping and language both generalize.
- Layout variety in training (bowl on both sides, objects farther out).
- A fixed protocol for any further test: layouts inside the training range, 10 rollouts per object, and bee control
  rollouts in the same session.
