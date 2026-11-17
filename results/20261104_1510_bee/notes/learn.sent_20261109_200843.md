The orchestrator's wrist check after learn run attempt (arm at home) shows the bee held in the closed jaws (reading 75.4): object colour 0.14 of the bottom 15% of the wrist image (threshold 0.1; image /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/episodes/ep_000493/attempt/gripper_check.jpg). Your program released it but it did not come off. Check the fingers after every release before moving away, and shake it off over the target if it is still there.
Operator (18:14): ep 490 (eval 17) ended at 65 s on the session's "release refused" (it opened while holding
the bee 22 cm from the bowl). The safeguard is right to refuse, but the rollout does not need to end: rollout_floor
now catches that one refusal (raised before anything moves), keeps the gripper closed, counts it in summary
release_refused, and lets the policy continue. It was 1 of about 60 rollouts since ep 388; the other eval 17 failures
are policy behaviour: no grasp after hovering 2.7-4.7 cm away (492, 494), slow (496 grasped at 94 s), and holding the
bee 3 cm from the bowl at z up to 0.43 without releasing until 240 s (493).
Operator (19:17): eval 17's ep 505 never ran the policy. The arm connection failed at start (Kortex
INVALID_USER_SESSION_ACCESS), so leave it out of the success rate (it is a start failure, not a policy failure). Fixed
in the robot library: longer connect retry, and the session waits for the watchdog before moving. The same error had
also stopped the watchdog from starting in 11 runs.
