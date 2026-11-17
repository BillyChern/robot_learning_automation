NEW: DAgger. Your program is now the expert that corrects the trained pi0.5 policy. After a policy rollout fails
in an evaluation, the orchestrator runs your program with RLPJ_DAGGER=1 and RLPJ_START_JOINTS=[q1..q6 deg].
Before your code runs, the session moves the arm back to the policy's last pose, retracing the rollout's own
return path, so the scene is unchanged. Recording starts there, and everything you do from then on becomes training
data that teaches the policy how to recover.
When RLPJ_DAGGER=1, please:
1. Do not assume you start at home. The tool may be low, beside or over the bee or the bowl.
2. Check the gripper first. If it is closed on nothing (reading >= 97.5), open it. The orchestrator skips cases where
   the policy ended holding something.
3. Lift straight up, slowly, to your usual hover height before anything else. Then perceive with the side camera as
   usual (the arm is then high, like at home) and do the normal grasp, carry and release over the bowl, with your
   release check.
4. Keep the recorded motion direct and smooth: no detours, no long pauses, no return home before the task is done.
   The policy will imitate it.
5. Claim from your own verification as usual.
Test it offline with a dry run that sets RLPJ_START_JOINTS to the last pose of evaluation episode 108:
[25.7, 47.3, 288.0, 178.1, 58.3, 46.9], gripper 99.1, closed on nothing.
Without RLPJ_DAGGER, collection episodes behave exactly as before.

The bee is now wrapped in ORANGE (since 2026-11-05). Your release check's yellow range (hue 15-40) barely sees it.
Use hue 0-40, S > 120, V > 110 (OpenCV HSV). Measured in the bottom 15 % of the wrist image at home: bee hanging 0.27,
empty <= 0.006; bee lying below the gripper after a release <= 0.05.
