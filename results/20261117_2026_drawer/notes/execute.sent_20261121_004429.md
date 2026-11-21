Operator note 00:45 (episode 33, attempt 96): the grasp held (87.8), then the new yaw cancel crashed the program:
RuntimeError "line cancel_yaw: joint jump 3.7 deg between samples" (an IK branch flip in the line), so the session
went home holding the drawer. Wrap the cancel move: catch the joint-jump/model refusal and retry with the rotation
split into smaller steps or as a short joint move (model-checked) at a safe height, else skip the cancel and log it.
No exception may end an episode with the drawer still in the hand.
