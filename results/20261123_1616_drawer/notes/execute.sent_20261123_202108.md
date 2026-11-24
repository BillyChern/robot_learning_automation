Operator note 20:27 (user decision: extend the experiment for the report):
1. Training now runs to 50 episodes, then ONE fixed evaluation of 10 deterministic episodes (config changed).
2. Keep the scripted baseline (episode % 3 == 2) and the freeze at r4.
3. NEW third arm "sim" (sim-to-real, zero-shot): episodes 31, 34, 37, 40, 43 run the SIM policy unchanged
   (repo sim/drawer_insert, README "Lying cabinet": Descent(Policy(".../weights/policy_lying.npz"), ...), frames, signs
   and logging as documented there) in place of your actor, with the same grasp, measured alignment and carry; no
   touch search of yours (the sim policy does its own search); keep your force stop at 25-40 N; tag arm="sim" and log
   what the README asks per decision. Dry-run it once before episode 31 (a risky change: one real test is allowed if
   the dry run cannot show the frames are right). If it misbehaves (pushing away from the opening, > 40 N twice),
   end that episode cleanly and keep the record: a failure is a valid result.
4. Do not train your actor on the sim-arm transitions with the same action meaning unless they match your layout;
   keep them in a separate folder if not.
