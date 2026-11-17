The foam bee sometimes stays on the gripper after the jaws open (the reset program saw it wedged between the fingers
and hanging below the fingertips; once it was carried away and fell off the table to the floor). Your program
releases the bee over the bowl but never checks that it came off. Please add a release check before the arm leaves
the bowl:
- wrist camera: share of bee-yellow pixels (OpenCV HSV 15-40, S > 100, V > 90) in the bottom 15 % of the wrist image,
  full width. Measured: bee stuck 0.21-0.22, bee lying below the gripper at most 0.052, empty at most 0.027;
  use 0.10 as the threshold.
- squeeze test: close the open gripper once; closing on nothing reads about 99.1, on the bee 78-91 (use < 97.5).
If either says the bee is still on the gripper: lower back over the bowl, open/close the jaws, move a little sideways,
rise slowly, and check again (up to 3 times); if it stays, stop cleanly over the table rather than carrying it home.
The orchestrator now also checks the wrist image after every robot program and stops the run if the bee stays on the
gripper twice in a row.
