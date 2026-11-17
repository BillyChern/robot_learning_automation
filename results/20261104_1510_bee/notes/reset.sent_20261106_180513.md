Your test run attempt 220 hung for 2 h 45 min. After the log line "bowl centre median of 7" (15:16:37), the process
ran at full CPU and logged nothing, and the arm never moved. Ctrl+C did not stop it: the session's stop flag is
only checked inside session calls, so a pure-Python loop never sees it. The operator killed it (exit 137); the arm
was idle at Home.
Find the loop. It is most likely in the code added for the standing bee and the flip drop, or in a spot or IK search.
Bound every loop: a maximum number of iterations, or a time limit, and s.require(...) on failure. Test the case with
a dry run before the next robot run.
Every robot run now has a 20-minute time limit (collection.attempt_timeout_s), so a hang ends there.
