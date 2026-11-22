# Operator log: run 20261122_1732_drawer (task drawer_ng)

A new, fully recorded experiment (user, 2026-11-22): near-goal, reset-free RL. Every reset leaves the drawer partly in
the bottom slot (front 3-5 cm out, right way round); after a failed episode the next one starts where it ended; a reset
only after a success, a start state outside 0.5-9 cm out, or 3 failures in a row. Fresh agent sessions; their
workspaces start with the previous run's (20261117_2026_drawer) reset and RL programs, as the user chose. Robot library
fixes, the corrected cabinet geometry (slot centre t -0.036) and sim/drawer_insert are carried over. Every
intervention below.

- 2026-11-22 17:32: run created; programs seeded; operator briefs written (notes/). Arm offline since 11-21 15:24
  (eno1 no carrier); launch waits for the user.
