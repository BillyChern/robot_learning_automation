# Operator log: run 20261123_1616_drawer (task drawer_lying)

New experiment (user, 2026-11-23): the cabinet lies on its back; the reset takes the drawer out and stands it on its
back wall beside the cabinet (pull up); the RL agent inserts it into the compartment nearest the arm, pull up, flush.
Before the run: side camera recalibrated automatically (11 views, spread 9.7 mm; camera had moved 131 mm / 10 deg),
cabinet measured (arm pulled back + wrist look-down), geometry, keep-out and success check written; the check passes
on the hand-inserted drawer (flush 1.0). Agents fresh; previous run's programs in programs/previous_standing_cabinet/.
- 2026-11-23 16:22: launched (LAUNCH.sh in tmux rlpj_drawer:pipeline); check passed; develop_reset started. Before launch the arm was returned to Home by scripts/go_home.py (a joint move was refused for leaving the cell; straight tool line + joint adjust). Background: web :8091, media mover, plots every 5 min (plots/), cabinet guard (auto reference), GitHub sync.
