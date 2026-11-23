# Operator log: run 20261123_1616_drawer (task drawer_lying)

New experiment (user, 2026-11-23): the cabinet lies on its back; the reset takes the drawer out and stands it on its
back wall beside the cabinet (pull up); the RL agent inserts it into the compartment nearest the arm, pull up, flush.
Before the run: side camera recalibrated automatically (11 views, spread 9.7 mm; camera had moved 131 mm / 10 deg),
cabinet measured (arm pulled back + wrist look-down), geometry, keep-out and success check written; the check passes
on the hand-inserted drawer (flush 1.0). Agents fresh; previous run's programs in programs/previous_standing_cabinet/.
- 2026-11-23 16:22: launched (LAUNCH.sh in tmux rlpj_drawer:pipeline); check passed; develop_reset started. Before launch the arm was returned to Home by scripts/go_home.py (a joint move was refused for leaving the cell; straight tool line + joint adjust). Background: web :8091, media mover, plots every 5 min (plots/), cabinet guard (auto reference), GitHub sync.
- 2026-11-23 16:33: sim worker started for the lying cabinet (vertical insertion; trains on the free 4090 GPU, 5090 fallback). Note to the reset agent: keep the first reset simple and get it on the robot soon.
- 2026-11-23 16:41: user warned about the table edge; the reset target (y -0.22) was AT the table's camera-side edge (side cloud: edge at y -0.18..-0.23). Moved to [0.28-0.30, -0.03..-0.01] (>= 14 cm from every edge, ~5 cm clear of the cabinet), commit 96b8c78; reset agent told the edge positions and a 10 cm rule; safe restart to reload the target.
- 2026-11-23 16:42: the agent re-ran the same test with the old edge target (it passes targets itself from its first workspace sample); attempt 9 had picked the drawer and was lowering it at [0.494, -0.242] (off the edge) when I stopped it with SIGINT; it returned home (drawer likely still held). Explicit note sent with the safe test region.
- 2026-11-23 16:45: the user moved the drawer inward by hand (away from the table edge) at my suggestion.
- 2026-11-23 17:58: RL started 17:32 (4 episodes by 17:53, all failed: drawer bottom on the rim, alignment off by more than the few-mm clearance, opening detector failing). Note to the RL agent: visual alignment with the drawer held just above the rim, larger search steps, tilt trick.
- 2026-11-23 18:30: since episode 4 (18:00) the RL agent has been testing fixes (attempts 22-26: visual hover, tilt), still resting on the rim. Found: the hover image cannot see the opening (the held drawer hides it); the opening detector fails, so alignment uses one stored measurement that differs from the yaml by 1.5 cm. Note: measure before the grasp, touch/spiral search over +-15 mm, resume episodes.
- 2026-11-23 18:32: sim worker finished (4090): policy 74.8% vs straight 21% / spiral 14% / force-guided script 59.6% in sim; commits 7690f86, e374011. RL agent told (force-direction cue, geometry, API).
