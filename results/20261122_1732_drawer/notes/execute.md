Operator brief (new experiment 20261122_1732_drawer, task drawer_ng; the user wants a fresh, fully recorded run):
Your programs/ hold the previous run's code (wrist footprint, table alignment, held insertion with the sim-policy
hook sim/drawer_insert v2, push with jam rules, reactive retreat, learner.py). Adapt it; do not start from zero.
New in this experiment (near-goal, reset-free RL; the user's design): every episode starts with the drawer PARTLY IN
THE BOTTOM SLOT, right way round. RLPJ_START=reset: fresh, front ~3-5 cm out, straight. RLPJ_START=continue: the end
state of your last failed episode (cocked, jammed, 0.5-9 cm out); RLPJ_START_OUT = the measured distance out (m).
After a success the reset pulls it out again; after 3 failures in a row it resets. Your job: get it flush (within
1 cm) and release cleanly without dragging it out; recover from whatever your last attempt left (straighten: pinch
the tab, pull back 1-2 cm, push again). No table pick, no carry for now; later the start distance grows toward the
full task. Design the learner for this (state: drawer out-distance and yaw from perception, force; reward from depth
progress + success), log per-episode depth/yaw/forces.
Lessons from the previous run: the side camera reads ~5-6 cm low in y (use wrist views for positions); the slot centre
is t -0.036 (corrected 11-21 15:30 from wrist views); pushes at 65 N tilted the cabinet: 40 N while cocked or not
entering, 65 N only when straight and moving; the retreat must not drag a flush drawer out (open fully, go up first).
