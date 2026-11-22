Operator brief (new experiment 20261122_1732_drawer, task drawer_ng; the user wants a fresh, fully recorded run):
Your programs/ hold the previous run's reset code (pinch-and-pull of the knob, wrist look-downs, escapes, tipped-drawer
handling, low-force approaches). Adapt them to this task; do not start from zero.
New in this experiment (near-goal, reset-free RL; the user's design): the reset NEVER leaves the drawer on the table.
targets.json holds {"drawer_out": d} (0.03-0.05 m): leave the drawer partly in the bottom slot, right way round, roughly
straight, front d in front of the cabinet front (+-2 cm), gripper clear, arm home. After an RL success it is flush:
pinch the tab, pull straight out to d. From any other state (recessed, cocked, jammed, held, out of the slot) bring it
back to this state. The orchestrator checks it with the side camera (front d +-2 cm out) and the vision check.
Lessons from the previous run (keep them):
- Use the WRIST camera for positions; the side camera reads table objects ~5-6 cm low in y. The slot geometry in the
  yaml (slot centre t -0.036) was corrected from wrist views; tell me if your knob measurement disagrees.
- Low force (self.low_force(20-25)) for every approach into an unknown contact; 65 N is the cap, only for pulls.
- Open the gripper and back off along the tool axis before rising from a low pose; refined grasps need closed-loop
  checks; stop and report early instead of looping.
