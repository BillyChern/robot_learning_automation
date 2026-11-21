Operator note 12:40 (user request: "construct the sim in parallel, RL in sim, SGS-RL, sim2real"): a separate worker is
now building a physics sim of the INSERTION only (cabinet slot + drawer held with in-hand yaw/offset slip, domain
randomized clearance/friction/slot pose; NVIDIA Newton if it installs, else MuJoCo) and training a PPO policy with
SGS-style reset-to-states on the 4090. Output: sim/drawer_insert/ with a numpy-only Policy(path).act(obs) -> (dn, dt,
dyaw) small tool steps in the slot frame at ~5-10 Hz, obs = tool pose in the slot frame, tool force estimate, previous
action, the in-hand offset/yaw estimate, step count; README with the exact call. Do not build your own sim. Please:
1. Keep improving the real pipeline as now (wrist-footprint grasp, drawer-centre-on-slot alignment, jam rule).
2. Prepare a switch in your insert phase so a step policy can drive it: a loop that reads the tool pose in the slot
   frame and s.feedback() force, calls a policy for (dn, dt, dyaw), runs a short model-checked line per step under the
   65 N cap (low_force for the approach), and stops on success/jam/step limit. Test it with a trivial scripted policy
   (straight push) offline/dry. When sim/drawer_insert/policy.py appears you will get a note to try it, mixed with
   your current prior (e.g. alternate episodes) so both are compared on the robot.
USER SUGGESTION (verbatim, 12:55): "also for the RL agent is the current strategy that it insert into the drawer and
and and move / drag it to adjust its pose to be right in front of the cabinet slot and then push in from the front
into the slot stuff like these".

Please add this as a new strategy (keep the current one as the fallback, and let the episodes compare them):
1. Wrist look-down for the drawer pose on the table (your footprint code).
2. ALIGN ON THE TABLE, no grasp: closed fingers straight down into the drawer's open top (to ~2-3 cm above its floor,
   low_force(20)), then small drags/turns (2-4 cm, 3-5 deg per step, low_force(30)) that bring the drawer right in
   front of the slot: pull wall facing the robot, axis along N, centre on the slot centre t, back wall ~1-2 cm from
   the cabinet front. Lift out, re-measure with a wrist look-down after each drag; repeat until |t err| <= 5 mm and
   |yaw| <= 2 deg. Measured free on the table, the pose has none of the in-hand uncertainty that jams the insertion now.
3. INSERT STRAIGHT: grasp the pull wall at its centre with the jaws square to it (the pose is now known), lift only
   ~2.5 cm (the slot floor is ~1.5 cm above the table: the bottom lip), move straight along N until the back half is
   in or the force rises (jam rule), lower, release, then push the pull wall in to flush with closed fingers.
   Alternative for step 3 without any grasp: lift the drawer's back end over the lip by pushing it in from the open
   top at a slight upward angle, then push the pull wall from the front.
Log which strategy each episode used, the alignment error before the insertion, and the result.
