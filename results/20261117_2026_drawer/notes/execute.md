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
