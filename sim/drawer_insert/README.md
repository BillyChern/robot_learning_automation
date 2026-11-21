# Drawer insertion sim + RL policy (sim-to-real for the Kinova drawer task)

A held wooden drawer is pushed back-wall-first into the cabinet's bottom slot. The sim models only the insertion:
the hand holds the drawer by its front wall (compliant grasp) and moves in small Cartesian steps in the slot frame;
the cabinet slot has side walls, floor, ceiling rail and a back stop. A PPO policy trained with reset-to-states
(SGS-RL-style curriculum) outputs one bounded step per decision from signals the real robot has.

Files: `params.py` (measured values + randomization), `scene.py` (MJCF), `env.py` (MuJoCo CPU env), `vec.py`
(worker pool), `curriculum.py` (reset-to-states sampler), `train.py` (PPO), `policy.py` (numpy policy, robot side),
`baselines.py` (scripted straight / wiggle), `evaluate.py`, `render.py`, `newton_env.py` (Newton GPU twin),
`weights/policy.npz` (trained actor), `results/` (eval json, image strip), `tests/`.

RESULTS_PLACEHOLDER

## How to run

Environment (venv on /data/sonic, `.venv` is a symlink; /home is nearly full):
```
uv venv /data/sonic/venvs/drawer_insert --python 3.12 && ln -s /data/sonic/venvs/drawer_insert sim/drawer_insert/.venv
uv pip install --python sim/drawer_insert/.venv/bin/python "newton[sim]" torch mujoco pillow pytest
```
From the repo root (`PY=sim/drawer_insert/.venv/bin/python`, `export PYTHONPATH=. OMP_NUM_THREADS=1`):
```
nice -n 15 $PY -m sim.drawer_insert.train --name r1 --iters 400            # CPU MuJoCo, 12 workers x 32 envs
nice -n 15 $PY -m sim.drawer_insert.evaluate --policy sim/drawer_insert/weights/policy.npz -n 500 --out sim/drawer_insert/results
MUJOCO_GL=egl $PY -m sim.drawer_insert.render --policy sim/drawer_insert/weights/policy.npz --names policy,straight --seeds 100002,100005 --out strip.png
$PY -m pytest -q sim/drawer_insert/tests
```
Training output goes to `sim/drawer_insert/runs/<name>/` (symlink to /data/sonic, git-ignored): `log.csv`,
`train.out`, `policy_XXXX.npz`, `best.npz`, `agent.pt`.

## Frames and units

- Slot frame (n, t, z): n = unit xy normal of the cabinet front pointing INTO the cabinet (`fixtures.cabinet.front_normal`),
  t = z x n = (-n_y, n_x) (the yaml's `t = -n_y x + n_x y`), z up. Right-handed; yaw is about +z (counter-clockwise
  seen from above), 0 when the tool's drawer axis is along n.
- Origin: n = 0 on the front plane (`n . xy = front_offset`), t = 0 on the slot centre line (centre of `slot_t`).
- The policy sees the BELIEVED slot frame (the yaml). The sim's world is the TRUE slot frame; they differ by
  `slot_dn, slot_dt` (+-1 cm) and `slot_dyaw` (+-1.5 deg).
- Tool point: top of the drawer's front wall, mid-thickness, i.e. between the fingertips. Flush = tool n ~ +wall/2 (~+5 mm).
- Units: metres, degrees, newtons.

## Observation (`policy.obs_vector` scales them; keys of the dict passed to `Policy.act`)

| key | meaning | unit |
|---|---|---|
| `n` | commanded tool n (believed frame; negative = in front of the cabinet) | m |
| `t` | commanded tool t relative to the believed slot centre line | m |
| `yaw` | commanded tool yaw relative to the believed slot normal (0 = un-rotated drawer would be straight) | deg |
| `off_est` | estimated in-hand offset: drawer centre line minus tool point, along the tool's t axis (constant per episode) | m |
| `yaw_est` | estimated in-hand yaw: drawer yaw minus tool yaw (constant per episode) | deg |
| `f` | (f_n, f_t, f_z): external force ON the tool (environment on tool) in the slot frame, minus its value at the start of the insertion; pushing into a jam gives f_n < 0 | N |
| `f_peak` | max \|f\| over the force samples (40 Hz) during the last step | N |
| `stopped` | 1.0 if the last step was halted by the force stop, else 0.0 | - |
| `prev` | last executed step (dn, dt, dyaw) | m, m, deg |
| `k` | decision index 0..47 | - |

Sign check against the real arm: in `rl/transitions/eval_ep0040.npz` (real, 2026-11-18) the obs column f.N reads
-30 N at the jam of an insertion pushing along +N, i.e. `feedback()["force"]` (Kinova `tool_external_wrench_force`,
base frame) is the force on the tool, as in the sim.

## Action

One decision = one tool step in the believed slot frame, `Policy.act` returns `(dn, dt, dyaw)`:
dn in [-5, +10] mm, dt in [-4, +4] mm, dyaw in [-2, +2] deg (yaw about the tool point). The env clips the commanded
pose to |t - t0| <= 30 mm, |yaw - yaw0| <= 12 deg, n in [n0 - 30 mm, wall/2 + 20 mm]. Decision period 0.2 s (5 Hz):
the target moves linearly over the first 80 % of the period, then holds.

## Episode

- Start: the scripted alignment puts the tool at t0 = -off_est, yaw0 = -yaw_est (the estimated drawer is centred and
  straight in the believed frame), with the drawer's back face 2-4 cm in front of the TRUE front plane.
- 48 decisions (9.6 s). Force samples at 40 Hz; force stop when the measured \|f\| > 60-70 N: the target freezes at
  the measured hand pose for the rest of the step (the real executor's 65 N stop).
- Success: both front corners within 10 mm of the true front plane and the true contact force < 40 N.
- Reward per step: 3 x (depth gain of the drawer's back, m) - 0.02 x max(0, peak force - 40 N)/10 N + 0.1 x success;
  +1 x success at the last step.
- Reset-to-states: a bank of 4096 start configurations (domain parameters + optional scripted pre-roll of 2-23 steps:
  straight push or push with random lateral/yaw steps, giving partly inserted, misaligned and jammed starts), sampled
  with weight ((p + 0.05)(1 - p + 0.05))^2 on each one's recent success p (peak at 0.5, unvisited first);
  mastered (p > 0.95) and hopeless (p < 0.02 after 12 visits) entries are replaced; 40 % of every batch are fresh
  starts from the evaluation distribution.

## Measured parameters (and randomization)

| parameter | sim range | source |
|---|---|---|
| drawer L x W x H | 0.115-0.130 x 0.098-0.108 x 0.060-0.070 m | reset notes (moved side camera: 0.12 x 0.105), side clouds of 24 reset attempts (height median 0.066 m; outlines 0.14 x 0.13 include ~1 cm depth edge bleeding), wrist front-wall top 0.065 |
| wall thickness | 8-11 mm | wrist image (insert_end_wrist.jpg), user (1 cm) |
| mass | 0.2-0.5 kg | user |
| clearance (slot - drawer width) | 3-10 mm | user |
| slot height - drawer height | 6-15 mm | keep-out: opening ~8 cm tall above the bottom rail |
| slot floor above table | 12-20 mm | reset notes (window floor 1.5 cm), execute.py BOTTOM_H 0.018 |
| drawer bottom above floor at start | 1-4 mm | execute.py (BOTTOM_H 18 mm vs ~15 mm floor) |
| back stop beyond flush | 0-3 mm (75 %), 3-20 mm (25 %) | user: back stop at full depth; reset notes: drawers were found recessed 2-10 cm, so the real slot may have no stop at flush |
| edge rounding | drawer back edges 0.5-3 mm, slot side-wall front edges 0.5-2 mm | wood lead-in (guess) |
| friction drawer-cabinet | 0.2-0.7 | user; slides at 2-6 N, sticks at ~50 N when jammed (emerges) |
| slot pose error | +-10 mm n, +-10 mm t, +-1.5 deg | user; yaml was re-fit by 9-46 mm several times |
| in-hand yaw | +-10 deg | measured 4-9 deg (execute logs "held drawer ... yaw err") |
| in-hand offset | +-30 mm, estimate error sd 2-15 mm per episode | side camera biased by several cm (logs, WRIST_SIDE_BIAS 5.8 cm) |
| in-hand yaw estimate error | sd 1-5 deg | side-depth yaw swung +-12 deg vs the wrist |
| in-hand compliance | sideways 1000-5000 N/m, slip at 8-25 N; yaw 1-5 N m/rad, slip at 0.3-1.0 N m | guess (pads); slips under jams |
| arm stiffness | 2e4-8e4 N/m, 50-200 N m/rad | real: 66 N at < 0.5 mm tracking deviation (exec logs) |
| force estimate | white noise sd 0.5-2.5 N/axis, residual bias +-3 N/axis | real: 3.5-5.4 N \|f\| during free moves |
| force stop | 60-70 N | task force_limit_n 65 |
| force feedback rate | 40 Hz | exec rate_hz 37-39 |

Physics: MuJoCo 3.17, timestep 1 ms, implicitfast, elliptic cones, contact solref 3 ms / solimp 0.95-0.99;
hand = 3 position-servoed joints (n, t, yaw) carrying a compliant grip (slide t + hinge yaw with springs and
Coulomb joint friction = slip); hollow drawer of 5 boxes + 2 capsule edges; the hand itself does not collide.

ENGINE_PLACEHOLDER

## Calling the policy on the robot (insertion phase)

`policy.py` needs only numpy (robot venv). `Insertion` does the frame bookkeeping in the base frame:

```python
import sys; sys.path.insert(0, "/home/r2r/robot_learning_final_pj")
from sim.drawer_insert.policy import Policy, Insertion

pol = Policy("/home/r2r/robot_learning_final_pj/sim/drawer_insert/weights/policy.npz")
# after carry + alignment: tool (fingertip centre on the front wall's mid-plane) at t = -off_est, yaw = -yaw_est in the
# believed slot frame, drawer back 2-4 cm in front of the front plane, drawer bottom 1-4 mm above the slot floor
ins = Insertion(pol, normal=N, offset=OFF, t_centre=0.5 * (SLOT_T[0] + SLOT_T[1]),
                yaw_straight=yaw_sq,         # tool yaw (deg, base) at which a drawer held square points along N
                off_est=off_est,             # drawer centre minus tool point along the tool's t axis (m)
                yaw_est=yaw_est,             # drawer yaw minus tool yaw (deg, CCW from above), e.g. HAND_YAW
                tool_xy=T_now[:2, 3], tool_yaw=yaw_of(T_now), f0=force())   # f0: base-frame force at the start
done = False
while not done:                                   # 48 decisions; the tool height stays constant
    xy, yaw = ins.next(force())                   # base-frame force now -> next tool target (base xy, yaw deg)
    # move there in ~0.2 s (straight line or streamed joint targets), force stop at the task limit (65 N);
    # sample the force during the move: peak = max |f - f0| (N)
    peak, stopped = move_with_force_stop(xy, yaw)
    T_now = measured_tool_pose()
    done = ins.done(peak, stopped, T_now[:2, 3], yaw_of(T_now))   # after a stop the measured pose becomes the command
```

- `yaw_sq` in execute.py terms is `wrapd(held["yaw"] + wrapd(ang(N) - ang(held["d0"])))` (without HAND_YAW), and the
  start yaw is `yaw_sq - yaw_est`; the start t puts the estimated drawer centre on the slot centre (`T_C + EDGE_TARGET`).
- `force()` is `feedback()["force"]` (base frame, force on the tool). `f_peak` must be max |f - f0| over the move;
  the executor's `max_abs_force_n` is |f| (not minus f0) and `max_force_n` is relative to the segment start, so stream
  the feedback (40 Hz) during the move or approximate `f_peak = max_force_n + |f_seg_start - f0|`.
- 5 Hz with 0.2 s moves is faster than today's ~1 step/1.3 s `try_step` lines; a slower real step is fine for the
  quasi-static contact, but the policy has 48 decisions: keep calling it until `done` or until the tool is at flush
  (`ins.cmd[0] > 0`) with |f| < 40 N for 3 steps.
- Reach: the policy drives the tool to n ~ +5 mm (flush) and may command up to +25 mm. The current keep-out stops a
  vertical gripper at `INS_NMAX = OFF - 0.036` (wrist body vs the cabinet front above the slot). With that limit, run
  the policy until it commands n > -0.036, then release and use the scripted push for the last ~4 cm; the policy's job
  there is the alignment and jam recovery during the first ~8-9 cm, where the real jams happened.

## Sim-to-real risks

- Contact model: rigid boxes with soft MuJoCo contacts (3 ms), Coulomb friction 0.2-0.7, rounded back edges as a
  guess for wood lead-in. Real wood has burrs, swelling, varying friction along the slot and stick-slip; jam forces
  in the sim (median peak ~70-100 N for the baselines) are in the real range (25-94 N) but their distribution is unverified.
- No pitch/roll: the hand height is fixed and the grasp is rigid in pitch and roll, so vertical cocking in the
  slot (6-15 mm vertical play) and catching the 1.5 cm floor lip are not modelled; the drawer bottom starts 1-4 mm
  above the slot floor. If the real bottom is below the lip the policy cannot fix it (it has no dz).
- Force estimate: the sim gives the true contact force + white noise + a constant bias at 40 Hz. The real Kinova
  estimate (from joint torques) has pose-dependent bias, lag and payload-model errors, and the program only gets
  per-segment maxima today. The policy reads f and f_peak; check on the robot that |f| at rest stays within ~5 N
  after zeroing and that the sign convention matches (pushing into a jam gives f_n < 0).
- Compliance: in-hand slip is a spring + Coulomb joint (sideways and yaw); real pads also compress along n and
  rotate about other axes; real in-hand yaw changes during carry are not in the estimate. The arm is a stiff servo
  (2e4-8e4 N/m), so impacts at the back stop are sharp; the real arm stops on the first 40 Hz sample over 65 N too.
- Execution: the sim moves 0.2 s linear steps at 5 Hz; the real program uses ~1-1.5 s planned lines. Slower is
  probably fine (quasi-static), faster streaming must keep the force stop.
- Geometry: the slot pose error (+-1 cm, +-1.5 deg) and the back stop position are the biggest unknowns; the real
  cabinet also moved under 60-90 N jams (it is taped down), which the sim does not model (fixed cabinet).
- Keep-out: see "Calling the policy": the real vertical gripper cannot follow the drawer to flush today.
