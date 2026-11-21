# Drawer insertion sim + RL policy (sim-to-real for the Kinova drawer task)

A held wooden drawer is pushed back-wall-first into the cabinet's bottom slot. The sim models only the insertion:
the hand holds the drawer by its front wall (compliant grasp) and moves in small Cartesian steps in the slot frame;
the cabinet slot has side walls, floor, ceiling rail and a back stop. A PPO policy trained with reset-to-states
(SGS-RL-style curriculum) outputs one bounded step per decision from signals the real robot has.

Files: `params.py` (measured values + randomization), `scene.py` (MJCF), `env.py` (MuJoCo CPU env), `vec.py`
(worker pool), `curriculum.py` (reset-to-states sampler), `train.py` (PPO), `policy.py` (numpy policy, robot side),
`baselines.py` (scripted straight / wiggle), `evaluate.py`, `render.py`, `newton_env.py` (Newton GPU twin),
`weights/policy.npz` (trained actor), `results/` (eval summaries, training curves, image strip), `tests/`,
`newton_try/` (Newton feasibility test and the CPU-vs-Newton validation).

## Results (shipped policy: `weights/policy.npz`, run r5)

500 episodes per row in the CPU MuJoCo reference env (`evaluate.py`, seeds 100000-100499, the same episodes for every
controller; deterministic policy). Success = front within 10 mm of flush and contact force < 40 N at the end of the
48 decisions. Full tables: `results/eval_*.json` (`python -m sim.drawer_insert.evaluate --report sim/drawer_insert/results`).

| start | straight push | wiggle/spiral search | PPO policy |
|---|---|---|---|
| carry (today's program: in-hand +-30 mm / +-10 deg, estimate sd 2-15 mm / 1-5 deg, slot pose +-10 mm / +-1.5 deg) | 36.2 % | 64.0 % | **96.6 %** |
| aligned (user's 11-21 proposal: square on the table, grasp the pull wall centre, in-hand +-5 mm / +-2 deg) | 51.8 % | 84.0 % | **98.4 %** |
| aligned + slot pose known to +-3 mm / +-0.5 deg | 81.8 % | 90.6 % | **100 %** |
| carry, after 12 straight-push steps (mostly jammed at the entrance) | 36.2 % | 59.6 % | **96.2 %** |
| carry, after 16 push steps with random t/yaw drift (cocked, part-way in) | 36.6 % | 44.2 % | **96.6 %** |

Carry start, success by start error (episodes per bin in brackets; e_t = true lateral offset of the drawer from
the slot centre line, e_yaw = true drawer yaw vs the slot normal, both at the start):

| | 0-3 mm (125) | 3-6 (95) | 6-10 (117) | 10-20 (125) | 20-60 (38) | 0-2 deg (268) | 2-4 (144) | 4-6 (54) | 6-10 (33) | clr 3-5 mm (152) | 5-7.5 (180) | 7.5-10 (168) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight | 0.58 | 0.62 | 0.30 | 0.11 | 0.00 | 0.41 | 0.37 | 0.31 | 0.06 | 0.32 | 0.31 | 0.46 |
| wiggle | 0.84 | 0.81 | 0.62 | 0.52 | 0.03 | 0.69 | 0.64 | 0.61 | 0.33 | 0.57 | 0.65 | 0.69 |
| policy | 0.95 | 0.98 | 0.99 | 0.96 | 0.92 | 0.97 | 0.95 | 0.98 | 0.97 | 0.95 | 0.99 | 0.95 |

Forces (carry): median per-episode peak contact force 71 N (straight 79, wiggle 72), 1.1 force stops per episode
(straight 15.9, wiggle 1.8), median 19 steps (3.8 s at 5 Hz) to the first success state. The peaks come from the
executor's force stop overshooting between 40 Hz samples when the drawer meets the cabinet face during the search
(about 60 % of the peaks) or the back stop (about 30 %); short fine-tunes with a larger force penalty (0.1 and 0.3
per 10 N, runs r3/r4, 20 and 12 iterations) did not lower them. What does lower them is a lower stop: with the stop at 30-40 N instead of 65 N the policy still
succeeds in 92.6 % (median peak 48 N, 3.7 stops per episode; r5 trained with 30 % of its starts at such a stop).

Stress tests outside the training ranges (carry start, policy / wiggle): force noise sd 2.5-5 N 93.8 / 64.6 %;
friction 0.7-1.0 95.2 / 58.2 %; clearance 2-3 mm 96.4 / 44.2 %; arm stiffness 5e3-2e4 N/m 97.0 / 57.6 %; decision
period 0.5-1.0 s 97.4 / 54.2 %; stiff grasp (slip at 40-80 N, 2-4 N m) 96.6 / 62.0 %; stop at 30-40 N 92.6 / 70.4 %.

`results/strip.jpg`: top and oblique views at decisions 0, 6, 12, 18, 24, 32, 47. Row 1: policy with a 32 mm lateral
start error (bumps the side wall, backs off, moves sideways and turns, goes in); row 3: the straight push on the same
episode stays jammed on the side wall; rows 2 and 4: a small-error episode.

### Training

| run | engine | start | samples | wall time | eval success (deterministic, 4096 or 384 starts) |
|---|---|---|---|---|---|
| r1 | CPU pool (12 x 32 envs) | push prior | 2.7 M | 20 min | 31 % (0.2 M) -> 57 % (0.5 M) -> 76 % (1.0 M) -> 83 % (2.0 M), flat after |
| r2 | Newton, 4096 worlds | r1 at 1.0 M | 10.7 M | 16 min | 74 % -> 92 % (5.3 M) -> 95 % (10.7 M); force penalty 0.05 (r1: 0.02) |
| r5 (shipped) | Newton, 4096 worlds | r2 | 8.9 M | 14 min | 95 % -> 96.1 %; starts: 50 % carry, 20 % aligned, 30 % stop at 30-40 N |
| abl_sgs | Newton, from scratch | push prior | 10.7 M | 30 min* | 31 % -> 50 % (3.6 M) -> 87 % (7.2 M) -> 95.4 % |
| abl_fresh | Newton, from scratch, no reset-to-states | push prior | 11.8 M | 30 min* | 31 % -> 50 % (3.9 M) -> 90 % (7.9 M) -> 95.9 % |

(*: three runs shared the GPU.) Curves: `results/curves.csv`. PPO: 2 x 256 tanh MLPs, Gaussian policy (state-free
std 0.5 -> 0.25), gamma 0.99, GAE 0.95, clip 0.2, 4 epochs, minibatch 4096, lr 3e-4, entropy 0.001; the actor's last
layer starts at "push 5.5 mm per step, no correction".

Ablation (SGS reset-to-states vs plain nominal starts, both from scratch, 60 iterations, CPU eval of the final
weights): carry 95.4 / 95.4 %, after 12 push steps 94.2 / 94.0 %, after 16 drift steps 93.4 / 91.2 %. In this task the
curriculum did not speed up learning on the nominal starts and helped at most slightly on cocked starts: the
straight-push prior already reaches the jams (64 % of its episodes jam), so exploration is not the bottleneck SGS-RL
targets; the success-weighted bank mostly redistributes samples. The shipped policy keeps the curriculum (40 % fresh
starts, 60 % bank) because it does not hurt and covers jammed / part-way states explicitly.

## How to run

Environment (venv on /data/sonic, `.venv` is a symlink; keep big files off /home):
```
uv venv /data/sonic/venvs/drawer_insert --python 3.12 && ln -s /data/sonic/venvs/drawer_insert sim/drawer_insert/.venv
uv pip install --python sim/drawer_insert/.venv/bin/python "newton[sim]" torch mujoco pillow pytest
```
From the repo root (`PY=sim/drawer_insert/.venv/bin/python`, `export PYTHONPATH=. OMP_NUM_THREADS=2
WARP_CACHE_PATH=/data/sonic/cache/warp_drawer`):
```
nice -n 15 $PY -m sim.drawer_insert.train --name a --engine newton --envs 4096 --iters 60        # ~95 % from scratch, ~15 min alone on the 4090
nice -n 15 $PY -m sim.drawer_insert.train --name b --engine newton --envs 4096 --iters 50 \
    --variants carry:0.5,aligned:0.2,low_stop:0.3 --init sim/drawer_insert/runs/a/agent.pt          # the r5 recipe
nice -n 15 $PY -m sim.drawer_insert.train --name c --iters 200                                     # CPU MuJoCo pool instead (12 workers)
nice -n 15 $PY -m sim.drawer_insert.evaluate --policy sim/drawer_insert/weights/policy.npz -n 500 [--variant aligned] [--preroll push:12] --out sim/drawer_insert/results
MUJOCO_GL=egl $PY -m sim.drawer_insert.render --policy sim/drawer_insert/weights/policy.npz --names policy,straight --seeds 100002,100011 --out strip.png --gif ep.gif
$PY -m pytest -q sim/drawer_insert/tests
```
Training output goes to `sim/drawer_insert/runs/<name>/` (symlink to /data/sonic, git-ignored): `log.csv`,
`train.out`, `policy_XXXX.npz`, `best.npz` (best deterministic eval), `agent.pt`. `--f_pen` sets the force penalty.

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
| `prev` | last step the policy commanded (dn, dt, dyaw), before the bounds and the force stop | m, m, deg |
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
- 48 decisions (9.6 s). Force samples at 40 Hz; force stop when the measured \|f\| > 60-70 N (30-40 N in the
  `low_stop` starts): the target freezes at the measured hand pose for the rest of the step (the real executor's stop).
- Success: both front corners within 10 mm of the true front plane and the true contact force < 40 N.
- Reward per step: 3 x (depth gain of the drawer's back, m) - 0.05 x max(0, true peak force - 40 N)/10 N
  + 0.1 x success; +1 x success at the last step (r1 used 0.02 for the force term).
- Reset-to-states: a bank of max(4096, 4 x batch) start configurations (start variant + domain parameters + optional
  scripted pre-roll of 2-23 steps: straight push or push with random lateral/yaw steps, giving partly inserted,
  misaligned and jammed starts), sampled
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
| force stop | 60-70 N (training r5: 30 % of starts at 30-40 N) | task force_limit_n 65 |
| start variants | carry (above) / aligned: in-hand +-5 mm, +-2 deg, estimate sd 1-3 mm, 0.3-1 deg, bottom 4-10 mm above the floor | user's 2026-11-21 proposal (lift ~2.5 cm above the table, floor ~1.5 cm) |
| force feedback rate | 40 Hz | exec rate_hz 37-39 |

Physics: MuJoCo 3.17, timestep 1 ms, implicitfast, elliptic cones, contact solref 3 ms / solimp 0.95-0.99;
hand = 3 position-servoed joints (n, t, yaw) carrying a compliant grip (slide t + hinge yaw with springs and
Coulomb joint friction = slip); hollow drawer of 5 boxes + 2 capsule edges; the hand itself does not collide.

## Engine: Newton (GPU) for training, MuJoCo (CPU) as the reference

- NVIDIA Newton was tried first as asked: `newton[sim]` 1.8.0 (warp-lang 1.19.0, mujoco-warp 3.17.0) installed from
  PyPI in minutes and its `SolverMuJoCo` (mujoco_warp) handles the box-in-box contacts: the aligned drawer slides to the
  stop, no tunnelling of the 1 cm walls, yawed/offset drawers jam with a readable force (`newton_try/test_slot.py`).
  One setting matters: Newton's default contact gap (0.1 m) gave ~200 contacts per world; `rigid_gap = 0` matches MuJoCo.
- `env.py` (MuJoCo 3.17 CPU, one model compiled per episode, full per-episode randomization) is the reference: it
  defines the task, runs the baselines, evaluation, rendering and tests, and is what the README numbers use.
- `newton_env.py` is its GPU twin: one Newton model replicated over N worlds, each world's parameters parsed from its
  own `scene.xml(p)` and written into the model arrays per reset. With the same 256 parameter sets and the straight
  push it agrees with the CPU env per episode in 96.5 % of the episodes (success 40.6 % vs 38.7 %, median depth
  difference 0.0 mm; remaining differences come from force-noise draws; `newton_try/validate.py`).
- Throughput (RTX 4090): ~20k decisions/s at 4096 worlds (8 s per 48-step batch incl. reset) vs ~2.8k/s for the
  12-worker CPU pool incl. PPO updates. Run r1 (first policy) trained on the CPU pool; r2 and r5 (the shipped policy)
  continued on Newton; the CPU evaluation of the Newton-trained policy is also a sim-to-sim transfer check (r2: 95.0 %
  on 4096 Newton starts, 94.6 % on 500 CPU starts).

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
- Force stop: the policy was trained with the 60-70 N stop and (30 % of starts) a 30-40 N stop. Lowering
  `s.executor.force_limit` to ~40 N for the insertion moves (a program may lower it, never raise it) costs ~4 points
  of sim success and cuts the median peak from ~71 N to ~48 N; worth it given that 60-90 N jams moved the cabinet.
- Start variants: the same weights serve the current carry-and-insert start and the user's "square it on the table
  first" start (pass small `off_est`/`yaw_est` and lift the drawer bottom ~0.5-1 cm above the slot floor).
- 5 Hz with 0.2 s moves is faster than today's ~1 step/1.3 s `try_step` lines; a slower real step is fine (sim: 97.4 % with 0.5-1.0 s steps) for the
  quasi-static contact, but the policy has 48 decisions: keep calling it until `done` or until the tool is at flush
  (`ins.cmd[0] > 0`) with |f| < 40 N for 3 steps.
- Reach: the policy drives the tool to n ~ +5 mm (flush) and may command up to +25 mm. The current keep-out stops a
  vertical gripper at `INS_NMAX = OFF - 0.036` (wrist body vs the cabinet front above the slot). With that limit, run
  the policy until it commands n > -0.036, then release and use the scripted push for the last ~4 cm; the policy's job
  there is the alignment and jam recovery during the first ~8-9 cm, where the real jams happened.

## Sim-to-real risks

- Contact model: rigid boxes with soft MuJoCo contacts (3 ms), Coulomb friction 0.2-0.7, rounded back edges as a
  guess for wood lead-in. Real wood has burrs, swelling, varying friction along the slot and stick-slip; jam forces
  in the sim (median peaks ~70-80 N, set by the force stop's overshoot) are in the real range (25-94 N), but their
  distribution is unverified.
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
  fine in the sim (97.4 % with 0.5-1.0 s steps); faster streaming must keep the force stop.
- Geometry: the slot pose error (+-1 cm, +-1.5 deg) and the back stop position are the biggest unknowns. With no
  stop at flush the policy can only stop at the believed front plane, so a front-plane error near 1 cm leaves the
  drawer recessed or proud by ~1 cm (11 of the shipped policy's 17 failures in 500 carry episodes: recessed 7-12 mm
  in a cabinet whose stop is 1-2 cm past flush; 2 more had a lateral error near the 30 mm search bound); a better front-plane estimate
  helps more than anything else (straight push 36 % -> 82 % when also aligned). The real cabinet also moved under
  60-90 N jams (it is taped down), which the sim does not model (fixed cabinet).
- Keep-out: see "Calling the policy": the real vertical gripper cannot follow the drawer to flush today.
