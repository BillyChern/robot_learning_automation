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

## v2: vertical model and dz action (`weights/policy_v2.npz`)

After attempt 112 (fixed-depth jam, drawer top ~1 mm under the rail) the sim got a vertical model; v1 weights,
env semantics and the Newton twin are unchanged (v2 is opt-in: `params.sample(rng, variant, v2="mix")`, CPU env).
- Geometry: opening 76-82 mm (measured 1.5-9.5 cm above the table), inner floor 13-17 mm, a front lip on the bottom
  rail 0-3 mm high and 10-20 mm deep, rounded lip and rail edges (2 mm).
- Grasp: pitch and roll slip joints at the tool point (3-15 N m/rad, slip at 0.15-0.6 N m, +-8 deg); start pitch
  -1.5..+3 deg (+ = back down), roll +-1.5 deg.
- Carry height (drawer bottom above the lip, `lift`): 50 % fits (2 mm .. room - 2 mm), 25 % too low (-5..+0.5 mm: the
  back catches the lip), 25 % too high (top within -1..+4 mm of the rail: wedges under it); `--v2 wedge` = only the
  last two. `gap_est` = lift + N(0, 1-3 mm).
- Action: 4th component dz in [-3, +3] mm per step (cumulative +-12 mm), the hand is servoed in z too.
- Obs adds `z` (cumulative commanded dz, m), `gap_est` (m, constant), `jam_rep` (force stops so far within 5 mm of the
  commanded n) and dz in `prev` (4 entries): 18 inputs. `Policy` picks v1/v2 from the weights; v1 weights also run on
  a v2 obs (they read the first 14 inputs, dz = 0).
- Training (`train --v2 mix --init <v1 agent.pt>`, CPU pool, v1 weights zero-padded so dz starts at 0): 130 iterations, 2.2 M samples, 17 min on 12
  CPU workers (starts: 50 % carry, 20 % aligned, 30 % with a 30-40 N stop; heights `mix`). Deterministic eval on 384
  carry/mix starts: 60 % (v1 weights) -> 83 % (0.17 M) -> 91 % (0.8 M), flat after; shipped: best checkpoint (it 50-60).

Results (500 episodes per row, v2 model, CPU MuJoCo; `results/v2/eval_*.json`):

| start | straight | wiggle | straight + height from gap_est | v1 policy (dz = 0) | **v2 policy** |
|---|---|---|---|---|---|
| carry, heights mix (50 % fit, 25 % low, 25 % high) | 22.6 % | 21.4 % | 20.0 % | 41.8 % | **86.2 %** |
| aligned, heights mix | 33.2 % | 32.0 % | 26.6 % | 45.4 % | **89.8 %** |
| carry, wedge only (low or high) | 17.4 % | 7.6 % | 19.2 % | 18.8 % | **85.0 %** |
| aligned, wedge only | 28.8 % | 12.4 % | 25.0 % | 22.6 % | **83.6 %** |

By height (carry/mix): fits 91.6 %, too low 76.4 %, too high 85.2 % (straight 25.5 / 8.7 / 31.1 %). The v2 policy
fixes the height first: on aligned wedge starts the median commanded z after 3 / 6 steps is +3.1 / +5.9 mm when the
back would catch the lip and -0.9 / -2.0 mm when the top meets the rail, and its sideways excursion is 11 mm; v1 (no
dz) searches sideways to 26 mm on the low starts and succeeds in 4 % (attempt 112's pattern). Peak forces stay at
the stop level (median ~80 N at the 60-70 N stop). The v1 model and benchmark are unchanged (v1 weights: 96.6 % on
the v1 carry set).

Robot side for v2: build `Insertion(..., tool_z=<start tool z, m>, gap_est=<drawer bottom above the slot lip, m>)`,
and after each `xy, yaw = ins.next(force())` move to height `ins.z_target_abs` as well (same 5 Hz step, keep the tool
vertical); pass the measured tool z to `ins.done(peak, stopped, xy, yaw, tool_z)`. `ins.obs['jam_rep']` is computed
inside `Insertion` from the stops. gap_est = start tool z - (drawer height) - (lip top: table + ~1.5 cm), with the
tool point at the top of the front wall.

## First real run (attempt 112, 2026-11-21): diagnosis

- Bookkeeping checked, no sign or frame error: the tool poses rebuilt by FK from the step plans match `ins.cmd`
  (t +0.3 -> +29 mm, yaw 21.86 -> 32.0 deg = +10.1 about ang(N)); the jaw axis is along +N as in the sim; t = (-N1, N0)
  in both; the force is in the base frame (push-stage transitions read f.N -50, f.z +45 N; the tool-frame reading
  would imply a 61 N downward reaction to a horizontal push), so f_n < 0 at a jam as trained.
- Same start in the sim (aligned, back at the front plane, 40 N stop, 33 decisions): 93 % success, |t| ~2 mm.
- What differs is the jam: the drawer stopped at tool n-OFF -0.048..-0.068 (back ~6-7 cm in), the fixed-depth
  stop the real program has hit since attempt 44 ("for every yaw/dz tried"). Nothing in the sim stops a lined-up
  drawer mid-slot, so the policy runs its learned recovery (back off, follow f_t and its previous lateral step,
  +1 deg per back-off) up to the bounds. Adding an impassable floor bump at 6.5 cm to the sim reproduces the
  stop at the same tool depth (0 % success) with a smaller search.
- Likely cause (not in the sim): a vertical wedge. Tool height was 9.4 cm above the table with a 6.3-6.6 cm drawer and
  an opening of ~1.5-9.5 cm; the sim holds the drawer level (no pitch, no dz) with 1-4 mm under it.
- Robot-side changes: log `ins.obs` and `ins.step` every decision (the f vector was not logged); when two force stops
  happen within 5 mm of the same n, stop the policy and push (lateral search cannot fix a fixed-depth stop); with
  the table-aligned start, clip the search (`ins.T_BOUND = 0.008; ins.YAW_BOUND = 3.0`); check the drawer height in
  the opening (top >= 5 mm under the rail).
- Sim-side next step: pitch compliance in the grasp, vertical play from measurement, and a dz action.

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

## Lying cabinet (vertical insertion, task drawer_lying, 2026-11-23)

The cabinet lies on its back, compartments open to the top. The drawer, held by its pull (tool pointing down), is
lowered back wall first into the compartment nearest the robot until its top is flush with the rim. The sim models
only that descent: the hand carries the drawer through a compliant pinch grasp at the pull and moves in small steps in
the opening frame; the compartment has a floor, four walls with rounded inner rim edges and, beyond the divider, the
middle compartment. A PPO policy (reset-to-states curriculum as above) outputs one bounded tool step per decision from
signals the robot has. The standing-cabinet sim (sections above) is unchanged.

Files (next to the old ones): `lying_params.py` (measured values + randomization), `lying_scene.py` (MJCF),
`lying_env.py` (MuJoCo CPU reference), `newton_lying.py` (Newton GPU twin), `lying_policy.py` (numpy runtime: `Policy`,
`Descent`, `start_pose`), `lying_baselines.py` (straight descent, spiral search, pre-rolls), `lying_eval.py`,
`lying_render.py`, `weights/policy_lying.npz`, `results/lying/`, `tests/test_lying.py`,
`newton_try/validate_lying.py`. `train.py --task lying` (default `--task slot` = the old task).

### Measured geometry

From `config/tasks/drawer_lying.yaml` and the side clouds `runs/drawers_setup/lying/{cloud_20261123.npy,
measure_20261123_160851/cloud_clear.npy}` and `wrist_down.jpg`:
- Both clouds show the drawer INSIDE the target compartment (so does `side_20261123.jpg`), not on the table. In the
  cabinet frame the target compartment's top reads 94-98 mm above the table (rim 95; label frame and knob up to 103):
  the inserted drawer is flush.
- Wrist look-down (30 cm above the rim, cabinet 776 px = 0.275 m): the drawer front spans 217 x 335 px = 0.075-0.077 x
  0.113 m, i.e. it fills the 0.0757 x 0.115 opening with ~1-2 mm per side. Sim: clearance 2-6 mm per axis (1-2 mm in
  the `tight` stress test).
- The pull is a label frame (~33 x 59 mm) with a round knob (~16 mm) at its edge, ~17 mm off the drawer centre along
  the drawer's thickness (the opening's x axis); depending on how the drawer stands it is on either side. In-hand
  offset in the sim: x +-25 mm, y +-15 mm (the policy gets the estimate; what matters is the estimate error).
- The standing drawer itself is in the live run's reset survey cloud (`runs/20261123_1616_drawer/agents/reset/attempts/
  attempt_14/cloud.npy`, 16:56): its top face is 0.0905-0.093 m above the table (knob up to 0.100), footprint
  0.073 x 0.110 m (incl. depth edge bleeding). So the drawer is ~0.091 m front-to-back (not 0.1-0.12), and since it sits flush
  (0.097) when inserted, the compartment floor is ~5 mm above the table (the yaml's floor_h 0.013 is too high; the
  side camera cannot see the bottom of the 76 mm wide compartments). Sim: L 0.088-0.094 m, flush -2..+3 mm, floor
  = rim + flush - L (>= 3 mm), believed L 0.091.

### Model

| parameter | sim range | source |
|---|---|---|
| opening (x along the cabinet, y across) | 0.0745-0.0770 x 0.1135-0.1165 m | yaml 0.0757 x 0.115 (12 mm walls) |
| rim / wall thickness | 0.093-0.097 m / 10-14 mm | yaml 0.095 / 0.012; inserted top 0.097 |
| drawer length, flush offset | 0.088-0.094 m, -2..+3 mm | standing drawer 0.091 (above) |
| clearance per axis (opening - drawer) | 2-6 mm (stress test 1-2 mm) | wrist look-down: front fills the opening |
| edge rounding | drawer bottom edges 0.5-3 mm, rim inner edges 0.5-2 mm | guess (wood) |
| mass, friction | 0.12-0.30 kg, 0.2-0.6 | wood drawer ~0.18 kg (estimate) |
| opening pose error (true - believed) | +-10 mm x and y, +-3 deg | task brief |
| in-hand offset (drawer top centre - tool point, tool frame) | x +-25 mm, y +-15 mm | brief (+-15 mm), knob ~17 mm off centre |
| in-hand yaw / tilt | +-6 deg / 0-8 deg in any direction | brief |
| tool point above the drawer's top face (`hold`) | 6-18 mm | knob ~2 cm |
| wrist estimate errors (sd) | offset 2-5 mm, yaw 1-3 deg, tilt 1-3 deg per axis, hold 1-3 mm | guess |
| grasp compliance (at the tool point) | tilt about the jaw axis 0.5-5 N m/rad, slip 0.08-0.5 N m; about the other horizontal axis 5-20, 0.4-1.5; yaw 2-8, 0.3-1.0; slide along z: slip at 20-50 N, -8..+4 mm | pinch on a round knob (guess) |
| arm | 2e4-8e4 N/m, yaw 50-200 N m/rad (position servo) | as the standing sim |
| force estimate | contact force + bias +-3 N/axis + drift +-30 N/m of height + white noise sd 1-3 N/axis, 40 Hz | brief (+-5 N) |
| force stop | 40-65 N on \|f - f0\| | brief, task 65 N |
| start | drawer's lowest point 5-25 mm above the true rim | scripted carry |
| decision period | 0.18-0.22 s, 60 decisions | |

Physics as in the standing sim (MuJoCo 3.17, 1 ms, implicitfast, elliptic cones, solref 3 ms); the hand is four
position-servoed joints (x, y, z, yaw); the drawer hangs from four spring + Coulomb slip joints at the tool point
(z slide, tilt about the tool x and y axes, yaw; the jaw-closing axis, i.e. the soft tilt axis, is tool x or y with
equal odds); limits stiffened (solreflimit 4 ms). The drawer's in-hand pose (offset, yaw, tilt, hold) is baked into its
geoms so the Newton twin needs no per-world body frames. The hand itself does not collide (the fingers stay `hold`
above the face, above the rim at flush).

### Frames, observation, action (`lying_policy.py`)

- Opening frame: origin at `fixtures.cabinet.target_centre` (believed, base xy) on the table, x = (cos, sin) of
  `fixtures.cabinet.yaw_deg` (21.2 deg; +x points away from the robot base, toward the middle compartment), y = z x x,
  z up. The sim world is the TRUE opening frame; the policy only sees the believed one.
- Tool frame (sim): x = the jaw-closing axis (the `yaw_deg` direction of `rlpj.robot.plan.tool_down_T`), y = x turned
  90 deg counter-clockwise seen from above (= minus the robot's tool y), z up. Tool yaw in the opening frame = tool
  yaw (deg, base, `atan2(T[1,0], T[0,0])` of the measured tool pose) - `yaw_straight`; `yaw_straight` = the base yaw
  at which the jaw axis lies along the opening's +x, i.e. `yaw_deg` (21.2).
- Tool point: between the fingertips, where they pinch the knob; `hold` = its height above the drawer's top face.
- Units: m, deg, N. Forces are the base-frame force ON the tool (`feedback()["force"]`, as verified for the standing
  sim), minus f0 taken at rest at the start of the descent, rotated into the opening frame. The floor or rim pushing
  the drawer up gives f_z > 0.

| obs key | meaning | unit |
|---|---|---|
| `x`, `y` | commanded tool point in the believed opening frame | m |
| `yaw` | commanded tool yaw relative to `yaw_straight` | deg |
| `z` | commanded tool height minus the believed flush height `rim_z + hold_est` (0 = drawer top flush if it hangs straight) | m |
| `off` | wrist estimate (x, y) of drawer top-face centre minus tool point, tool frame (constant) | m |
| `yaw_est` | wrist estimate of drawer yaw minus tool yaw (constant) | deg |
| `tilt` | wrist estimate (tx, ty) of the drawer's in-hand tilt: the drawer axis is Rz(yaw_est) Ry(-tx) Rx(ty) (0, 0, -1), i.e. its bottom sits ~L tan(tx) further along tool +x than its top (constant) | deg |
| `f` | (f_x, f_y, f_z) force on the tool minus f0, opening frame | N |
| `f_peak` | max \|f - f0\| over the 40 Hz samples of the last move | N |
| `stopped` | 1.0 if the last move was halted by the force stop | - |
| `prev` | last step the policy returned (dx, dy, dyaw, dz), before bounds and the stop | m, m, deg, m |
| `k` | decision index 0..59 | - |

Action (`Policy.act` -> `(dx, dy, dyaw, dz)`): dx, dy in [-4, +4] mm along the opening's x / y, dyaw in [-2, +2] deg
about the tool point, dz in [-8, +4] mm (negative = down). The commanded pose is clipped to |x - x0|, |y - y0| <= 15 mm,
|yaw - yaw0| <= 8 deg, z in [-15 mm, z0 + 20 mm]. After a force stop the measured pose becomes the command.

Episode: start (scripted carry + wrist estimate, `start_pose`): tool yaw squares the estimated drawer with the opening,
tool xy puts the estimated bottom-face centre over the believed opening centre, drawer's lowest point 5-25 mm above
the rim. 60 decisions of 0.18-0.22 s; the target moves linearly over 80 % of the period, then holds. Force samples at
40 Hz, stop when \|f - f0\| (measured) > 40-65 N. Success: all four top corners within 10 mm of the true rim and the
true contact force < 40 N at the end. Reward per step: 3 x (depth gain of the bottom centre, m) + 5 x (gain in lateral
distance of the bottom centre to the true opening centre, m; privileged shaping) - 0.02 x max(0, true peak - 40 N)/10 N
+ 0.1 x success, +1 x success at the last step. (With 0.05 per 10 N, as in the standing sim, the first run learned to
hover above the rim within 20 iterations.) Reset-to-states: the same SGS-style bank; pre-rolls are a straight descent
(`descend`) or a descent with random lateral/yaw steps (`drift`) of 2-29 steps (resting on the rim, cocked on an edge,
part-way in); 40 % fresh starts; start variants 60 % `carry` (full ranges), 40 % `small` (opening +-3 mm / +-1 deg,
tilt 0-3 deg).

### Results (shipped policy: `weights/policy_lying.npz`, run ly_n3 iteration 50)

500 episodes per row in the CPU MuJoCo reference env (`lying_eval.py`, seeds 100000-100499, the same episodes for every
controller; deterministic policy; 60 decisions). Success = all four top corners within 10 mm of the true rim and the
true contact force < 40 N at the end. Full tables: `results/lying/eval_*.json`
(`python -m sim.drawer_insert.lying_eval --report sim/drawer_insert/results/lying`).

| start | straight descent | spiral search | PPO policy |
|---|---|---|---|
| carry (full ranges: opening +-10 mm / +-3 deg, in-hand estimate sd 2-5 mm / 1-3 deg, tilt 0-8 deg) | 21.0 % | 14.0 % | **66.2 %** |
| carry, released when seated (z < 1 cm and f_z > 10 N for 2 decisions) | 15.6 % | 9.4 % | **64.8 %** |
| small (opening known to +-3 mm / +-1 deg, tilt 0-3 deg) | 46.2 % | 32.8 % | **86.6 %** |
| big_open (opening +-15 mm / +-5 deg) | 9.2 % | 5.8 % | **44.2 %** |
| after 8 straight-descent steps (resting on the rim / cocked) | 21.0 % | 14.4 % | **66.2 %** |
| after 10 drift steps (random lateral/yaw on the way down) | 19.2 % | 15.0 % | **61.8 %** |

Carry start, success by start error (episodes per bin in brackets; e_xy = true offset of the drawer's bottom-face
centre from the true opening centre at the start; e_yaw = true drawer yaw vs the opening; tilt = in-hand tilt):

| | 0-3 mm (36) | 3-6 (100) | 6-10 (157) | 10-15 (145) | 15-30 (62) | 0-1 deg (151) | 1-2 (111) | 2-4 (167) | 4-10 (71) | tilt 0-2 (116) | 2-4 (135) | 4-6 (112) | 6-8 (137) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight | 0.69 | 0.51 | 0.15 | 0.04 | 0.00 | 0.24 | 0.29 | 0.20 | 0.04 | 0.28 | 0.21 | 0.23 | 0.12 |
| spiral | 0.72 | 0.35 | 0.05 | 0.01 | 0.00 | 0.17 | 0.21 | 0.13 | 0.00 | 0.16 | 0.14 | 0.18 | 0.09 |
| policy | 0.86 | 0.89 | 0.72 | 0.56 | 0.27 | 0.69 | 0.75 | 0.67 | 0.45 | 0.78 | 0.67 | 0.64 | 0.56 |

Stress tests outside the training ranges (carry start; straight / spiral / policy): force noise sd 3-5 N and bias
+-6 N 18.4 / 10.8 / 46.4 %; clearance 1-2 mm 5.4 / 3.8 / 40.0 %; in-hand estimate sd 5-10 mm, 3-6 deg 8.2 / 5.2 /
45.4 %; tilt estimate passed as zero 21.6 / 12.0 / 62.2 %; friction 0.6-0.9 17.8 / 10.6 / 65.2 %; loose grasp (slip
torques / forces / 3) 20.0 / 12.8 / 66.6 %; arm 5e3-2e4 N/m 32.4 / 20.2 / 57.4 %; decision period 0.5-1.0 s
21.4 / 16.0 / 62.2 %; force stop 25-40 N 16.8 / 12.6 / 65.8 % (median peak 62 N instead of 84 N).

Forces (carry): median per-episode true peak 84 N (straight 81, spiral 79), 15.7 force stops per episode (straight
24.7, spiral 12.4), median 23 decisions to the first success state. The policy probes with the force stop: it
descends until the stop, reads the lateral force, shifts and tries again. The peaks are the stop's overshoot between
40 Hz samples. Failures (300 carry episodes, 105 fails): 87 never entered the opening (start error median 11.6 mm),
10 stuck part-way, 8 fully in but pressing > 40 N at the end.

`results/lying/strip.jpg`: oblique and side views at decisions 0, 5, 10, 15, 20, 30, 45, 59. Rows 1 and 3: the policy
with 14 and 13 mm start errors (lands on the rim, shifts toward the lateral force, drops in, seats); rows 2 and 4: the
straight descent on the same episodes stays on the rim. GIF of the four episodes: `/data/sonic/drawer_insert_runs/
lying_episodes.gif` (8 MB, not in git).

### Training (RTX 4090, local; niced, 1-2 CPU threads)

| run | engine | init | change | samples | wall | eval success (deterministic, 4096 carry starts) |
|---|---|---|---|---|---|---|
| ly_c1 | CPU pool (10 x 32) | prior "descend 5 mm/step" | force penalty 0.05 / 10 N | 0.27 M | 2 min | collapsed to hovering above the rim (0 %) by it 20; stopped |
| ly_n1 | Newton, 4096 worlds | prior | penalty 0.02, xy shaping; first geometry (drawer 0.082 m, off_x +-15 mm), 40 decisions | 3.7 M | 8 min | 12 % (it 10) -> 24 % (it 20) |
| ly_n2 | Newton | ly_n1 it 20 | measured drawer length 0.091 m, off_x +-25 mm | 4.8 M | 9 min | 28 % -> 36 % (it 30) |
| ly_n3 | Newton | ly_n2 it 30 | 60 decisions (was 40), soft grasp axis 0.5-5 N m/rad / 0.08-0.5 N m | 11.1 M | 23 min | 55 % (it 10) -> 59 -> 60 -> 64 -> 66 % (it 50) |

PPO as for the standing sim (2 x 256 tanh, Gaussian, gamma 0.99, GAE 0.95, clip 0.2, 4 epochs, minibatch 4096, lr 3e-4);
19 inputs, 4 actions; the actor's last layer starts at "descend 5 mm per step, no correction". ~16-27 s per 4096-world
iteration (incl. pre-rolls and resets). Curves: `results/lying/curves_<run>.csv`.

### Engine

`newton_lying.py` is the GPU twin of `lying_env.py` (same MJCF per world, parsed into the replicated Newton model: geom
poses and sizes incl. the baked in-hand pose, drawer inertia, joint springs / slip / damping, servo gains; Newton 1.8,
`SolverMuJoCo`, `rigid_gap = 0`). Validation with the straight descent on the same parameter sets
(`newton_try/validate_lying.py`, `results/lying/newton_validate.json`): 1024 carry episodes, success 11.7 % on both,
per-episode agreement 97.5 %, median depth difference 0.07 mm, median peak force 79.5 vs 80.5 N, no contact overflow.
Throughput on the RTX 4090: ~11k decisions/s at 2048 worlds (CPU MuJoCo: ~350 decisions/s per process).

### Calling the policy on the robot (descent phase)

`lying_policy.py` needs only numpy (robot `.venv`). `Descent` does the bookkeeping in the base frame:

```python
import sys; sys.path.insert(0, "/home/r2r/robot_learning_final_pj")
import numpy as np
from sim.drawer_insert.lying_policy import Policy, Descent, start_pose

cab = config.load()["task"]["fixtures"]["cabinet"]
C, AX = cab["target_centre"], cab["yaw_deg"]               # believed opening centre (base xy), opening x axis (deg)
RIM_Z = calib.table_z() + cab["rim_h"]                     # rim height, base z
YS = AX                                                    # tool_down_T yaw at which the jaw axis lies along +x
est = dict(off=(ox, oy), yaw=dyaw, tilt=(tx, ty), hold=h)  # wrist look-down of the held drawer, sim tool frame (see above)
pol = Policy("/home/r2r/robot_learning_final_pj/sim/drawer_insert/weights/policy_lying.npz")
xyz0, yaw0 = start_pose(C, AX, RIM_Z, YS, est, length=0.091, gap=0.015)   # move there (tool vertical), settle
T = measured_tool_pose()
ins = Descent(pol, C, AX, RIM_Z, YS, est, T[:3, 3], yaw_of(T), f0=force())  # f0: base force at rest, drawer held
done = False
while not done:                                           # 60 decisions
    xyz, yaw = ins.next(force())                          # base force now -> next tool target (base xyz, yaw deg)
    peak, stopped = move_with_force_stop(tool_down_T(xyz, yaw), limit=50)   # ~0.2-1 s; peak = max |f - f0| (N)
    T = measured_tool_pose()
    done = ins.done(peak, stopped, T[:3, 3], yaw_of(T))  # after a stop the measured pose becomes the command
    log(k=ins.k, obs=ins.obs, step=ins.step, target=ins.target, tool=T, f=force(), peak=peak, stopped=stopped)
# then open the gripper and rise straight up
```

- Signs: `ins.obs["f"][2] > 0` when the drawer presses on the floor or the rim; `z` decreases as the tool goes down;
  `dz < 0` is down. `off`, `tilt`, `yaw_est` are in the sim tool frame (x = jaw axis, y = 90 deg CCW from it seen from
  above, z up): convert from the camera / robot tool frame (robot tool y = - sim y, robot tool z = - sim z).
- `yaw_of(T)` = `degrees(atan2(T[1, 0], T[0, 0]))` (the jaw axis in the base plane, as `tool_down_T` defines it).
  If the arm holds the drawer with the jaw axis near `yaw_deg + 180`, pass `axis_deg = yaw_straight = yaw_deg + 180`
  (and the estimates in that tool frame); `Descent` refuses a start yaw more than 30 deg from `yaw_straight`.
- Force stop: the policy was trained with stops at 40-65 N on |f - f0| (40 Hz). Use the task's limit or lower; a stop
  must freeze the tool at its measured pose for the rest of the step.
- f_peak must be max |f - f0| over the move (stream the 40 Hz feedback); `max_force_n` relative to the segment start
  is a fallback (`max_force_n + |f_seg_start - f0|`).
- Step time: the sim used 0.2 s steps; 0.5-1 s real steps are fine for this quasi-static contact (see `slow_steps`).
- Stop early (and release) when `ins.cmd[3] < 0.01` and `ins.obs["f"][2] > 10` for two decisions: the drawer is on the
  floor. Otherwise run the 60 decisions; the success check (side camera top height) decides.
- Log per decision: `ins.k`, `ins.obs` (all keys), `ins.step`, `ins.target`, the measured tool pose after the move,
  the raw base force at decision time, `peak`, `stopped`, move duration. Per attempt: C, AX, RIM_Z, YS, `est` and how
  it was measured (wrist image + pose), f0, the start pose, the final drawer top height and success.

### Sim-to-real risks (lying cabinet)

- Opening pose: the policy's success falls with the start error of the drawer's bottom vs the true opening (table
  above); the +-1 cm / +-3 deg opening error of the brief dominates it. A wrist look-down at the opening right before
  the descent (opening edges at ~2 px/mm) that brings the error to +-3 mm / +-1 deg (`small`) helps more than anything.
- Lateral force: the search uses the lateral force (~10 N toward the opening when the bottom rests on a rim edge, at
  30-40 N down). The real estimate comes from joint torques with pose-dependent bias; if its lateral error over a 10 cm
  descent is well above the sim's (bias +-3 N, drift +-3 N, noise 1-3 N), success drops (`noisy_force`, noise 3-5 N,
  bias +-6 N: 46.4 %). Check on the robot: hold
  the drawer, rest it on the rim on purpose, read `ins.obs["f"]` (f_z > 0, f_xy pointing toward the opening).
- Grasp: the pinch on the knob is modelled as spring + Coulomb slip joints (guessed stiffness and slip torques, the
  soft axis along the jaw axis), with a slide of -8..+4 mm along z. The real drawer can pivot more, slip out or be
  dropped; `loose_grip` covers weaker grasps. Releasing is not modelled.
- In-hand estimates: the policy relies on `off`, `yaw_est`, `tilt` (sd 2-5 mm, 1-3 deg in training). Tilt from a
  wrist look-down at a held drawer is hard: passing tilt = (0, 0) costs ~4 points (62.2 vs 66.2 %), estimate errors
  of sd 5-10 mm / 3-6 deg (`poor_est`) ~21 points (45.4 %).
- Geometry: drawer length 0.091 from one survey cloud, floor height inferred (~5 mm), clearance from one wrist image
  (front fills the opening, ~1-2 mm per side; `tight`), rim edge rounding guessed. Wood friction, burrs and swelling
  are not modelled; the cabinet is fixed in the sim (the real one could shift or tip under 40-65 N).
- Execution: 0.2 s steps in the sim; real 0.5-1 s steps (`slow_steps`) are fine if the force stop freezes the tool at
  its measured pose. Fingers and the hand do not collide in the sim.

### How to run (lying)

From the repo root (`PY=sim/drawer_insert/.venv/bin/python`, `export PYTHONPATH=. OMP_NUM_THREADS=2
WARP_CACHE_PATH=/data/sonic/cache/warp_drawer`):
```
nice -n 15 $PY -m sim.drawer_insert.train --task lying --engine newton --envs 4096 --iters 100 --name a     # from scratch
nice -n 15 $PY -m sim.drawer_insert.train --task lying --engine newton --envs 4096 --iters 100 --name b --init sim/drawer_insert/runs/a/agent.pt
nice -n 15 $PY -m sim.drawer_insert.train --task lying --workers 10 --envs 320 --name c                    # CPU MuJoCo pool instead
nice -n 15 $PY -m sim.drawer_insert.lying_eval --policy sim/drawer_insert/weights/policy_lying.npz -n 500 [--variant small] [--preroll descend:8] [--early] --out sim/drawer_insert/results/lying
MUJOCO_GL=egl $PY -m sim.drawer_insert.lying_render --policy sim/drawer_insert/weights/policy_lying.npz --names policy,straight --seeds 100092,100064 --out strip.jpg --gif ep.gif
$PY sim/drawer_insert/newton_try/validate_lying.py 1024 carry
$PY -m pytest -q sim/drawer_insert/tests
```
