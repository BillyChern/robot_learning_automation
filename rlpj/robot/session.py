"""RobotSession: the one API every robot program uses (DESIGN.md section 5), generalised from the hardware-verified
FAR programs. Start-state and model FK checks, watchdog, model check of every segment before it is sent, tracking
with deviation limits, stop handling (Abort / Ctrl+C / SIGTERM / SIGHUP -> record the stop, return home on a
checked path), live view on port 8090, optional episode recording. Motion needs RLPJ_MOTION=1; otherwise the run
is dry: the robot is only read and the session advances a simulated joint state instead of moving.

Env: RLPJ_ATTEMPT_DIR (default runs/manual/<timestamp>), RLPJ_RECORD_DIR (episode dir, optional), RLPJ_MOTION."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

from rlpj import config
from . import calib, cams, kortex, sim
from .executor import Executor
from .home import plan_return
from .kin import Kin, unwrap_near, wrap180
from .plan import NORMAL, SLOW, Planner, TimedSegment

READY = "ARMSTATE_SERVOING_READY"
SPEEDS = {"normal": NORMAL, "slow": SLOW}
FK_CHECK_MM = 6.0
CAM_WAIT_S = 10.0


class Abort(Exception):
    """A failed check: outcome fail, return home, exit 2."""


class StopRequested(Exception):
    """Operator stop (Ctrl+C, SIGTERM, SIGHUP): record the stop, return home, exit 130."""


def r3(a, n=3):
    return np.round(np.asarray(a, float), n).tolist()


def speed_of(speed):
    """'normal' | 'slow' | a speed dict -> speed dict."""
    if isinstance(speed, dict):
        return speed
    try:
        return SPEEDS[str(speed).lower()]
    except KeyError:
        raise ValueError(f"speed must be 'normal' or 'slow', got {speed!r}") from None


class Summary:
    """summary.json of the attempt, rewritten on every change."""

    def __init__(self, path):
        self.p, self.d = path, {}

    def set(self, **kw):
        self.d.update(kw); self.write()

    def write(self):
        try:
            with open(self.p, "w") as f:
                f.write("{\n" + ",\n".join(f'"{k}": {json.dumps(v, default=self._js)}' for k, v in self.d.items()) + "\n}\n")
        except Exception as e:
            print("summary write failed:", e)

    @staticmethod
    def _js(o):
        return np.round(o, 4).tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating) else str(o)


class RobotSession:
    """Context manager; exit codes 0 success, 2 fail, 130 operator stop, 3 exception (raised as SystemExit)."""

    def __init__(self, program="execute", mode="collect", attempt_dir=None, record_dir=None, dry=None, view_port=8090,
                 watchdog=None, exit_on_close=True, kin=None):
        self.program, self.mode = program, mode
        self.A = os.path.abspath(attempt_dir or os.environ.get("RLPJ_ATTEMPT_DIR")
                                 or str(config.RUNS / "manual" / time.strftime("%Y%m%d_%H%M%S")))
        self.record_dir = record_dir if record_dir is not None else os.environ.get("RLPJ_RECORD_DIR") or None
        self.dry = bool(dry) or not kortex.motion_allowed()
        self.use_watchdog = (not self.dry) if watchdog is None else bool(watchdog)
        self.view_port, self.exit_on_close = view_port, exit_on_close
        self.kin = kin or Kin.load()
        self.S = Summary(os.path.join(self.A, "summary.json"))
        self.stop = dict(flag=False, hard=False, sig=None, t=None)
        self.returning = False
        self.outcome = "fail"
        self.commit = ""
        self.phases, self.phase_idx = ["start"], 0
        self.seg_log, self.nseg = [], 0
        self.cur_seg = None
        self.grip_frac, self.grip_cmd_sent = 0.0, False
        self.table_z = calib.table_z()
        self.workspace = self._workspace()
        self.q_home = self.q_sim = None
        self.side = self.wrist = self.view = self.robot = self.planner = self.executor = None
        self.wd = self.recorder = None
        self._replay, self._replay_key = None, None
        self._old_handlers = {}
        self.exit_code = None
        self.started = False
        os.makedirs(os.path.join(self.A, "plans"), exist_ok=True)

    # ------------------------------------------------------------------ context manager
    def __enter__(self):
        try:
            self.start()
        except BaseException as e:
            self._finish(e)
            raise
        return self

    def __exit__(self, et, ev, tb):
        self._finish(ev)
        return True

    # ------------------------------------------------------------------ infrastructure
    def log(self, m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        try:
            with open(os.path.join(self.A, "log.txt"), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if self.view:
            self.view.log(m)

    def _on_signal(self, signum, frame):
        if self.stop["flag"]:
            self.stop["hard"] = True
        name = signal.Signals(signum).name
        self.stop.update(flag=True, sig=name, t=time.time())
        self.log(f"{name} received: stopping ({'hard' if self.stop['hard'] else 'after the current command'})")

    def stop_flag(self):
        return self.stop["hard"] if self.returning else self.stop["flag"]

    def _check_stop(self):
        if self.stop["flag"] and not self.returning:
            raise StopRequested(f"{self.stop['sig']} received")

    def hb(self, state):
        if self.wd is not None:
            try:
                self.wd.stdin.write(f"HB {state} {self.grip_frac:.2f}\n".encode()); self.wd.stdin.flush()
            except Exception:
                pass

    def _on_tick(self, seg, last):
        fb = last["fb"]
        self.view.set(segment=seg.name, progress=last["progress"], dev_deg=last["dev_deg"], dev_mm=last["dev_mm"])
        self.view.set_robot(fb["q"], fb["pose"], fb["gripper"])

    def poll_view(self):
        """Refresh the view's robot state outside control loops (UDP read, ~0.3 ms)."""
        try:
            fb = self.feedback(); self.view.set_robot(fb["q"], fb["pose"], fb["gripper"]); return fb
        except Exception:
            return None

    def _workspace(self):
        """Workspace for tabletop(): task yaml 'workspace' if present, else the layout ranges padded by 8 cm."""
        try:
            task = config.load().get("task", {})
        except Exception:
            task = {}
        if "workspace" in task:
            return task["workspace"]
        lay = task.get("layout", {}) or {}
        rng = [v for v in lay.values() if isinstance(v, dict) and "x" in v and "y" in v]
        if rng:
            return dict(x=[min(v["x"][0] for v in rng) - 0.08, max(v["x"][1] for v in rng) + 0.08],
                        y=[min(v["y"][0] for v in rng) - 0.08, max(v["y"][1] for v in rng) + 0.08],
                        radius=list(lay.get("radius", [0.25, 0.65])))
        return dict(x=[0.25, 0.70], y=[-0.40, 0.25], radius=[0.25, 0.65])

    def _git_commit(self):
        try:
            r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.getcwd(), capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                self.commit = r.stdout.strip()
                d = subprocess.run(["git", "status", "--porcelain"], cwd=os.getcwd(), capture_output=True, text=True, timeout=5).stdout.strip()
                print(f"COMMIT {self.commit}{' (DIRTY)' if d else ''}", flush=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ start
    def start(self):
        from .liveview import LiveView
        self._git_commit()
        self.S.set(claimed_outcome="aborted", program=self.program, mode=self.mode, dry=self.dry, commit=self.commit[:10],
                   command=" ".join(sys.argv), step="start")
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                self._old_handlers[s] = signal.signal(s, self._on_signal)
            except (ValueError, OSError):
                pass
        self.side = cams.SideCam(); self.wrist = cams.WristCam()
        self.view = LiveView(self.A, self.side, self.wrist, self.kin, port=self.view_port, program=self.program).start()
        self.robot = kortex.Robot(dry=self.dry)
        self.planner = Planner(self.kin, oracle=self.robot.fk_oracle, log=self.log)
        self.executor = Executor(self.robot, self.kin, stop_flag=self.stop_flag, on_tick=self._on_tick, heartbeat=self.hb, log=self.log)
        self.log(f"session {self.program}: attempt {self.A} dry={self.dry} record={self.record_dir}")

        # ---- start state
        st = self.robot.arm_state(); t0 = time.time()
        while st != READY and time.time() - t0 < 30 and not self.stop["flag"]:
            if int(time.time() - t0) % 5 == 0:
                self.log(f"waiting for {READY} (now {st})")
            time.sleep(0.5); st = self.robot.arm_state()
        fb = self.robot.feedback()
        q = wrap180(fb["q"])
        self.q_home, self.q_sim = q.copy(), q.copy()
        self.view.set_robot(fb["q"], fb["pose"], fb["gripper"])
        self.S.set(ready_wait_s=round(time.time() - t0, 1),
                   start=dict(arm_state=st, q=r3(fb["q"], 2), tool_pose=r3(fb["pose"], 4), gripper=round(float(fb["gripper"]), 1)))
        self.log(f"arm state {st}, joints {r3(fb['q'], 2)}, gripper {fb['gripper']:.1f}")
        fk_err = float(np.linalg.norm(self.kin.fk(q)[:3, 3] - fb["pose"][:3]) * 1000)
        self.S.set(fk_check_mm=round(fk_err, 2))
        self.log(f"model FK vs measured tool_pose at start: {fk_err:.1f} mm")
        if st != READY:
            raise Abort(f"arm state {st} (need {READY})")
        if fk_err > FK_CHECK_MM:
            raise Abort(f"model FK disagrees with measured tool pose by {fk_err:.1f} mm (run python -m rlpj.robot.fit_kin)")
        if self.use_watchdog:
            self.wd = subprocess.Popen([sys.executable, "-m", "rlpj.robot.watchdog", self.A, str(os.getpid())] + (["dry"] if self.dry else []),
                                       cwd=str(config.ROOT), stdin=subprocess.PIPE, start_new_session=True)
            self.wd.stdin.write(("HOME " + " ".join(f"{v:.4f}" for v in self.q_home) + "\n").encode()); self.wd.stdin.flush()
            self.hb("idle")

        # ---- cameras
        t0 = time.time()
        while (self.side.latest()[0] is None or self.wrist.latest()[0] is None) and time.time() - t0 < CAM_WAIT_S:
            time.sleep(0.1)
        if self.side.latest()[0] is None:
            raise Abort("side camera gives no frames")
        if self.wrist.latest()[0] is None:
            raise Abort("wrist camera gives no frames")
        if self.record_dir:
            from .recorder import EpisodeRecorder
            self.recorder = EpisodeRecorder(self, self.record_dir).start()
            self.recorder.snapshot("before")
        self.started = True

    # ------------------------------------------------------------------ program API
    def phase(self, name):
        """Named program phase: recorded per tick, shown in the live view, written to summary.json."""
        self._check_stop()
        self.phases.append(str(name)); self.phase_idx = len(self.phases) - 1
        self.S.set(step=name)
        if self.view:
            self.view.set(phase=name, segment="-", progress=0.0)
        self.log(f"== phase: {name}")

    def feedback(self):
        """dict(q, dq, pose, gripper, state, t); in a dry run q/pose/gripper come from the simulated state."""
        fb = self.robot.feedback()
        if self.dry and self.q_sim is not None:
            fb = dict(fb, q=self.q_sim.copy(), dq=np.zeros(6), pose=calib.T_to_pose(self.kin.fk(self.q_sim)),
                      gripper=100.0 * self.grip_frac if self.grip_cmd_sent else fb["gripper"], sim=True)
        return fb

    def tabletop(self, workspace=None, n=5):
        """Blobs on the table from a median side depth (rlpj.perception.tabletop.cloud + detect), base frame."""
        self._check_stop()
        try:
            from rlpj.perception import tabletop as tt
        except ImportError as e:
            raise RuntimeError(f"rlpj.perception.tabletop is not available ({e}); tabletop() needs cloud() and detect()") from e
        P = tt.cloud(self.side)
        blobs = tt.detect(P, workspace or self.workspace)
        objs = []
        for i, b in enumerate(blobs):
            try:
                ext = b.get("extent", 0.06)
                r = float(np.max(ext)) / 2 if np.ndim(ext) else float(ext) / 2
                objs.append(dict(xy=list(map(float, b["xy"][:2])), z_top=float(b["z_top"]), r=r, label=f"blob{i}"))
            except Exception:
                pass
        if objs:
            self.view.set_scene(objects=objs)
        self.log(f"tabletop: {len(blobs)} blobs " + str([(r3(b.get('xy', []), 3), round(float(b.get('z_top', 0)), 3)) for b in blobs])[:300])
        return blobs

    def decide(self, m):
        self.log("DECISION: " + m)
        if self.view:
            self.view.decide(m)
        self.S.d.setdefault("decisions", []).append(str(m)); self.S.write()

    def move_joint(self, name, q_goal, speed="normal"):
        """Quintic joint move from the measured joints to q_goal (deg), checked in the model first."""
        self._check_stop()
        q_goal = np.asarray(q_goal, float)
        q0 = self.measured_q(q_goal)
        self._execute(self.planner.joint_move(name, q0, q_goal, speed_of(speed)))

    def move_line(self, name, T_goal, speed="slow", tip_margin=0.008):
        """Straight tool line from the current pose to T_goal (4x4, base frame); plan.tool_down_T(xyz, yaw) helps."""
        self._check_stop()
        q0 = self.measured_q(self.last_ref())
        self._execute(self.planner.cart_line(name, q0, np.asarray(T_goal, float), speed_of(speed)), tip_margin=tip_margin)

    def follow(self, name, q_targets, dt, speed="normal", max_radius=0.65, tip_margin=0.008):
        """Stream joint targets (K,6) deg spaced dt s (policy chunk) through the tracker after one model check."""
        self._check_stop()
        q0 = self.measured_q(np.asarray(q_targets[0], float))
        path = [q0]
        for q in np.asarray(q_targets, float):
            path.append(unwrap_near(q, path[-1]))
        seg = TimedSegment(name, path, dt, speed_of(speed), self.kin)
        self._execute(seg, tip_margin=tip_margin, max_radius=max_radius, settle=(1.0, 0.0))

    def gripper(self, value, what=""):
        """0 open .. 1 closed; waits until the reading settles; returns the 0..100 reading."""
        self._check_stop()
        value = float(np.clip(value, 0, 1))
        self.log(f"gripper -> {value:.2f} ({what})")
        if self.dry:
            self.grip_frac, self.grip_cmd_sent = value, True
            self.robot.gripper_cmd = value
            return 100.0 * value
        g = self.executor.gripper(value)
        self.grip_frac, self.grip_cmd_sent = value, True
        self.poll_view()
        if self.stop["flag"]:
            raise StopRequested("stop during gripper motion")
        self.log(f"gripper reading {g:.1f}")
        return g

    def home(self, name="home", speed="normal"):
        """Joint move back to the start joints."""
        self.move_joint(name, self.q_home, speed)

    def summary(self, **kw):
        self.S.set(**kw)

    def require(self, cond, message):
        if not cond:
            raise Abort(message)

    def claim(self, outcome):
        if outcome not in ("success", "fail"):
            raise ValueError("claim must be 'success' or 'fail'")
        self.outcome = outcome
        self.S.set(claimed_outcome=outcome)
        self.log(f"CLAIM {outcome}")

    # ------------------------------------------------------------------ motion internals
    def measured_q(self, ref):
        if self.dry:
            return unwrap_near(self.q_sim, ref)
        return unwrap_near(self.robot.feedback()["q"], ref)

    def last_ref(self):
        return self.cur_seg.q[-1] if self.cur_seg is not None else self.q_home

    def _replay_for(self):
        """MuJoCo replay with the scene objects as cylinders; rebuilt when the objects change."""
        objs = list(self.view.objects) if self.view else []
        key = tuple((round(float(o["xy"][0]), 3), round(float(o["xy"][1]), 3), round(float(o.get("z_top", 0)), 3)) for o in objs)
        if self._replay is None or key != self._replay_key:
            if self._replay is not None:
                self._replay.close()
            cups = [(k[0], k[1], max(k[2] - calib.table_z(), 0.01), "0.2 0.8 0.2 1") for k in key]
            self._replay, self._replay_key = sim.Replay(self.kin, cups), key
        return self._replay

    def _execute(self, seg, tip_margin=0.008, max_radius=0.80, settle=None):
        self.nseg += 1
        name = f"{self.nseg:02d}_{seg.name}"
        ok, rep = sim.check(seg.q, self.kin, seg.speed, gripper_frac=self.grip_frac, tip_margin=tip_margin,
                            table_z=max(self.table_z, calib.table_z()), max_radius=max_radius)
        np.savez_compressed(os.path.join(self.A, "plans", name + ".npz"), t=seg.t, q=seg.q, qd=seg.qd, path=seg.path)
        with open(os.path.join(self.A, "plans", name + ".json"), "w") as f:
            json.dump(dict(name=seg.name, speed=seg.speed, T=seg.T, check=rep, goal=getattr(seg, "goal_info", {}),
                           q0=r3(seg.q[0], 2), q1=r3(seg.q[-1], 2)), f, default=str)
        try:
            frames = self._replay_for().video(os.path.join(self.A, "plans", name + ".mp4"), seg.q, [self.grip_frac] * len(seg.q), label=name)
            if frames:
                cv2.imwrite(os.path.join(self.A, "plans", name + "_end.jpg"), frames[-1])
        except Exception as e:
            self.log(f"replay video failed: {e}")
        self.log(f"plan {name}: T={seg.T:.1f}s check ok={ok} tip>{rep['min_tip_above_table']:.3f} vmax={rep['max_joint_speed']:.1f}deg/s lin={rep['max_lin_speed']:.3f}m/s")
        if not ok:
            raise Abort(f"model check failed for {name}: {rep}")
        self.cur_seg = seg
        self.view.set_scene(path=seg.tool)
        if self.dry:
            self.q_sim = seg.q[-1].copy()
            self.seg_log.append([seg.name, "dry", round(seg.T, 1)])
            self.S.set(segments=self.seg_log)
            self.view.set(segment=seg.name, progress=1.0)
            self.poll_view()
            return
        q_meas = self.measured_q(seg.q[0])
        if np.abs(q_meas - seg.q[0]).max() > 0.5:
            raise Abort(f"{name}: measured joints differ from plan start by {np.abs(q_meas - seg.q[0]).max():.2f} deg")
        r = self.executor.run(seg, **(dict(settle_tol=settle[0], settle_time=settle[1]) if settle else {}))
        self.seg_log.append([seg.name, r["ok"], r["max_dev_deg"], r["max_dev_mm"], r["t_exec"], r["rate_hz"]])
        self.S.set(segments=self.seg_log)
        self.log(f"exec {name}: {r}")
        if not r["ok"]:
            if self.stop["flag"]:
                raise StopRequested(r["reason"])
            raise Abort(f"{name}: {r['reason']}")
        self.poll_view()

    # ------------------------------------------------------------------ stop / failure handling
    def record_stop(self, kind, why):
        """Write the stop moment into summary.json and save the last frames (<= 640 px)."""
        rec = dict(kind=kind, why=str(why)[:300], step=self.phases[self.phase_idx],
                   segment=self.cur_seg.name if self.cur_seg is not None else None)
        try:
            last = self.executor.last if (self.executor and not self.dry) else {}
            if last:
                rec.update(progress=round(last["progress"], 3), t=round(last["t"], 2), q_meas=r3(last["q"], 2), q_plan=r3(last["q_ref"], 2),
                           tool_meas=r3(last["fb"]["pose"], 4), tool_plan_xyz=r3(self.kin.fk(last["q_ref"])[:3, 3], 4),
                           dev_deg=round(last["dev_deg"], 2), dev_mm=round(last["dev_mm"], 1))
            fb = self.robot.feedback() if self.robot else None
            if fb:
                rec.update(q_now=r3(fb["q"], 2), gripper=round(float(fb["gripper"]), 1))
        except Exception as e:
            rec["err"] = str(e)
        if self.stop["sig"]:
            rec["signal"] = self.stop["sig"]
        self.S.set(stop=rec, first_failed_check=str(why)[:300])
        try:
            if self.side and self.side.latest()[0] is not None:
                cv2.imwrite(os.path.join(self.A, "stop_side.jpg"), cv2.resize(self.side.latest()[0], (640, 360)))
            if self.wrist and self.wrist.latest()[0] is not None:
                cv2.imwrite(os.path.join(self.A, "stop_wrist.jpg"), cv2.resize(self.wrist.latest()[0], (640, 360)))
            if self.view:
                self.view.snapshot("stop_overlay")
        except Exception:
            pass
        self.log(f"STOP RECORD: {rec}")

    def go_home(self):
        """Halt, then return to the start joints along a model-checked path (lift first if low)."""
        if self.dry or self.robot is None or self.q_home is None or self.executor is None:
            return
        self.returning = True
        try:
            self.executor.halt()
            time.sleep(0.3)
            q_now = unwrap_near(self.robot.feedback()["q"], self.last_ref())
            if np.abs(q_now - unwrap_near(self.q_home, q_now)).max() < 0.3:
                self.S.set(return_home="already at default pose"); return
            self.phase("return_home")
            segs, reps, ok = plan_return(self.planner, q_now, self.q_home, self.grip_frac)
            self.log(f"return plan ok={ok}: {[(r.get('segment'), r.get('ok')) for r in reps if isinstance(r, dict)]}")
            if not ok:
                self.S.set(return_home=f"return path failed model checks: {reps}"); return
            res = []
            for s in segs:
                try:
                    self._execute(s, tip_margin=-0.01 if s.name == "return_lift" else 0.008)
                    res.append([s.name, True])
                except Exception as e:
                    res.append([s.name, str(e)[:120]]); break
            self.S.set(return_home=res)
        except Exception as e:
            self.S.set(return_home=f"error {e!r}")
        finally:
            self.returning = False

    def _finish(self, exc):
        """Outcome, stop record, return home, close; raises SystemExit(code) unless exit_on_close is False."""
        code = 0
        try:
            if exc is None and self.stop["flag"]:
                exc = StopRequested("stop signal")
            if exc is None:
                self.S.set(claimed_outcome=self.outcome)
                code = 0 if self.outcome == "success" else 2
            elif isinstance(exc, (StopRequested, KeyboardInterrupt)):
                self.S.set(claimed_outcome="aborted")
                self.record_stop("operator stop", f"{self.stop['sig']}: {exc}"); code = 130
            elif isinstance(exc, Abort):
                self.S.set(claimed_outcome="fail")
                self.record_stop("failed check", exc); code = 2
            else:
                traceback.print_exception(exc)
                self.S.set(claimed_outcome="fail")
                self.record_stop("exception", repr(exc)); code = 3
            if self.recorder:
                self.recorder.snapshot("after")
            self.go_home()
        except Exception:
            traceback.print_exc(); code = code or 3
        finally:
            self.S.set(segments=self.seg_log, exit_code=code)
            self.close()
        print(f"summary: {self.S.p} outcome: {self.S.d.get('claimed_outcome')} exit {code}", flush=True)
        self.exit_code = code
        if self.exit_on_close:
            raise SystemExit(code)
        return code

    def close(self):
        try:
            if self.robot is not None and not self.dry:
                t0 = time.time(); st = self.robot.arm_state()
                while st != READY and time.time() - t0 < 10:
                    time.sleep(0.5); st = self.robot.arm_state()
                self.S.set(end_arm_state=[st, round(time.time() - t0, 1)])
        except Exception:
            pass
        if self.wd is not None:
            try:
                self.wd.stdin.write(b"BYE\n"); self.wd.stdin.flush(); self.wd.wait(3)
            except Exception:
                pass
            self.wd = None
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception:
                pass
            self.recorder = None
        for o in (self.view, self.side, self.wrist, self._replay, self.robot):
            try:
                o and o.close()
            except Exception:
                pass
        self.view = self.side = self.wrist = self._replay = self.robot = None
        for s, h in self._old_handlers.items():
            try:
                signal.signal(s, h)
            except Exception:
                pass
        self._old_handlers = {}
