"""Closed-loop execution of a planned joint trajectory with joint-speed commands (feedforward of the planned
velocity + small P correction), with tracking checks. Used by the main program and by the watchdog."""
import time
import numpy as np
from .kin import unwrap_near

KP = 1.5           # 1/s
LOOKAHEAD = 0.04   # s, compensates command latency
DEV_DEG = 1.0
DEV_MM = 10.0
FORCE_SKIP_S = 0.3   # the force baseline is taken over the first ticks of a segment (start transients)


class Executor:
    def __init__(self, robot, kin, stop_flag=lambda: False, on_tick=None, heartbeat=None, log=print):
        self.robot, self.kin = robot, kin
        self.stop_flag, self.on_tick, self.heartbeat, self.log = stop_flag, on_tick, heartbeat, log
        self.last = {}
        self.running = False   # True while run() tracks a segment (the recorder reads last["q_ref"] then)
        self.force_limit = None  # N: stop when the estimated tool force changes by more than this within a segment

    def halt(self):
        for _ in range(2):
            try:
                self.robot.send_joint_speeds([0.0] * 6)
                break
            except Exception as e:
                self.log(f"halt: zero speed failed: {e}")
        try:
            self.robot.stop()
        except Exception as e:
            self.log(f"halt: Stop failed: {e}")

    def run(self, seg, settle_tol=0.15, settle_time=2.0):
        r = dict(segment=seg.name, ok=False, reason="", T_plan=round(seg.T, 2), max_dev_deg=0.0, max_dev_mm=0.0, n=0)
        vmax = seg.speed["jv"] * 1.1 + 1.0
        t0 = time.time(); ts = None; q = None; f0 = None
        self.running = True
        try:
            while True:
                if self.stop_flag():
                    r["reason"] = "stop requested"; break
                fb = self.robot.feedback()
                t = fb["t"] - t0
                qr, _ = seg.ref(t)
                q = unwrap_near(fb["q"], qr)
                e = qr - q
                dev = float(np.abs(e).max())
                dmm = float(np.linalg.norm(self.kin.fk(q)[:3, 3] - self.kin.fk(qr)[:3, 3]) * 1000)
                r["max_dev_deg"] = max(r["max_dev_deg"], dev); r["max_dev_mm"] = max(r["max_dev_mm"], dmm)
                self.last = dict(t=t, q=q, q_ref=qr, dev_deg=dev, dev_mm=dmm, fb=fb, progress=min(t / max(seg.T, 1e-6), 1.0))
                if self.on_tick:
                    try:
                        self.on_tick(seg, self.last)
                    except Exception:
                        pass
                if self.heartbeat:
                    self.heartbeat("moving")
                if fb.get("state") and fb["state"] not in ("ARMSTATE_SERVOING_READY", "ARMSTATE_SERVOING_MANUALLY_CONTROLLED"):
                    r["reason"] = f"arm state {fb['state']}"; break
                f = fb.get("force")
                if self.force_limit and f is not None:
                    if f0 is None or t < FORCE_SKIP_S:
                        f0 = np.asarray(f, float)
                    else:
                        df = float(np.linalg.norm(np.asarray(f, float) - f0))
                        r["max_force_n"] = round(max(r.get("max_force_n", 0.0), df), 1)
                        if df > self.force_limit:
                            r["reason"] = f"contact force {df:.1f} N > {self.force_limit:.0f} N at t={t:.2f}s"; break
                if dev > DEV_DEG or dmm > DEV_MM:
                    r["reason"] = f"tracking deviation {dev:.2f} deg / {dmm:.1f} mm at t={t:.2f}s"; break
                if t >= seg.T:
                    if ts is None:
                        ts = t
                    if dev < settle_tol:
                        r["ok"] = True; break
                    if t - ts > settle_time:
                        r["ok"] = True; r["reason"] = f"settle timeout, final dev {dev:.2f} deg"; break
                _, vff = seg.ref(t + LOOKAHEAD)
                v = np.clip(vff + KP * e, -vmax, vmax)
                self.robot.send_joint_speeds(v)
                r["n"] += 1
        except Exception as ex:
            r["reason"] = f"exception: {ex!r}"
        self.running = False
        self.halt()
        r["t_exec"] = round(time.time() - t0, 2)
        r["max_dev_deg"] = round(r["max_dev_deg"], 3); r["max_dev_mm"] = round(r["max_dev_mm"], 2)
        r["rate_hz"] = round(r["n"] / max(r["t_exec"], 1e-3), 1)
        if q is not None:
            r["final_err_deg"] = round(float(np.abs(seg.q[-1] - q).max()), 3)
        if self.heartbeat:
            self.heartbeat("idle")
        return r

    def gripper(self, value, timeout=5.0):
        """Command the gripper with the arm stopped; wait until the reading settles. Returns final reading."""
        g0 = self.robot.feedback()["gripper"]
        self.robot.gripper_position(value)
        t0 = time.time(); hist = []
        while time.time() - t0 < timeout:
            if self.stop_flag():
                break
            g = self.robot.feedback()["gripper"]; hist.append(g)
            if self.heartbeat:
                self.heartbeat("idle")
            moved = abs(g - g0) > 2.0 or abs(g - 100 * value) < 3.0
            if len(hist) > 6 and max(hist[-6:]) - min(hist[-6:]) < 0.3 and (moved or time.time() - t0 > 1.5):
                break
            time.sleep(0.05)
        return hist[-1] if hist else float("nan")
