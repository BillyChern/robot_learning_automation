"""Policy rollout robot program (4090; started by the orchestrator like execute.py, recorded like an execute episode):

    python -m rlpj.vla.rollout [--host H] [--port P] [--instruction TEXT] [--max-seconds 150] [--chunk 10] [--replan-every 10]

Cycle: latest side + wrist frames and feedback -> obs -> policy -> (K,7) absolute joint targets (rad) + gripper ->
degrees unwrapped near the current joints -> piecewise-linear joint trajectory at the recording rate, time-stretched
to the normal speed class (20 deg/s, 0.10 m/s) -> rlpj.robot.sim.check -> streamed by the session's executor (abort on
deviation) -> gripper command when the predicted gripper crosses 0.5 (hysteresis 0.4/0.6). Stops at --max-seconds or
Ctrl+C, never claims success (the orchestrator's verdict decides) and writes summary numbers. Without RLPJ_MOTION=1
it infers and checks but does not move (dry run, --dry-chunks cycles)."""
import argparse
import logging
import os
import sys
import time

import numpy as np

from rlpj.robot.kin import unwrap_near
from rlpj.robot.plan import DT, MARGIN, NORMAL
from .convert import wrap_pi
from .policy_client import PolicyClient, make_obs

log = logging.getLogger("rollout")


def state_from_feedback(q_deg, gripper_reading):
    """Policy state (7,) float32 from feedback: joints rad wrapped to (-pi, pi], gripper reading / 100 (NaN -> 0)."""
    g = float(gripper_reading) if np.isfinite(gripper_reading) else 0.0
    return np.concatenate([wrap_pi(np.deg2rad(np.asarray(q_deg, float))), [np.clip(g / 100.0, 0, 1)]]).astype(np.float32)


def actions_to_targets_deg(actions_rad, q_now_deg):
    """(K,6) absolute joint targets rad -> deg, each unwrapped near the previous one (first near the current joints)."""
    deg = np.rad2deg(np.asarray(actions_rad, float).reshape(-1, 6))
    out = np.empty_like(deg)
    ref = np.asarray(q_now_deg, float)
    for k in range(len(deg)):
        out[k] = unwrap_near(deg[k], ref)
        ref = out[k]
    return out


class PolicySegment:
    """Executor-compatible segment: piecewise-linear through the knots at dt * scale, sampled every DT."""

    def __init__(self, name, knots, dt, scale, speed, kin):
        self.name, self.speed, self.kind = name, speed, "policy"
        self.path = np.asarray(knots, float)
        self.scale = float(scale)
        kt = np.arange(len(self.path)) * dt * self.scale
        self.T = float(kt[-1])
        n = max(int(np.ceil(self.T / DT)), 1)
        self.t = np.arange(n + 1) * DT
        self.t[-1] = self.T
        self.q = np.stack([np.interp(self.t, kt, self.path[:, j]) for j in range(6)], 1)
        self.qd = np.gradient(self.q, DT, axis=0) if len(self.t) > 1 else np.zeros_like(self.q)
        self.tool = np.array([kin.fk(q)[:3, 3] for q in self.q[:: max(1, len(self.q) // 60)]])

    def ref(self, t):
        """Reference joints and velocities at time t (clamped), like plan.Segment.ref."""
        if t <= 0:
            return self.q[0], np.zeros(6)
        if t >= self.T:
            return self.q[-1], np.zeros(6)
        k = t / DT
        i = min(int(k), len(self.q) - 2)
        f = k - i
        return self.q[i] * (1 - f) + self.q[i + 1] * f, self.qd[i] * (1 - f) + self.qd[i + 1] * f


def make_policy_segment(kin, q0_deg, targets_deg, dt, speed=NORMAL, name="policy"):
    """Trajectory q0 -> targets (K,6) deg, one knot per dt; time stretched so joint / tool speeds stay within speed."""
    q0 = np.asarray(q0_deg, float).reshape(6)
    knots = np.vstack([q0, np.asarray(targets_deg, float).reshape(-1, 6)])
    vj = np.abs(np.diff(knots, axis=0)).max() / dt if len(knots) > 1 else 0.0
    P = np.array([kin.fk(q)[:3, 3] for q in knots])
    vl = np.linalg.norm(np.diff(P, axis=0), axis=1).max() / dt if len(knots) > 1 else 0.0
    scale = max(1.0, vj / (speed["jv"] * MARGIN), vl / (speed["lin"] * MARGIN))
    return PolicySegment(name, knots, dt, scale, speed, kin)


class GripperLatch:
    """Open/closed state with hysteresis; update() returns the value to command when the state flips, else None."""

    def __init__(self, closed=False, lo=0.4, hi=0.6):
        self.closed, self.lo, self.hi = bool(closed), lo, hi

    def update(self, g):
        g = float(np.clip(g, 0.0, 1.0))
        if not self.closed and g > self.hi:
            self.closed = True
            return g
        if self.closed and g < self.lo:
            self.closed = False
            return g
        return None


def _stream(s, seg, settle_tol):
    """Run a checked segment through the session (run_segment / stream / executor.run); returns the executor's dict."""
    for name in ("run_segment", "stream"):
        fn = getattr(s, name, None)
        if callable(fn):
            try:
                return fn(seg, settle_tol=settle_tol)
            except TypeError:
                return fn(seg)
    ex = getattr(s, "executor", None)
    if ex is not None:
        return ex.run(seg, settle_tol=settle_tol)
    raise RuntimeError("RobotSession has no run_segment/stream method and no executor to stream a policy segment")


def _short(rep):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()}


def main(argv=None):
    from rlpj import config
    cfg = config.load()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default=cfg["hosts"]["policy_server"])
    p.add_argument("--port", type=int, default=cfg["hosts"]["policy_port"])
    p.add_argument("--instruction", default=cfg["task"]["instruction"])
    p.add_argument("--max-seconds", type=float, default=cfg["evaluation"]["max_policy_seconds"])
    p.add_argument("--chunk", type=int, default=10, help="actions taken from each policy call")
    p.add_argument("--replan-every", type=int, default=10, help="actions executed before the next call")
    p.add_argument("--fps", type=float, default=cfg["recording"]["fps"], help="recording rate = action rate")
    p.add_argument("--tip-margin", type=float, default=0.008,
                   help="fingertip clearance above the table (m); the session re-checks every segment at 0.008")
    p.add_argument("--settle-tol", type=float, default=0.5, help="deg; chunk ends when within this of its target")
    p.add_argument("--dry-chunks", type=int, default=3, help="cycles to run without RLPJ_MOTION=1")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from rlpj.robot import calib, sim
    from rlpj.robot.kin import Kin
    from rlpj.robot.session import RobotSession

    motion = os.environ.get("RLPJ_MOTION") == "1"
    st = dict(chunks=0, steps=0, infer_ms=[], max_dev_deg=0.0, max_dev_mm=0.0, gripper_events=0, check_fail=0,
              stop_reason="", time_scale_max=1.0)
    client = PolicyClient(a.host, a.port, log=log.info)
    t_start = time.time()
    with RobotSession(program="rollout") as s:
        s.phase("policy")
        kin = getattr(s, "kin", None) or Kin.load()
        table_z = calib.table_z()
        fb = s.feedback()
        latch = GripperLatch(closed=np.nan_to_num(fb["gripper"]) / 100.0 > 0.5)
        s.decide(f"policy {a.host}:{a.port} prompt={a.instruction!r} chunk={a.chunk} replan={a.replan_every} "
                 f"metadata={client.metadata} motion={motion}")
        t_end = t_start + a.max_seconds
        try:
            while True:
                if time.time() >= t_end:
                    st["stop_reason"] = "time limit"
                    break
                if not motion and st["chunks"] >= a.dry_chunks:
                    st["stop_reason"] = "dry run done"
                    break
                side, _, _ = s.side.latest()
                wrist, _ = s.wrist.latest()
                fb = s.feedback()
                s.require(side is not None and wrist is not None, "no camera frame")
                obs = make_obs(side, wrist, state_from_feedback(fb["q"], fb["gripper"]), a.instruction)
                out = client.infer(obs)
                st["infer_ms"].append(client.last_ms)
                actions = np.asarray(out["actions"], float)
                n_exec = max(1, min(a.replan_every, a.chunk, len(actions)))
                targets = actions_to_targets_deg(actions[:n_exec, :6], fb["q"])
                seg = make_policy_segment(kin, fb["q"], targets, 1.0 / a.fps, NORMAL, name=f"policy_{st['chunks']}")
                st["time_scale_max"] = max(st["time_scale_max"], seg.scale)
                grip = float(np.clip(np.nan_to_num(fb["gripper"]) / 100.0, 0, 1))
                ok, rep = sim.check(seg.q, kin, NORMAL, gripper_frac=grip, table_z=table_z, tip_margin=a.tip_margin)
                if not ok:
                    st["check_fail"] += 1
                    st["stop_reason"] = "model check failed"
                    s.require(False, f"policy chunk {st['chunks']} failed the model check: {_short(rep)}")
                log.info(f"chunk {st['chunks']}: infer {client.last_ms:.0f} ms, T={seg.T:.2f}s x{seg.scale:.2f}, "
                         f"max dq {np.abs(np.diff(seg.path, axis=0)).max():.2f} deg, gripper {actions[n_exec - 1, 6]:.2f}")
                if motion:
                    res = _stream(s, seg, a.settle_tol) or {}
                    st["max_dev_deg"] = max(st["max_dev_deg"], float(res.get("max_dev_deg", 0.0)))
                    st["max_dev_mm"] = max(st["max_dev_mm"], float(res.get("max_dev_mm", 0.0)))
                    if not res.get("ok", True):
                        st["stop_reason"] = f"tracking: {res.get('reason', '')}"
                        s.require(False, f"policy chunk aborted: {res.get('reason', '')}")
                st["chunks"] += 1
                st["steps"] += n_exec
                g_cmd = latch.update(actions[n_exec - 1, 6])
                if g_cmd is not None:
                    st["gripper_events"] += 1
                    if motion:
                        s.gripper(g_cmd, "policy_close" if g_cmd > 0.5 else "policy_open")
                    else:
                        log.info(f"dry run: gripper -> {g_cmd:.2f}")
        finally:
            ms = np.asarray(st["infer_ms"]) if st["infer_ms"] else np.array([np.nan])
            s.summary(policy_server=f"{a.host}:{a.port}", instruction=a.instruction, chunks=st["chunks"],
                      steps=st["steps"], infer_ms_mean=round(float(np.nanmean(ms)), 1),
                      infer_ms_max=round(float(np.nanmax(ms)), 1), max_dev_deg=round(st["max_dev_deg"], 3),
                      max_dev_mm=round(st["max_dev_mm"], 2), gripper_events=st["gripper_events"],
                      check_fail=st["check_fail"], time_scale_max=round(st["time_scale_max"], 2),
                      stop_reason=st["stop_reason"] or "interrupted", seconds=round(time.time() - t_start, 1))
            client.close()
        s.claim("fail")      # the orchestrator's verdict decides success
    return 0


if __name__ == "__main__":
    sys.exit(main())
