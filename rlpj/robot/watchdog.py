"""Watchdog process: if the robot program dies (pipe EOF without BYE) or stops heartbeating while the arm moves,
it stops the arm and returns it to the default pose along a path checked in the model.
Started by the session: python -m rlpj.robot.watchdog <attempt_dir> <main_pid> [dry]   (stdin = pipe from main).
Protocol on stdin, one line each: HOME q1..q6 | HB <idle|moving> <grip_frac> | BYE."""
import json
import os
import select
import signal
import sys
import time

import numpy as np

HB_TIMEOUT = 0.6   # s without a heartbeat while moving before the takeover


class Protocol:
    """State fed by the main program's lines."""

    def __init__(self):
        self.q_home, self.state, self.grip_frac, self.bye = None, "idle", 0.0, False
        self.last_hb = time.time()
        self.buf = b""

    def feed(self, chunk):
        """Consume raw bytes; returns the parsed commands."""
        self.buf += chunk
        cmds = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            parts = line.decode(errors="replace").split()
            if not parts:
                continue
            cmds.append(parts[0])
            if parts[0] == "HOME" and len(parts) >= 7:
                self.q_home = np.array([float(v) for v in parts[1:7]])
            elif parts[0] == "HB" and len(parts) >= 2:
                self.state, self.last_hb = parts[1], time.time()
                if len(parts) > 2:
                    self.grip_frac = float(parts[2])
            elif parts[0] == "BYE":
                self.bye = True
        return cmds

    def starved(self, now=None):
        return self.state == "moving" and (now or time.time()) - self.last_hb > HB_TIMEOUT


def takeover(robot, kin, q_home, grip_frac, reason, log, stop_flag):
    """Halt, then return home along a checked path; returns a dict for summary.json."""
    from .executor import Executor
    from .home import plan_return
    from .kin import unwrap_near
    from .plan import Planner
    ex = Executor(robot, kin, stop_flag=stop_flag, log=log)
    ex.halt()
    time.sleep(0.5)
    out = dict(watchdog_takeover=reason)
    if q_home is None:
        log("no home pose known; arm stopped, not moving"); out["result"] = "stopped, no home"
        return out
    q_now = unwrap_near(robot.feedback()["q"], q_home)
    planner = Planner(kin, oracle=robot.fk_oracle, log=log)
    try:
        segs, reps, ok = plan_return(planner, q_now, q_home, grip_frac)
    except Exception as e:
        segs, reps, ok = [], [str(e)], False
    log(f"return plan ok={ok} checks={reps}")
    out["return_checks_ok"] = ok
    if not ok:
        out["result"] = "stopped; return path failed checks"
        return out
    res = []
    for s in segs:
        q_now = unwrap_near(robot.feedback()["q"], s.q[0])
        if np.abs(q_now - s.q[0]).max() > 0.5:   # drifted: do not send a plan that starts elsewhere
            break
        rr = ex.run(s); res.append(rr); log(f"segment {rr}")
        if not rr["ok"]:
            break
    out["return"] = res
    out["result"] = "returned home" if res and all(x["ok"] for x in res) and len(res) == len(segs) else "return incomplete"
    return out


def write_summary(attempt_dir, out, log):
    """Leave a record in summary.json (the main program is dead)."""
    try:
        sp = os.path.join(attempt_dir, "summary.json")
        s = json.load(open(sp)) if os.path.exists(sp) else {}
        s["claimed_outcome"] = "aborted"; s["watchdog"] = out
        with open(sp, "w") as f:
            f.write("{\n" + ",\n".join(f'"{k}": {json.dumps(v, default=str)[:400]}' for k, v in s.items()) + "\n}\n")
    except Exception as e:
        log(f"summary write failed {e}")


def main(argv=None, robot=None, fd=None, kin=None):
    """argv: <attempt_dir> <main_pid> [dry]; robot/fd/kin are injectable for offline tests."""
    argv = sys.argv[1:] if argv is None else argv
    A, main_pid = argv[0], int(argv[1])
    dry = len(argv) > 2 and argv[2] == "dry"
    logf = open(os.path.join(A, "watchdog_log.txt"), "a", buffering=1)

    def log(m):
        logf.write(f"{time.strftime('%H:%M:%S')} {m}\n")

    term = {"flag": False}
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # the operator's Ctrl+C is for the main program
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *a: term.__setitem__("flag", True))
    except ValueError:
        pass   # not the main thread (tests)

    if robot is None:
        from .kortex import Robot
        robot = Robot(dry=dry)
    if kin is None:
        from .kin import Kin
        kin = Kin.load()
    log(f"watchdog up, main pid {main_pid}, dry={dry}")
    p = Protocol()
    fd = sys.stdin.fileno() if fd is None else fd
    reason, last_ka = None, 0.0
    while True:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            chunk = os.read(fd, 4096)
            if not chunk:
                reason = "main program pipe closed without BYE (crash / kill)"; break
            for c in p.feed(chunk):
                if c == "HOME":
                    log(f"home {p.q_home.round(2).tolist()}")
            if p.bye:
                log("BYE received, exiting"); robot.close(); return 0
        if p.starved():
            reason = f"no heartbeat for {time.time() - p.last_hb:.2f}s while moving"
            try:
                os.kill(main_pid, signal.SIGKILL)
            except Exception:
                pass
            break
        if time.time() - last_ka > 1.0:
            robot.keepalive(); last_ka = time.time()
        if term["flag"]:
            log("SIGTERM, exiting"); robot.close(); return 0
    log(f"TAKEOVER: {reason}")
    try:
        out = takeover(robot, kin, p.q_home, p.grip_frac, reason, log, lambda: term["flag"])
    except Exception as e:
        out = dict(watchdog_takeover=reason, result=f"takeover error {e!r}")
    log(json.dumps(out, default=str))
    write_summary(A, out, log)
    robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
