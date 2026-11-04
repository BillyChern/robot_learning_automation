"""Run one command as an attempt: env, log tee, Ctrl+C relay with escalation, exit code -> outcome."""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rlpj import config

HOME_WAIT_S = 60     # after the first Ctrl+C: time for the program to stop and return home
KILL_WAIT_S = 15     # after SIGTERM


@dataclass
class AttemptResult:
    exit_code: int
    outcome: str                   # success | fail | aborted | error
    summary: dict                  # summary.json of the attempt ({} if missing)
    duration: float
    stopped_by_operator: bool
    attempt_dir: Path
    stopped_by: str | None = None  # operator | timeout


def _killpg(proc, sig):
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


class _Stopper:
    """SIGINT once (the program homes and exits), SIGTERM after HOME_WAIT_S or a second Ctrl+C, SIGKILL KILL_WAIT_S later."""

    def __init__(self, proc, attempt_dir):
        self.proc, self.dir = proc, attempt_dir
        self.by, self.t_int, self.t_term, self.killed, self.n = None, None, None, False, 0

    def request(self, who):
        self.n += 1
        if self.n == 1:
            self.by, self.t_int = who, time.time()
            (self.dir / "operator_stop").write_text(f"{self.t_int:.0f} {who}\n")
            print(f"\n== stop ({who}): SIGINT sent, waiting up to {HOME_WAIT_S} s for the program to return home "
                  "(Ctrl+C again = SIGTERM)", flush=True)
            _killpg(self.proc, signal.SIGINT)
        elif self.n == 2:
            self.term()
        else:
            self.kill()

    def term(self):
        self.t_term = time.time()
        print(f"== SIGTERM sent; SIGKILL in {KILL_WAIT_S} s", flush=True)
        _killpg(self.proc, signal.SIGTERM)

    def kill(self):
        if not self.killed:
            self.killed = True
            print("== SIGKILL", flush=True)
            _killpg(self.proc, signal.SIGKILL)

    def tick(self, elapsed, timeout_s):
        now = time.time()
        if self.t_int is None:
            if timeout_s and elapsed > timeout_s:
                self.request("timeout")
        elif self.t_term is None:
            if now - self.t_int > HOME_WAIT_S:
                self.term()
        elif now - self.t_term > KILL_WAIT_S:
            self.kill()


def _tee(stream, log):
    for line in iter(stream.readline, b""):
        log.write(line)
        log.flush()
        sys.stdout.write(line.decode(errors="replace"))
        sys.stdout.flush()


def _outcome(code, summary, stopped):
    if stopped or code == 130:
        return "aborted"
    claim = summary.get("claimed_outcome")
    if code in (0, 2):
        return claim if claim in ("success", "fail") else ("success" if code == 0 else "fail")
    return "error"


def _read_json(p):
    try:
        d = json.loads(Path(p).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def run_attempt(workspace, command, attempt_dir, moves_robot, record_dir=None, extra_env=None, timeout_s=None):
    """Run command (bash, cwd=workspace) as one attempt and wait for it, relaying Ctrl+C; returns AttemptResult.
    Env: PYTHONPATH=repo, PATH with the repo venv, RLPJ_ATTEMPT_DIR, RLPJ_MOTION=1 only if moves_robot, RLPJ_RECORD_DIR.
    Writes command.txt, started, ended, exit_code, console.txt (also log.txt when the program wrote none), attempt.json."""
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(config.ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PATH"] = os.pathsep.join([str(config.ROOT / ".venv/bin"), env.get("PATH", "")])
    env["RLPJ_ATTEMPT_DIR"] = str(attempt_dir)
    for k in ("RLPJ_MOTION", "RLPJ_RECORD_DIR", "RLPJ_RESET_TARGETS"):
        env.pop(k, None)
    if moves_robot:
        env["RLPJ_MOTION"] = "1"
    if record_dir:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        env["RLPJ_RECORD_DIR"] = str(record_dir)
    env.update({k: str(v) for k, v in (extra_env or {}).items()})
    (attempt_dir / "command.txt").write_text(command + "\n")
    t0 = time.time()
    (attempt_dir / "started").write_text(f"{t0:.0f}\n")
    print(f"== {attempt_dir.name} [{'ROBOT' if moves_robot else 'dry'}] $ {command}", flush=True)
    log = open(attempt_dir / "console.txt", "wb")
    proc = subprocess.Popen(command, shell=True, executable="/bin/bash", cwd=str(workspace), env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    tee = threading.Thread(target=_tee, args=(proc.stdout, log), daemon=True)
    tee.start()
    stopper = _Stopper(proc, attempt_dir)
    main = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGINT, lambda *_: stopper.request("operator")) if main else None
    try:
        while True:
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                stopper.tick(time.time() - t0, timeout_s)
    finally:
        if main:
            signal.signal(signal.SIGINT, old)
        tee.join(5)
        log.close()
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    t1 = time.time()
    (attempt_dir / "exit_code").write_text(f"{code}\n")
    (attempt_dir / "ended").write_text(f"{t1:.0f}\n")
    if not (attempt_dir / "log.txt").exists():
        shutil.copy(attempt_dir / "console.txt", attempt_dir / "log.txt")
    summary = _read_json(attempt_dir / "summary.json")
    res = AttemptResult(code, _outcome(code, summary, stopper.by is not None), summary, round(t1 - t0, 1),
                        stopper.by == "operator", attempt_dir, stopper.by)
    meta = _read_json(attempt_dir / "attempt.json")
    meta.update(command=command, moves_robot=moves_robot, record_dir=str(record_dir) if record_dir else None,
                exit_code=code, outcome=res.outcome, duration_s=res.duration, stopped_by=stopper.by, started=t0, ended=t1)
    (attempt_dir / "attempt.json").write_text(json.dumps(meta, indent=1))
    print(f"== {attempt_dir.name} finished: exit {code}, outcome {res.outcome}, {res.duration:.0f} s", flush=True)
    return res


def wrist_stream_free(ip, wait_s=45.0):
    """Best effort: True once the Kinova RTSP stream accepts a client (a crashed client keeps its slot ~40 s)."""
    try:
        import cv2
    except ImportError:
        return True
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;3000000")
    t0, said = time.time(), False
    while True:
        cap = cv2.VideoCapture(f"rtsp://{ip}/color", cv2.CAP_FFMPEG)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            if said:
                print(f"== wrist stream free after {time.time() - t0:.0f} s", flush=True)
            return True
        if time.time() - t0 > wait_s:
            print(f"== wrist stream still refused after {wait_s:.0f} s (another client holds it); starting anyway", flush=True)
            return False
        if not said:
            print(f"== wrist stream busy (stale client); waiting up to {wait_s:.0f} s", flush=True)
            said = True
        time.sleep(3)
