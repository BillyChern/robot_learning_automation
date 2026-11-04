"""Headless Claude Code on the agent machine: one `claude -p` process per turn, one structured action back (DESIGN.md 7)."""
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from rlpj import config
from . import schema
from .schema import Action, Usage
from .workspace import Workspace

AGENT_ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
CLAUDE_FLAGS = ["--output-format", "json", "--permission-mode", "acceptEdits", "--permission-prompts", "none",
                "--setting-sources", "project,local", "--strict-mcp-config", "--disable-slash-commands"]
REMOTE_PATH = "$HOME/.local/bin:$PATH"


class AgentError(RuntimeError):
    """A turn failed: ssh error, timeout, no JSON result, is_error from claude, or an invalid action."""


def _tail(text, n=20) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])


def _json_object(text: str):
    """The result object in claude's stdout (whole output, or the last line that is a JSON object)."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                return obj if isinstance(obj, dict) else None
            except ValueError:
                continue
    return None


class AgentRunner:
    """Start and resume one agent's Claude Code session in its workspace; records usage and raw results."""

    def __init__(self, workspace: Workspace, model: str | None = None, effort: str | None = None,
                 max_budget_usd: float | None = None, timeout_s: float | None = None, claude_bin: str | None = None):
        cfg = config.load()["agents"]
        self.ws = workspace
        self.model = model or workspace.session.get("model") or cfg["model"]
        self.effort = effort or workspace.session.get("effort") or cfg["effort"]
        self.max_budget_usd = cfg.get("max_budget_usd") if max_budget_usd is None else max_budget_usd
        self.timeout_s = timeout_s or cfg.get("turn_timeout_s", 3600)
        self.fake = os.environ.get("RLPJ_FAKE_CLAUDE")          # local fake binary: run here, no ssh (tests)
        self.claude_bin = claude_bin or os.environ.get("RLPJ_CLAUDE_BIN") or self.fake or "claude"

    def command(self, session_id: str, resume: bool = False) -> list:
        """The claude argv for one turn; the prompt goes on stdin."""
        argv = [self.claude_bin, "--add-dir", *self.ws.add_dirs, "-p", "--model", self.model, "--effort", self.effort,
                *CLAUDE_FLAGS, "--json-schema", schema.schema_json()]
        if self.max_budget_usd:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        argv += ["--resume", session_id] if resume else ["--session-id", session_id]
        return argv

    def shell_command(self, argv) -> str:
        """What runs on the agent machine: cd into the workspace, agent env, PATH with claude, then claude."""
        env = " ".join(f"{k}={v}" for k, v in AGENT_ENV.items())
        return f"cd {shlex.quote(str(self.ws.path))} && export PATH={REMOTE_PATH} {env} && exec {shlex.join(argv)}"

    def start(self, prompt: str | None = None) -> tuple[Action, Usage]:
        """First turn with a new session id (stored in session.json before the call); prompt defaults to PROMPT.md."""
        if self.ws.session.get("session_id"):
            raise AgentError(f"session {self.ws.session['session_id']} already started; use resume()")
        sid = str(uuid.uuid4())
        self.ws.session.update(session_id=sid, model=self.model, effort=self.effort)
        self.ws.save_session()
        return self._turn(self.command(sid), self.ws.prompt() if prompt is None else prompt, "start")

    def resume(self, message: str) -> tuple[Action, Usage]:
        """Continue the session with a new user message (a digest, an operator note)."""
        sid = self.ws.session.get("session_id")
        if not sid:
            raise AgentError("no session to resume; call start() first")
        return self._turn(self.command(sid, resume=True), message, "resume")

    def _turn(self, argv, prompt, kind) -> tuple[Action, Usage]:
        t0 = time.time()
        proc = self._run(argv, prompt)
        result = _json_object(proc.stdout)
        if result is None:
            raise AgentError(f"claude exit {proc.returncode} without a JSON result after {time.time() - t0:.0f} s\n"
                             f"stderr: {_tail(proc.stderr)}\nstdout: {_tail(proc.stdout, 5)}")
        n = self.ws.session.get("turns", 0) + 1
        self.ws.session["turns"] = n
        self.ws.save_session()
        turns = self.ws.path / "turns"
        turns.mkdir(exist_ok=True)
        (turns / f"turn_{n:03d}.json").write_text(json.dumps(
            {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "argv": argv, "prompt": prompt,
             "exit_code": proc.returncode, "stderr": proc.stderr[-4000:], "result": result}, indent=1))
        usage = Usage.from_result(result, model=self.model)
        self.ws.record_usage(usage, turn=n, kind=kind)
        if result.get("is_error"):
            raise AgentError(f"claude reported an error (exit {proc.returncode}): {str(result.get('result'))[:1500]}\n"
                             f"stderr: {_tail(proc.stderr)}")
        try:
            action = schema.parse(result)
        except schema.SchemaError as e:
            raise AgentError(f"invalid action: {e}; result text: {str(result.get('result'))[:500]}") from e
        return action, usage

    def _run(self, argv, prompt) -> subprocess.CompletedProcess:
        env = {**os.environ, **AGENT_ENV}
        if self.fake:
            cmd, cwd = argv, self.ws.path
        else:
            cmd, cwd = ["ssh", "-o", "BatchMode=yes", self.ws.host, self.shell_command(argv)], None
        try:
            return subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_s, cwd=cwd, env=env)
        except subprocess.TimeoutExpired as e:
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            raise AgentError(f"claude timed out after {self.timeout_s} s; stderr: {_tail(err)}") from e
        except OSError as e:
            raise AgentError(f"cannot run {cmd[0]}: {e}") from e


OUTCOMES = {0: "success (the program claims success; the orchestrator verifies it)", 2: "fail (claimed fail or a failed check)",
            130: "STOPPED BY THE OPERATOR (Ctrl+C)", 3: "exception (traceback in the log)"}
IMAGES = (".jpg", ".jpeg", ".png")


def digest(attempt_dir, exit_code=None, outcome=None, summary_lines: int = 60, log_lines: int = 40) -> str:
    """Message that resumes the agent after an attempt: exit code, outcome, summary.json, log tail, images."""
    a = Path(attempt_dir)
    meta = {}
    if (a / "attempt.json").exists():
        try:
            meta = json.loads((a / "attempt.json").read_text())
        except ValueError:
            meta = {}
    if exit_code is None:
        exit_code = meta.get("exit_code")
        if exit_code is None and (a / "exit_code").exists():
            exit_code = (a / "exit_code").read_text().strip()
    try:
        exit_code = int(exit_code)
    except (TypeError, ValueError):
        exit_code = None
    if outcome is None:
        outcome = meta.get("outcome")
    if isinstance(outcome, dict):
        outcome = ", ".join(f"{k}={v}" for k, v in outcome.items() if v not in (None, ""))
    if not outcome:
        outcome = OUTCOMES.get(exit_code, "unknown (no exit code recorded)" if exit_code is None else f"exit code {exit_code}")
    out = [f"# {a.name}: exit code {exit_code if exit_code is not None else '?'}; outcome: {outcome}"]
    head = [f"{k}: {meta[k]}" for k in ("command", "what", "commit", "duration_s", "moves_robot") if k in meta]
    if head:
        out.append("; ".join(head))
    if exit_code == 130:
        out.append("The operator stopped this attempt: it FAILED, whatever the program claims. Find what the robot was doing "
                   "at that moment (last phase, segment, decisions in the log) before changing anything.")
    if (a / "summary.json").exists():
        lines = (a / "summary.json").read_text(errors="replace").strip().splitlines()
        cut = lines[:summary_lines] + ([f"... ({len(lines) - summary_lines} more lines)"] if len(lines) > summary_lines else [])
        out.append("summary.json:\n" + "\n".join(line[:300] for line in cut))
    else:
        out.append("summary.json: MISSING (the program must write it, also on failure)")
    if (a / "log.txt").exists():
        log = (a / "log.txt").read_text(errors="replace").splitlines()
        out.append(f"log.txt: {len(log)} lines; last {min(log_lines, len(log))}:\n" + "\n".join(line[:300] for line in log[-log_lines:]))
    else:
        out.append("log.txt: MISSING")
    files = sorted(p for p in a.rglob("*") if p.is_file())
    images = [p for p in files if p.suffix.lower() in IMAGES]
    if images:
        out.append("images (Read one only if needed, <= 640 px):\n" + "\n".join(
            f"  {p.relative_to(a)}  {max(1, p.stat().st_size // 1024)} KB" for p in images))
    others = [p for p in files if p not in images and p.name not in ("summary.json", "log.txt", "attempt.json", "exit_code")]
    if others:
        out.append("other files: " + ", ".join(f"{p.relative_to(a)} ({max(1, p.stat().st_size // 1024)} KB)" for p in others[:20])
                   + (f", +{len(others) - 20} more" if len(others) > 20 else ""))
    out.append("An attempt the operator stopped (exit code 130) counts as a failure, whatever the program claims. "
               "Decide the next action from this digest; end your turn with the structured action.")
    return "\n\n".join(out)
