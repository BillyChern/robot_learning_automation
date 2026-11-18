"""The phases of one run: develop_program (agents 1/2), collect, learn/evaluate (agent 3), run_all.
Other modules are imported inside the methods so each phase only needs what it uses."""
import importlib
import json
import shlex
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from rlpj import config
from . import attempts
from .state import RunState, now_iso, report_md, update_meta

RESTART_MSG = "The orchestrator restarted. Summarize in one line where your work stands, then continue."
NO_OPERATOR = ("No operator is available: this run is fully autonomous and nobody will act on requests. Do it yourself "
               "with the robot through a program (for example, set up the scene state you need with your own pick-and-place), "
               "work around it, or skip that test if it cannot be done safely. Do not ask for an operator again.")
MAX_ASKS = 3                # autonomous: need_operator requests in a row before the agent counts as gave_up
AUTONOMOUS_RESET_REPAIRS = 2  # autonomous: extra reset-agent repairs instead of a manual reset, then the run stops
REFLECT_RULES = ("This is a reflection turn while the robot runs the other agent's program: you cannot use the robot or "
                 "the cameras now (no run actions, not even dry runs). If this run shows a problem or a clear, safe "
                 "improvement (robustness first, then speed), edit {program}, check it offline on this machine, commit, "
                 "and reply done; reply done at once if nothing should change. Your next run uses what you committed, "
                 "and its result comes back to you the same way.")
REFLECT_BUSY = "Not now: the robot is busy. Edit and test offline only, commit, then reply done."
REFLECT_TURNS = 4           # agent turns per reflection before it is closed
RESTART_EXIT = 75          # exit code of a requested restart (runs/<run>/RESTART); the launcher loop relaunches
PULLED = ".pulled_from"     # marker in a checkpoint dir copied from the agent host for local serving


def _mod(name):
    return importlib.import_module(name)


def _get(o, k, default=None):
    """Field of an Action/Usage object or dict."""
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _ts(iso):
    return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


def _strip_targets(command):
    toks = shlex.split(command)
    if "--targets" in toks:
        i = toks.index("--targets")
        del toks[i:i + 2]
    return shlex.join(toks)


def _images(d):
    """before/after/scene jpgs of a directory, name -> path (what the VLM judge gets)."""
    return {p.stem: str(p) for p in sorted(Path(d).glob("*.jpg")) if p.stem.startswith(("before", "after", "scene"))}


def _fmt_targets(t):
    return ", ".join(f"{k} ({v[0]:.2f}, {v[1]:.2f})" for k, v in (t or {}).items())


def _reason(rep):
    return rep.get("reason", str(rep)) if isinstance(rep, dict) else str(rep)


def _group_cmdlines(pgid):
    """Command lines of the processes in process group pgid (from /proc)."""
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if int((d / "stat").read_text().rsplit(")", 1)[1].split()[2]) == pgid:
                    out.append((d / "cmdline").read_bytes())
            except (OSError, ValueError, IndexError):
                pass
    return out


def _stop_group(pgid, proc=None, wait_s=30):
    """SIGTERM a process group, SIGKILL what is left after wait_s."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc is not None:
            proc.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Pipeline:
    """Runs the phases of one run with a RunState, an Operator and the config; every phase resumes from the state."""

    def __init__(self, st: RunState, op, cfg=None):
        self.st, self.op = st, op
        self.cfg = cfg or config.load(st.task)
        self.task = self.cfg["task"]
        self.rng = np.random.default_rng()
        self._ws, self._runners = {}, {}
        self._last = {}                               # role -> last robot attempt of the current agent session
        self._fails = {"execute": [], "reset": []}    # (attempt_dir, why) since the last success, for repair digests
        self._pending_targets = None                  # targets of the episode being reset (reset repair runs use them)
        self._server = None                           # local policy server process (hosts.policy_host: local)
        self._eval_code = {}                          # agent 3's own server / rollout program for the running evaluation
        self._reflections = {}                        # role -> thread of its running reflection (collection)
        self._reset_runs = []                         # [(attempt_dir, what happened)] of the resets of the current episode
        self._stuck = 0                               # robot programs in a row that ended with the object on the gripper

    # ------------------------------------------------------------------ shared pieces
    def workspace(self, role, **extra):
        """Open runs/<run>/agents/<role>, creating it (with its rendered PROMPT.md) the first time."""
        if role not in self._ws:
            W = _mod("rlpj.agents.workspace").Workspace
            path = self.st.dir / "agents" / role
            if (path / "session.json").exists():
                self._ws[role] = W(path)
            else:
                condition = self.st.condition if role == "execute" else None
                self._ws[role] = W.create(self.st.dir, role, self.task, condition, **extra)
        return self._ws[role]

    def runner(self, role):
        if role not in self._runners:
            a = self.cfg["agents"]
            self._runners[role] = _mod("rlpj.agents.runner").AgentRunner(
                self.workspace(role), model=a["model"], effort=a["effort"], max_budget_usd=a.get("max_budget_usd"))
        return self._runners[role]

    def _session_id(self, role):
        ws = self._ws.get(role)
        return (getattr(ws, "session", None) or {}).get("session_id") if ws else self.st.agent(role).get("session_id")

    def _account(self, role, usage):
        """Add one turn's usage to the agent's record (field names of rlpj.agents.schema.Usage or short ones)."""
        a = self.st.agent(role)
        pairs = (("cost_usd", "cost_usd"), ("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                 ("cache_read", "cache_read_input_tokens"), ("cache_read", "cache_read"),
                 ("cache_creation", "cache_creation_input_tokens"), ("cache_creation", "cache_creation"),
                 ("turns", "turns"), ("duration_s", "duration_s"))
        for mine, theirs in pairs:
            a[mine] = a.get(mine, 0) + (_get(usage, theirs, 0) or 0)
        a["duration_s"] += (_get(usage, "duration_ms", 0) or 0) / 1000
        a["session_id"] = _get(usage, "session_id") or self._session_id(role) or a.get("session_id")
        self.st.save()
        self._archive_transcript(role)
        self.st.ledger("agent", role=role, model=_get(usage, "model"), session_id=a["session_id"],
                       cost_usd=_get(usage, "cost_usd", 0), input_tokens=_get(usage, "input_tokens", 0),
                       output_tokens=_get(usage, "output_tokens", 0), cache_read=_get(usage, "cache_read_input_tokens", 0),
                       cache_creation=_get(usage, "cache_creation_input_tokens", 0), turns=_get(usage, "turns", 0),
                       duration_s=round((_get(usage, "duration_ms", 0) or 0) / 1000, 1))

    def _archive_transcript(self, role):
        """Copy the agent's full Claude Code transcript (every message and tool call) from the agent host into
        runs/<run>/transcripts/<role>/; Claude Code deletes old transcripts there after 30 days."""
        enc = re.sub(r"[^A-Za-z0-9]", "-", str(self.st.dir / "agents" / role))
        dst = self.st.dir / "transcripts" / role
        dst.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["rsync", "-a", f"{self.cfg['hosts']['agent']}:.claude/projects/{enc}/", f"{dst}/"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode:
            self.op.notify(f"transcript archive for {role} failed: {r.stderr.strip()[-200:]}")

    def _safe(self, mod, fn, *args, **kw):
        """Call mod.fn; a missing module or an exception is reported and gives None instead of stopping the loop."""
        try:
            return getattr(_mod(mod), fn)(*args, **kw)
        except Exception as e:
            self.op.notify(f"{mod}.{fn} failed: {e!r}")
            return None

    def _measure(self, save_jpg=None):
        """Blobs on the table from one fresh side-camera measurement; the camera is opened and closed here."""
        cam, tt = _mod("rlpj.robot.cams").SideCam(), _mod("rlpj.perception.tabletop")
        try:
            blobs = tt.detect(tt.cloud(cam))
            if save_jpg is not None:
                color = cam.latest()[0]
                if color is not None:
                    import cv2
                    Path(save_jpg).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_jpg), cv2.resize(color, (640, 360)))
        finally:
            cam.close()
        return blobs

    def _safe_measure(self, save_jpg=None):
        try:
            return self._measure(save_jpg)
        except Exception as e:
            self.op.notify(f"side camera measurement failed: {e!r}")
            return None

    def _summ(self, blobs):
        return None if blobs is None else self._safe("rlpj.perception.tabletop", "summarize", blobs)

    def _attempt(self, cwd, command, attempt_dir, moves_robot, record_dir=None, extra_env=None, timeout_s=None, what="", commit=None):
        """run_attempt with the wrist-stream check before robot motion; what/commit go to attempt.json for the digest."""
        attempt_dir = Path(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "attempt.json").write_text(json.dumps(dict(what=what, commit=commit)))
        if moves_robot:
            attempts.wrist_stream_free(self.cfg["robot"]["ip"])
        # the default time limit is for robot runs only; GPU jobs (convert, train) and offline runs bring their own
        default = self.cfg["collection"].get("attempt_timeout_s") if moves_robot else None
        r = attempts.run_attempt(cwd, command, attempt_dir, moves_robot, record_dir=record_dir, extra_env=extra_env,
                                 timeout_s=timeout_s or default)
        if moves_robot:
            self._check_gripper(attempt_dir, Path(cwd).name if Path(cwd).parent.name == "agents" else None)
        return r

    def _gripper_frame(self, wait_s=6.0):
        """One fresh wrist-camera frame (the stream is opened and closed here), or None."""
        cls = getattr(_mod("rlpj.robot.cams"), "WristCam", None)
        if cls is None:
            return None
        cam, t0 = cls(self.cfg["robot"]["ip"]), time.time()
        try:
            while time.time() - t0 < wait_s:
                img, t = cam.latest()
                if img is not None and t > t0 + 0.3:
                    return img
                time.sleep(0.1)
            return None
        finally:
            cam.close()

    def _check_gripper(self, attempt_dir, role):
        """After a robot program (arm at home): is the object hanging on the fingers (task gripper_check)? Counts the
        programs in a row that ended so and tells the reset agent (and the agent whose program it was) by a note."""
        chk = self.task.get("gripper_check")
        if not chk:
            return None
        try:
            img = self._gripper_frame()
            f = _mod("rlpj.perception.wrist").gripper_load(img, chk)
        except Exception as e:
            self.op.notify(f"gripper check failed: {e!r}")
            return None
        if f is None:
            self.op.notify("gripper check: no wrist frame")
            return None
        import cv2
        cv2.imwrite(str(Path(attempt_dir) / "gripper_check.jpg"), cv2.resize(img, (640, 360)))
        if f <= chk["max_frac"]:
            self._stuck = 0
            return f
        held = self._held_at_end(attempt_dir)
        if held is None:
            self._stuck += 1
        obj = chk.get("object") or self.task["objects"][-1]
        how = f"held in the closed jaws (reading {held:.1f})" if held is not None else "hanging on the gripper"
        self.st.event("gripper_not_clear", attempt=str(attempt_dir), frac=round(f, 3), in_a_row=self._stuck, held=held)
        self.op.notify(f"gripper check: the {obj} is {how} after {Path(attempt_dir).name} "
                       f"(object colour {f:.2f} > {chk['max_frac']} at the bottom of the wrist image)")
        msg = (f"The orchestrator's wrist check after {role or 'a'} run {Path(attempt_dir).name} (arm at home) shows the "
               f"{obj} {how}: object colour {f:.2f} of the bottom {chk.get('bottom', 0.15):.0%} of the wrist "
               f"image (threshold {chk['max_frac']}; image {Path(attempt_dir) / 'gripper_check.jpg'}).")
        for r in {role, "reset"} - {None}:
            fix = ("Your next run must take it off first: put it down on the table, check again, and only then go on."
                   if r == "reset" else "Your program released it but it did not come off. Check the fingers after every "
                   "release before moving away, and shake it off over the target if it is still there.")
            self._add_note(r, f"{msg} {fix}")
        return f

    def _held_at_end(self, attempt_dir):
        """The final gripper reading when the program ended with the jaws closed on something (a firm hold, e.g. after a
        refused release), else None."""
        try:
            g = json.loads((Path(attempt_dir) / "summary.json").read_text()).get("gripper_final") or {}
        except (OSError, ValueError):
            return None
        r = g.get("reading")
        return float(r) if r is not None and (g.get("cmd") or 0) > 0.5 and 5.0 < r < 97.5 else None

    def deliver_notes(self):
        """Agents with a pending operator note get it now in a reflection turn (they may edit and test offline, no robot)."""
        for role in ("reset", "execute"):
            if (self.st.dir / "notes" / f"{role}.md").exists() and self._session_id(role):
                self.op.notify(f"delivering the operator note to the {role} agent before collection continues")
                self._reflect(role, "Before collection continues: no robot run now; update your program for the note above "
                              "and test it offline.", [])

    def _add_note(self, role, text):
        """Append to the operator note for an agent (delivered with its next message by _with_note)."""
        f = self.st.dir / "notes" / f"{role}.md"
        f.parent.mkdir(exist_ok=True)
        with open(f, "a") as fh:
            fh.write(text.strip() + "\n")

    def _stuck_stop(self):
        """True (and the run stops for a person) when the object stayed on the gripper after stop_after programs."""
        n = int((self.task.get("gripper_check") or {}).get("stop_after", 2))
        if self._stuck < n:
            return False
        obj = (self.task.get("gripper_check") or {}).get("object") or self.task["objects"][-1]
        self.op.notify(f"the {obj} has stayed on the gripper after {self._stuck} robot programs in a row: stopping before it "
                       f"falls; take it off by hand and put it on the table, then rerun")
        self.st.event("autonomous_stop", reason=f"{obj} stuck on the gripper", in_a_row=self._stuck)
        return True

    def _sample_targets(self):
        t = _mod("rlpj.perception.layout").sample(self.task, self.rng)
        return {k: [float(v[0]), float(v[1])] for k, v in t.items()}

    def _reset_orientation(self, attempt_dir):
        """(ok, why) of the task's reset_orientation criterion, judged by the VLM on a full-size side image after a reset
        ("" = no criterion). Unsure counts as not confirmed."""
        text = (self.task.get("reset_orientation") or "").strip()
        if not text:
            return True, ""
        img = Path(attempt_dir) / "orientation_side.jpg"
        try:
            cam = _mod("rlpj.robot.cams").SideCam()
            try:                                       # frames for 2.5 s after the first: auto-exposure settles
                color, t_first, t0 = None, None, time.time()
                while time.time() - t0 < 10:
                    c = cam.latest()[0]
                    if c is not None:
                        color, t_first = c, t_first or time.time()
                        if time.time() - t_first > 2.5:
                            break
                    time.sleep(0.1)
            finally:
                cam.close()
            if color is None:
                return False, "orientation: no side image"
            if float(color.mean()) < 40:
                return False, f"orientation: side image too dark (mean {float(color.mean()):.0f})"
            import cv2
            cv2.imwrite(str(img), color)
            res = _mod("rlpj.success.vlm_judge").judge({"after": img}, dict(goal=self.task.get("reset", ""), success=text))
        except Exception as e:
            return False, f"orientation check failed: {e!r}"
        self.st.event("reset_orientation", attempt=str(attempt_dir), success=res.get("success"),
                      confidence=res.get("confidence"), reason=res.get("reason"), cost_usd=res.get("cost_usd"))
        if res.get("success") is True and float(res.get("confidence") or 0) >= 0.6:
            return True, res.get("reason", "")
        return False, f"orientation not confirmed by the vision check: {res.get('reason', '')}"

    def _layout_ok(self, targets, blobs):
        """(ok, report) of rlpj.perception.layout.check on a measurement (None = no measurement)."""
        if blobs is None:
            return False, "no side-camera measurement"
        try:
            ok, rep = _mod("rlpj.perception.layout").check(self.task, targets, blobs)
            return bool(ok), rep
        except Exception as e:
            return False, f"layout check failed: {e!r}"

    def _verdict(self, claim, before, after, images, aborted=False):
        """Program claim + geometric check + VLM judge -> dict(final, reason, perception, vlm) via success.verdict.decide."""
        if aborted:
            return dict(final="fail", reason="stopped by the operator or timed out", perception=None, vlm=None)
        perception = vlm = None
        if before is not None and after is not None:
            perception = self._safe("rlpj.success.checks", "run", self.task, before, after)
        if images:
            vlm = self._safe("rlpj.success.vlm_judge", "judge", images, self.task)
            if vlm:
                self.st.vlm["calls"] += 1
                self.st.vlm["cost_usd"] += float(vlm.get("cost_usd") or 0)
                self.st.ledger("vlm", cost_usd=vlm.get("cost_usd"), duration_s=vlm.get("duration_s"), success=vlm.get("success"),
                               confidence=vlm.get("confidence"), usage=vlm.get("usage"),
                               images=str(Path(next(iter(images.values()))).parent))
        p_ok = perception[0] if isinstance(perception, (tuple, list)) else perception
        v = _mod("rlpj.success.verdict").decide(claim, p_ok, vlm)
        return dict(final=v["final"], reason=v.get("reason", ""), vlm=vlm,
                    perception=None if perception is None else dict(ok=p_ok, report=perception[1] if isinstance(perception, (tuple, list)) else None))

    # ------------------------------------------------------------------ agents 1 and 2
    def develop_program(self, role):
        """Agent 1 (execute) or 2 (reset) writes its program until an accepted 'done'; returns done|gave_up|max_attempts."""
        a = self.st.agent(role)
        self.st.phase = f"develop_{role}"
        a["started"] = a.get("started") or now_iso()
        a["model"], a["status"] = self.cfg["agents"]["model"], "developing"
        self.st.save()
        runner = self.runner(role)
        self._ensure_remote(self.workspace(role))
        restart = RESTART_MSG + (" " + NO_OPERATOR if self._autonomous() else "")
        action, usage = runner.resume(restart) if self._session_id(role) else runner.start()
        status = self._agent_loop(role, action, usage)
        a["ended"] = now_iso()
        self.st.event("agent_status", role=role, status=status)
        self.st.save()
        self.op.notify(f"{role} agent: {status} after {a['attempts']} runs ({a['robot_attempts']} on the robot), ${a['cost_usd']:.2f}")
        return status

    def _agent_loop(self, role, action, usage):
        """Drive one agent session until an accepted done, give_up or agents.max_attempts runs; returns the status."""
        a, runner = self.st.agent(role), self.runner(role)
        n0, limit, asks = a["attempts"], self.cfg["agents"]["max_attempts"], 0
        while True:
            self._account(role, usage)
            kind = _get(action, "action")
            self.st.event("agent", role=role, action=kind, what=_get(action, "what"), report=_get(action, "report"))
            asks = asks + 1 if kind == "need_operator" else 0
            if kind == "run":
                if a["attempts"] - n0 >= limit:
                    a["status"] = "max_attempts"
                    return "max_attempts"
                msg = self._agent_run(role, action)
            elif kind == "need_operator":
                if self._autonomous() and asks > MAX_ASKS:
                    a["status"], a["last_report"] = "gave_up", f"asked for an operator {asks} times in autonomous mode"
                    self.op.notify(f"{role} agent keeps asking for an operator; counted as gave_up")
                    return "gave_up"
                msg = self._operator_reply(role, _get(action, "report"))
            elif kind == "done":
                a["last_report"] = _get(action, "report")
                ok, why = self._accept(role)
                if ok:
                    a["status"] = "done"
                    self.op.notify(f"{role} agent done: {a['last_report']}")
                    return "done"
                self.op.notify(f"{role} done not accepted: {why}")
                msg = f"not accepted because {why}. Fix it and run again, or give_up."
            else:
                a["status"], a["last_report"] = "gave_up", _get(action, "report")
                self.op.notify(f"{role} agent gave up: {a['last_report']}")
                return "gave_up"
            self.restart_point()
            action, usage = runner.resume(self._with_note(role, msg))

    def _with_note(self, role, msg):
        """msg with the operator's note for this agent first (runs/<run>/notes/<role>.md, delivered once, then renamed
        to <role>.sent_<time>.md)."""
        f = self.st.dir / "notes" / f"{role}.md"
        try:
            note = f.read_text().strip()
            f.rename(f.with_name(f"{role}.sent_{time.strftime('%Y%m%d_%H%M%S')}.md"))
        except OSError:
            return msg
        if not note:
            return msg
        self.st.event("operator_note", role=role, note=note)
        self.op.notify(f"operator note delivered to the {role} agent")
        return f"Note from the operator (overrides your earlier assumptions where they disagree):\n{note}\n\n{msg}"

    def _agent_run(self, role, action):
        """Execute one 'run' action (gate, pull the workspace, attempt, measurements, push); returns the resume message."""
        a, ws = self.st.agent(role), self.workspace(role)
        command, moves, what = _get(action, "command") or "", bool(_get(action, "moves_robot")), _get(action, "what") or ""
        if not self.op.gate(what, moves, command):
            return (f"the operator {self.op.last} this run" +
                    ("; explain the plan or ask what to change (need_operator)." if self.op.last == "refused" else "; continue without it."))
        ws.pull()
        attempt_dir, targets, extra = Path(ws.next_attempt_dir()), None, {}
        if role == "reset" and moves:
            command, targets, tpath = self._reset_command(command, attempt_dir)
            extra["RLPJ_RESET_TARGETS"] = str(tpath)
        if role == "execute" and self._rl():
            extra.update(self._rl_env("train", -1))
            # every real test starts from a fresh reset; "repairing"/"reviewing" here means a repair cut short by a restart
            # (one agent runs at a time), so the last accepted reset program is used and a new failure starts a new repair
            if moves and self.st.agent("reset").get("status") in ("done", "repairing", "reviewing"):
                if self._reset_to(self._sample_targets()) is None:
                    return "the reset before your run failed and the run was not started; the orchestrator is on it, run again."
        before = self._safe_measure(attempt_dir / "scene_before.jpg") if moves else None
        r = self._attempt(ws.path, command, attempt_dir, moves, extra_env=extra, what=what, commit=self._safe_head(ws))
        after = self._safe_measure(attempt_dir / "scene_after.jpg") if moves else None
        a["attempts"] += 1
        note = ""
        if moves:
            a["robot_attempts"] += 1
            ok, note = self._independent(role, r, before, after, targets)
            if ok:
                a["program_command"] = _strip_targets(command) if role == "reset" else command
                if a["first_success_attempt"] is None:
                    a.update(first_success_attempt=a["robot_attempts"], first_success_after_attempts=a["attempts"],
                             time_to_first_success_s=round(time.time() - _ts(a["started"]), 1))
            self._last[role] = dict(result=r, before=before, after=after, targets=targets, ok=ok, note=note, dir=attempt_dir)
        self.st.event("attempt", role=role, attempt=attempt_dir.name, moves_robot=moves, exit_code=r.exit_code,
                      outcome=r.outcome, duration=r.duration, what=what)
        self.st.save()
        self._push(ws)
        return _mod("rlpj.agents.runner").digest(attempt_dir) + (f"\n\nthe orchestrator: {note}" if note else "")

    def _safe_head(self, ws):
        try:
            return ws.head()
        except Exception:
            return None

    def restart_point(self):
        """Exit with RESTART_EXIT at a safe point (no robot program, no agent turn) when runs/<run>/RESTART exists, so
        the launcher loop reloads the code; the state is saved and every phase resumes from it."""
        f = self.st.dir / "RESTART"
        if f.exists():
            for role in list(self._reflections):
                self._join(role)
            why = f.read_text().strip() or "code update"
            f.unlink()
            self.st.event("restart", reason=why)
            self.st.save()
            self.op.notify(f"restart requested ({why}): exiting at a safe point")
            raise SystemExit(RESTART_EXIT)

    def _autonomous(self):
        return bool(getattr(self.op, "autonomous", False))

    def _operator_reply(self, role, report):
        """Answer a need_operator action: the operator's reply, or in autonomous mode the standing NO_OPERATOR reply."""
        if self._autonomous():
            self.op.notify(f"{role} agent asked for an operator ({report}); autonomous: told to do it itself")
            self.st.event("operator_auto_reply", role=role, request=report)
            return NO_OPERATOR
        self.op.notify(f"{role} agent needs you: {report}")
        return "Operator: " + (self.op.ask("reply to the agent (Enter = done, continue)") or "done, continue")

    def _ensure_remote(self, ws):
        """The agent runs in its workspace on the agent host: copy it there (full push) when it does not exist yet and
        mark it trusted, so Claude Code applies the workspace's permission rules."""
        try:
            if not ws.remote_exists():
                ws.push(full=True)
                self.op.notify(f"workspace {ws.path.name} copied to the agent host")
            ws.trust()
        except Exception as e:
            self.op.notify(f"workspace setup on the agent host failed: {e!r}")

    def _push(self, ws):
        """Sync attempt results to the agent machine; a failure is reported, the digest text still reaches the agent."""
        try:
            ws.push()
        except Exception as e:
            self.op.notify(f"workspace push failed: {e!r}")

    def _reset_command(self, command, attempt_dir):
        """Targets for a reset run: the agent's own --targets (file or inline JSON) or sampled ones; returns (command, targets, path)."""
        toks, tpath = shlex.split(command), attempt_dir / "targets.json"
        targets = None
        if "--targets" in toks and toks.index("--targets") + 1 < len(toks):
            arg = toks[toks.index("--targets") + 1]
            try:
                p = Path(arg) if Path(arg).is_absolute() else Path(self.workspace("reset").path) / arg
                targets = json.loads(p.read_text()) if p.exists() else json.loads(arg)
            except (OSError, ValueError):
                targets = None
        if not isinstance(targets, dict):
            targets = self._pending_targets or self._sample_targets()
            command = f"{_strip_targets(command)} --targets {shlex.quote(str(tpath))}"
        tpath.write_text(json.dumps(targets))
        return command, targets, tpath

    def _independent(self, role, r, before, after, targets):
        """The orchestrator's own view of a robot attempt: (ok, one-line note for the agent)."""
        if role == "reset":
            ok, rep = self._layout_ok(targets, after)
            return ok, f"layout check {'ok' if ok else 'FAILED'} ({_reason(rep)}); targets were {json.dumps(targets)}"
        ok = r.outcome == "success"
        chk = self._safe("rlpj.success.checks", "run", self.task, before, after) if before is not None and after is not None else None
        if isinstance(chk, (tuple, list)) and chk[0] is False:
            ok = False
        note = f"outcome {r.outcome}; independent geometric check: " + ("not available" if chk is None else f"{chk[0]} ({_reason(chk[1])})")
        return ok, note

    def _accept(self, role):
        """Success criterion behind a 'done': (accepted, why not)."""
        if role == "execute" and self._rl():
            return self._accept_rl()
        last = self._last.get(role)
        if not last:
            return False, "there was no robot run in this session yet (run the program with moves_robot true first)"
        if role == "reset":
            return (True, "") if last["ok"] else (False, f"the last reset run did not place the objects within tolerance: {last['note']}")
        if last["result"].outcome != "success":
            return False, f"the last robot run ended with outcome {last['result'].outcome}"
        v = self._verdict("success", last["before"], last["after"], _images(last["dir"]))
        last["verdict"] = v
        if v["final"] == "fail":
            return False, f"the independent verdict says fail: {v['reason']}"
        return True, ""

    def _repair(self, role):
        """Resume the agent with the digests of the recent failures and let it fix its program; returns the status."""
        fails = self._fails[role]
        self._fails[role] = []
        self.op.notify(f"{role} program failed {len(fails)} times in a row; asking the agent to repair it")
        head = (f"repair your program; it failed {len(fails)} times in a row during "
                f"{'collection' if role == 'execute' else 'the resets between episodes'}:")
        return self._improve(role, head, fails, "repair")

    def _improve(self, role, head, items, kind):
        """Resume the agent with a header and the digests of failed attempts [(attempt_dir, why)]; it fixes its program,
        tests it with a robot run and replies done. kind: repair (failures in a row) | review (periodic); returns the status."""
        self._join(role)
        a, ws = self.st.agent(role), self.workspace(role)
        a["repairs" if kind == "repair" else "reviews"] = a.get("repairs" if kind == "repair" else "reviews", 0) + 1
        a["status"] = "repairing" if kind == "repair" else "reviewing"
        self._last[role] = None
        self._push(ws)
        dg = _mod("rlpj.agents.runner").digest
        lines = [head]
        for d, why in items:
            lines += [f"--- {Path(d).name}: {why}", str(dg(d))]
        lines.append(f"Find the cause, fix programs/{role}.py, test it with a run (moves_robot true) and reply done when it works.")
        action, usage = self.runner(role).resume(self._with_note(role, "\n".join(lines)))
        status = self._agent_loop(role, action, usage)
        self.st.event(kind, role=role, status=status)
        self.st.save()
        return status

    def _reflect_async(self, role, head, items):
        """Start the agent's reflection on its last run in a thread (it overlaps with the other agent's robot run)."""
        if not self.cfg["collection"].get("reflect_after_every_run"):
            return
        self._join(role)
        t = threading.Thread(target=self._reflect, args=(role, head, list(items)), daemon=True, name=f"reflect-{role}")
        self._reflections[role] = t
        t.start()

    def _reflect(self, role, head, items):
        """One reflection: digest of the run, the agent may edit and test offline (no robot), ends with done."""
        try:
            ws, runner, a = self.workspace(role), self.runner(role), self.st.agent(role)
            self._push(ws)
            dg = _mod("rlpj.agents.runner").digest
            lines = [head] + [f"--- {Path(d).name}: {why}\n{dg(d)}" for d, why in items]
            msg = self._with_note(role, "\n".join(lines + [REFLECT_RULES.format(program=f"programs/{role}.py")]))
            for _ in range(REFLECT_TURNS):
                action, usage = runner.resume(msg)
                self._account(role, usage)
                kind = _get(action, "action")
                self.st.event("reflect", role=role, action=kind, what=_get(action, "what"), report=_get(action, "report"))
                if kind in ("done", "give_up"):
                    break
                msg = NO_OPERATOR if kind == "need_operator" else REFLECT_BUSY
            with self.st.lock:
                a["reflections"] = a.get("reflections", 0) + 1
            self.st.save()
        except Exception as e:
            self.op.notify(f"{role} reflection failed: {e!r}")

    def _join(self, role):
        t = self._reflections.pop(role, None)
        if t is not None and t.is_alive():
            self.op.notify(f"waiting for the {role} agent's reflection")
            t.join()

    def _ready(self, role):
        """Before the agent's program runs: its reflection is over and its committed files are here."""
        self._join(role)
        if self.cfg["collection"].get("reflect_after_every_run"):
            try:
                self.workspace(role).pull()
            except Exception as e:
                self.op.notify(f"{role} workspace pull failed: {e!r}")

    def _window(self):
        """Collection episodes since the last periodic review (kept in the state)."""
        c = self.st.collection
        if not c.get("window"):
            c["window"] = dict(n=0, success=0, fail=0, unsure=0, failed=[], reset_failed=[])
        return c["window"]

    def _review_due(self):
        every = int(self.cfg["collection"].get("review_every") or 0)
        return every > 0 and self._window()["n"] >= every

    def _review(self):
        """Periodic review: the execute agent gets this window's failed or unsure episodes, the reset agent its failed
        resets, each to make its program more robust; nothing is asked when everything succeeded."""
        w = self._window()
        head = (f"periodic review after {w['n']} collection episodes: {w['success']} success, {w['fail']} fail, "
                f"{w['unsure']} unsure; {len(w['reset_failed'])} failed resets.")
        self.op.notify(head)
        if w["fail"] or w["unsure"]:
            self._improve("execute", head + " Make programs/execute.py succeed on these cases too (more robust; faster only "
                          "where it stays safe). The failed or unsure episodes:", w["failed"][-4:], "review")
        if w["reset_failed"]:
            self._improve("reset", head + " Make programs/reset.py place both objects within tolerance on these cases too. "
                          "The failed resets:", w["reset_failed"][-4:], "review")
        self.st.collection["window"] = None
        self.st.save()

    # ------------------------------------------------------------------ collection
    def collect(self, n_target=None):
        """Episodes until this round has n_target successes or collection.max_episodes; returns the round's successes."""
        col, c = self.cfg["collection"], self.st.collection
        if not c.get("round_target"):
            c.update(round=c.get("round", 0) + 1, round_successes=0, round_target=int(n_target or col["round_size"]))
        elif int(n_target or col["round_size"]) > c["round_target"]:          # round size raised while the round is open
            c["round_target"] = int(n_target or col["round_size"])
        self.st.phase = "collect"
        self.st.save()
        self.deliver_notes()
        cf, ce = self.st.counters["consecutive_failures"], self.st.counters["episodes"]
        self.op.notify(f"collection round {c['round']}: {c['round_successes']}/{c['round_target']} successes, {ce['total']} episodes so far")
        while c["round_successes"] < c["round_target"] and ce["total"] + c.get("discarded", 0) < col["max_episodes"]:
            self.restart_point()
            idx, ep_dir = self._new_episode_dir()
            targets = self._sample_targets()
            self._reset_runs = []
            before = self._reset_to(targets, save=ep_dir / "scene_before.jpg")
            if self._reset_runs:
                self._reflect_async("reset", f"Reflection after the reset for collection episode {idx} "
                                    f"({len(self._reset_runs)} run(s), {'ok' if before is not None else 'not ok'}):", self._reset_runs)
            if before is not None and self._stuck:              # the reset left the object on the gripper: reset again
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue
            if before is not None and self._review_due():       # the agents test their fixes on this reset scene
                shutil.rmtree(ep_dir, ignore_errors=True)
                self._review()
                continue
            ep = self._episode(ep_dir, idx, targets, before, "collect") if before is not None else None
            if ep is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                self.op.notify("collection stopped by the operator")
                self.st.save()
                return c["round_successes"]
            e = self.st.counters["episodes"]
            done = f"{ep['final']} (deleted; the episode is redone)" if ep.get("discarded") else ep["final"]
            self._reflect_async("execute", f"Reflection after collection episode {idx}: {done} ({ep['reason']}), "
                                f"{ep['result'].duration:.0f} s; collection so far {e['success']}/{e['total']} success.",
                                [(ep["attempt_dir"], f"episode {idx}: {ep['final']}")])
            w = self._window()
            w["n"] += 1
            w[ep["final"] if ep["final"] in ("success", "unsure") else "fail"] += 1
            if ep["final"] != "success":
                w["failed"].append([str(ep["attempt_dir"]), f"episode {idx}: {ep['final']} - {ep['reason']}"])
            if ep["final"] == "success":
                c["round_successes"] += 1
                cf["execute"] = 0
                self._fails["execute"] = []
            elif ep["final"] == "fail":
                cf["execute"] += 1
                self._fails["execute"].append((ep["attempt_dir"], f"episode {idx}: {ep['reason']}"))
                if cf["execute"] >= col["refine_after_failures"]:
                    status = self._repair("execute")
                    cf["execute"] = 0
                    if status != "done" and self.op.ask("the execute agent could not repair its program", ["stop", "continue"]) == "stop":
                        self.st.save()
                        return c["round_successes"]
            self.st.save()
        for role in list(self._reflections):
            self._join(role)
        c["round_target"] = None
        self.st.save()
        self.op.notify(f"round {c['round']} done: {c['round_successes']} successes, {ce['total']} episodes in total")
        return c["round_successes"]

    def _new_episode_dir(self):
        idx = self.st.next_episode
        d = self.st.dir / "episodes" / f"ep_{idx:06d}"
        d.mkdir(parents=True, exist_ok=True)
        return idx, d

    def _reset_to(self, targets, save=None):
        """Reset the scene to targets (reset program, agent repair, then the operator); the blobs seen, or None to stop."""
        col, cf = self.cfg["collection"], self.st.counters["consecutive_failures"]
        self._pending_targets, fails, extra = targets, 0, 0
        try:
            while True:
                if self._stuck_stop():
                    return None
                if fails >= col["manual_reset_after_failures"] and self._autonomous():
                    if extra >= (self.cfg["task"].get("rl") or {}).get("reset_repairs", AUTONOMOUS_RESET_REPAIRS):
                        self.op.notify("resets keep failing after the reset agent's repairs; stopping (autonomous: no manual reset)")
                        self.st.event("autonomous_stop", reason="reset failures", targets=targets)
                        return None
                    extra += 1
                    self._repair("reset")
                    cf["reset"], fails = 0, 0
                    continue
                if fails >= col["manual_reset_after_failures"]:
                    if not self.op.wait_manual_reset(targets):
                        return None
                    self.st.counters["resets"]["manual"] += 1
                    self.st.event("manual_reset", targets=targets)
                    blobs = self._safe_measure(save)
                    ok, rep = self._layout_ok(targets, blobs)
                    if not ok:
                        a = self.op.ask(f"layout still off after the manual reset: {_reason(rep)}", ["retry", "accept", "stop"])
                        if a == "stop":
                            return None
                        if a == "retry":
                            continue
                    cf["reset"] = 0
                    return blobs
                r = self._run_reset(targets)
                if r is None:
                    return None
                blobs = self._safe_measure(save)
                ok, rep = self._layout_ok(targets, blobs)
                if ok:
                    ok, why = self._reset_orientation(r.attempt_dir)
                    rep = rep if ok else why
                self._record_reset(targets, r, blobs, ok, rep)
                if ok:
                    cf["reset"] = 0
                    self._fails["reset"] = []
                    return blobs
                fails += 1
                cf["reset"] += 1
                self._fails["reset"].append((r.attempt_dir, f"layout check failed: {_reason(rep)}"))
                self._window()["reset_failed"].append([str(r.attempt_dir), f"layout check failed: {_reason(rep)}"])
                self.op.notify(f"reset failed ({fails} in a row): {_reason(rep)}")
                if cf["reset"] >= col["refine_after_failures"]:
                    status = self._repair("reset")
                    cf["reset"] = 0
                    if status == "done":
                        fails = 0
                        blobs = self._safe_measure(save)
                        ok, rep = self._layout_ok(targets, blobs)
                        if ok:
                            return blobs
        finally:
            self._pending_targets = None

    def _run_reset(self, targets):
        """One gated run of the reset program with these targets; None if the operator refused."""
        ws = self.workspace("reset")
        self._ready("reset")
        attempt_dir = Path(ws.next_attempt_dir())
        attempt_dir.mkdir(parents=True, exist_ok=True)
        tpath = attempt_dir / "targets.json"
        tpath.write_text(json.dumps(targets))
        command = f"{self.st.agent('reset').get('program_command') or 'python programs/reset.py'} --targets {shlex.quote(str(tpath))}"
        if not self.op.gate(f"reset to {_fmt_targets(targets)}", True, command):
            return None
        return self._attempt(ws.path, command, attempt_dir, True, extra_env={"RLPJ_RESET_TARGETS": str(tpath)},
                             what="reset for the next episode", commit=self._safe_head(ws))

    def _record_reset(self, targets, r, blobs, ok, rep):
        n = self.st.next_reset
        self.st.next_reset = n + 1
        d = self.st.dir / "resets"
        d.mkdir(exist_ok=True)
        rec = dict(reset=n, t=now_iso(), targets=targets, ok=ok, report=rep, measured=self._summ(blobs),
                   claimed=r.summary.get("measured"), claimed_outcome=r.summary.get("claimed_outcome"),
                   attempt=str(r.attempt_dir), exit_code=r.exit_code, outcome=r.outcome, duration=r.duration)
        (d / f"reset_{n:04d}.json").write_text(json.dumps(rec, indent=1, default=str))
        c = self.st.counters["resets"]
        c["total"] += 1
        c["ok" if ok else "fail"] += 1
        self.st.event("reset", reset=n, ok=ok, outcome=r.outcome, duration=r.duration, attempt=str(r.attempt_dir))
        self._reset_runs.append((r.attempt_dir, f"reset {n}: layout {'ok' if ok else 'FAILED'} ({_reason(rep)}), {r.duration:.0f} s"))
        self.st.save()

    def _episode(self, ep_dir, idx, targets, before, mode, extra_env=None, meta_extra=None):
        """One recorded robot execution (collect/dagger: the execute program, eval: the policy rollout) with its verdict."""
        if mode == "eval":
            cwd = Path(self.workspace("learn").path) if self._eval_code.get("rollout") else self.st.dir
            attempt_dir, command = ep_dir / "attempt", self._rollout_command()
            timeout, commit = self.cfg["evaluation"]["max_policy_seconds"] + 120, None
        else:
            ws = self.workspace("execute")
            self._ready("execute")
            cwd, attempt_dir, timeout, commit = Path(ws.path), Path(ws.next_attempt_dir()), None, self._safe_head(ws)
            command = self.st.agent("execute").get("program_command") or "python programs/execute.py"
        if not self.op.gate(f"{mode} episode {idx}", True, command):
            return None
        r = self._attempt(cwd, command, attempt_dir, True, record_dir=ep_dir, timeout_s=timeout, what=f"{mode} episode {idx}",
                          commit=commit, extra_env=extra_env)
        after = self._safe_measure(ep_dir / "scene_after.jpg")
        aborted = r.outcome == "aborted"
        claim = "aborted" if aborted else (r.summary.get("claimed_outcome") or r.outcome)
        if mode in ("eval", "rl", "rl_eval") and not aborted:
            claim = None                                # a policy rollout never verifies itself: the independent checks decide
        v = self._verdict(claim, before, after, _images(ep_dir) or _images(attempt_dir), aborted=aborted)
        if mode == "collect" and v["final"] == "unsure" and self.cfg["collection"].get("discard_unsure"):
            return self._discard(ep_dir, idx, v, claim, r, attempt_dir)
        outcome = dict(claim=claim, perception=v["perception"], vlm=v["vlm"], operator=None, final=v["final"], reason=v["reason"])
        update_meta(ep_dir, episode=idx, run_id=self.st.run_id, task=self.task["name"], instruction=self.task["instruction"],
                    mode=mode, attempt=str(attempt_dir), exit_code=r.exit_code, duration=r.duration, layout_target=targets,
                    layout_before=self._summ(before), layout_after=self._summ(after), outcome=outcome, **(meta_extra or {}))
        key = {"collect": "episodes", "eval": "evals"}.get(mode, mode)
        c = self.st.counters.setdefault(key, dict(total=0, success=0, fail=0, unsure=0))
        c["total"] += 1
        c[v["final"] if v["final"] in ("success", "unsure") else "fail"] += 1
        self.st.next_episode = idx + 1
        if mode not in ("dagger", "rl", "rl_eval") and (v["final"] == "unsure" or self._safe("rlpj.success.verdict", "needs_operator", v)):
            self.op.queue_review(ep_dir, v)
        self.st.event("episode", episode=idx, mode=mode, final=v["final"], reason=v["reason"], claim=claim,
                      exit_code=r.exit_code, duration=r.duration, attempt=str(attempt_dir))
        self.st.save()
        self.op.notify(f"episode {idx} ({mode}): {v['final']} - {v['reason']}")
        return dict(index=idx, dir=ep_dir, attempt_dir=attempt_dir, final=v["final"], reason=v["reason"], result=r)

    def _discard(self, ep_dir, idx, v, claim, r, attempt_dir):
        """Unsure collection episode: delete its recording and keep its index for the redo; not counted, not queued."""
        shutil.rmtree(ep_dir, ignore_errors=True)
        c = self.st.collection
        c["discarded"] = c.get("discarded", 0) + 1
        self.st.event("episode_discarded", episode=idx, final=v["final"], reason=v["reason"], claim=claim,
                      exit_code=r.exit_code, duration=r.duration, attempt=str(attempt_dir))
        self.st.save()
        self.op.notify(f"episode {idx} (collect): unsure - {v['reason']}; deleted, redoing it")
        return dict(index=idx, dir=ep_dir, attempt_dir=attempt_dir, final=v["final"], reason=v["reason"], result=r, discarded=True)

    def _policy_end(self, ep_dir):
        """(joints deg, gripper reading) at the last tick of the policy phase of a recorded rollout."""
        d = np.load(Path(ep_dir) / "steps.npz", allow_pickle=True)
        ph = json.loads((Path(ep_dir) / "phases.json").read_text())
        i = next(x["i1"] - 1 for x in ph if x["name"] == "policy")
        return [round(float(v), 3) for v in d["q"][i]], float(d["gripper"][i])

    def _correction(self, ep):
        """DAgger (evaluation.dagger): after a failed rollout the expert, the execute program, takes over from the
        policy's last pose (RLPJ_START_JOINTS; the session goes back there before recording) and finishes the task.
        Recorded as a 'dagger' episode; `rlpj vla convert --modes collect,dagger` adds the successful ones."""
        try:
            q, g = self._policy_end(ep["dir"])
        except Exception as e:
            self.op.notify(f"dagger: no policy end pose in episode {ep['index']} ({e!r}); no correction")
            return None
        if 5.0 < g < 97.5:                             # the jaws ended on something: that state cannot be retraced
            self.op.notify(f"dagger: episode {ep['index']} ended holding something (gripper {g:.1f}); no correction")
            return None
        idx, ep_dir = self._new_episode_dir()
        before = self._safe_measure(ep_dir / "scene_before.jpg")
        fix = None
        if before is not None:
            self.op.notify(f"dagger: the expert takes over from the last pose of episode {ep['index']} (gripper {g:.1f})")
            fix = self._episode(ep_dir, idx, None, before, "dagger", extra_env={"RLPJ_START_JOINTS": json.dumps(q), "RLPJ_DAGGER": "1"},
                                meta_extra=dict(dagger_of=ep["index"], dagger_start=dict(q=q, gripper=round(g, 1))))
        if fix is None:
            shutil.rmtree(ep_dir, ignore_errors=True)
        return fix

    # ------------------------------------------------------------------ agent 3 and evaluation
    def dataset_stats(self):
        c = self.st.counters["episodes"]
        return dict(run_dir=str(self.st.dir), episodes_dir=str(self.st.dir / "episodes"), episodes=dict(c),
                    dagger_episodes=dict(self.st.counters.get("dagger", {})),
                    fps=self.cfg["recording"]["fps"], datasets=list(self.st.datasets), training_runs=list(self.st.training_runs),
                    evals={k: dict(checkpoint=v.get("checkpoint"), success_rate=v.get("success_rate"), n=v.get("n")) for k, v in self.st.evals.items()})

    def _learn_extras(self):
        """Placeholders of prompts/learn.md from the state."""
        s = self.dataset_stats()
        hist = [f"  training run {k}: dataset {v.get('dataset')}, exit {v.get('exit_code')}, `{v.get('command')}`" for k, v in self.st.training_runs.items()]
        hist += [f"  eval {k}: {v.get('success', 0)}/{v.get('n', 0)} success ({v.get('success_rate', 0):.0%}) for {v.get('checkpoint')}" for k, v in self.st.evals.items()]
        e = s["episodes"]
        return dict(N_EPISODES=str(e["total"]), EPISODES_NOTE=f"{e['success']} success, {e['fail']} fail, {e['unsure']} unsure; fps {s['fps']}",
                    EPISODES_DIR=s["episodes_dir"], DATASET=", ".join(s["datasets"]) or "none yet",
                    HISTORY="\n".join(hist) or "  (no training runs yet)")

    def learn(self):
        """Agent 3: convert, train and evaluate until the target success rate or training.max_runs; met|collect_more|gave_up."""
        a, tr, ev = self.st.agent("learn"), self.cfg["training"], self.cfg["evaluation"]
        self.st.phase = "learn"
        a["started"] = a.get("started") or now_iso()
        a["model"], a["status"] = self.cfg["agents"]["model"], "learning"
        pending = a.pop("pending_msg", None)           # the orchestrator restarted mid-round: hand over that run's result
        a["rounds"] = a.get("rounds", 0) + (0 if pending else 1)
        self.st.save()
        self._ensure_remote(self.workspace("learn", **self._learn_extras()))
        runner = self.runner("learn")
        if self._session_id("learn") and pending:
            action, usage = runner.resume(self._with_note("learn", "The orchestrator restarted (code or config update); "
                                                          f"nothing else changed. The result of your last run:\n{pending}"))
        elif self._session_id("learn"):
            action, usage = runner.resume(self._with_note("learn", f"The orchestrator is back with the current data: "
                                          f"{json.dumps(self.dataset_stats())}. Convert the new episodes, train and evaluate "
                                          f"again; training runs this round: {tr['max_runs']}."))
        else:
            action, usage = runner.start()
        n_train, status = 0, None
        while status is None:
            self._account("learn", usage)
            kind = _get(action, "action")
            self.st.event("agent", role="learn", action=kind, what=_get(action, "what"), report=_get(action, "report"))
            if kind == "run":
                command = (_get(action, "command") or "").strip()
                if self._is_train(command) and self._train_refusal(command):
                    msg = self._train_refusal(command)
                    self.st.event("train_refused", command=command, reason=msg)
                    self.restart_point()
                    action, usage = runner.resume(self._with_note("learn", msg))
                    continue
                if self._is_train(command):
                    if n_train >= tr["max_runs"]:
                        status = "collect_more"
                        continue
                    n_train += 1
                msg, met = self._learn_run(action, command)
                if met:
                    status = "met"
            elif kind == "need_operator":
                msg = self._operator_reply("learn", _get(action, "report"))
            elif kind == "done":
                a["last_report"] = _get(action, "report")
                if self.st.deployable_checkpoint:
                    status = "met"
                else:
                    msg = (f"not accepted because no evaluated checkpoint reached {ev['target_success_rate']:.0%}; "
                           "evaluate one with 'rlpj eval --checkpoint <dir>' or train further.")
            else:
                a["last_report"], status = _get(action, "report"), "gave_up"
            if status is None:
                a["pending_msg"] = msg                 # kept in the state if restart_point exits here
                self.restart_point()
                a.pop("pending_msg", None)
                action, usage = runner.resume(self._with_note("learn", msg))
        a["status"] = {"met": "done", "collect_more": "max_runs", "gave_up": "gave_up"}[status]
        a["ended"] = now_iso()
        self.st.event("agent_status", role="learn", status=status)
        self.st.save()
        self.op.notify(f"learn agent: {status} after {n_train} training runs, ${a['cost_usd']:.2f}")
        return status

    @staticmethod
    def _is_train(command):
        """True when one of the command's steps starts a training (not when a step only mentions it, as pgrep/tail do)."""
        for step in re.split(r"&&|\|\||;|\|", command):
            try:
                toks = shlex.split(step)
            except ValueError:
                toks = step.split()
            while toks and ("=" in toks[0] and not toks[0].startswith("-") or toks[0] in ("nohup", "setsid", "uv", "run", "exec", "time")):
                toks = toks[1:]
            if toks[:1] in (["bash"], ["sh"]):
                toks = toks[1:]
            if toks[:3] == ["rlpj", "vla", "train"] or (toks and Path(toks[0]).name == "train.sh") \
                    or ("-m" in toks[:3] and "rlpj.vla.openpi_entry" in toks and "train" in toks):
                return True
        return False

    def _train_refusal(self, command):
        """Why a training command breaks the operator's training policy (training.full_finetune_only, min_train_steps),
        or None. The config comes from RLPJ_CONFIG=... or --config (default pi05_kinova_lora), the steps from
        --num-train-steps (default 5000)."""
        tr = self.cfg["training"]
        m = re.search(r"RLPJ_CONFIG=(\S+)", command) or re.search(r"--config[ =](\S+)", command)
        cfg_name = m.group(1).strip("'\"") if m else "pi05_kinova_full"     # scripts/train.sh default
        m = re.search(r"--num[-_]train[-_]steps[ =](\d+)", command)
        steps = int(m.group(1)) if m else 5000
        why = []
        if tr.get("full_finetune_only") and "lora" in cfg_name.lower():
            why.append(f"config {cfg_name} is LoRA: the operator requires full fine-tuning (a full config, e.g. pi05_kinova_full)")
        if steps < int(tr.get("min_train_steps") or 0):
            why.append(f"{steps} steps: the operator requires at least {tr['min_train_steps']} (--num-train-steps)")
        return ("refused, not run and not counted: " + "; ".join(why)) if why else None

    def _learn_run(self, action, command):
        """One 'run' of agent 3: evaluations here on the robot, everything else over ssh on the agent host."""
        what, ws = _get(action, "what") or command, self.workspace("learn")
        toks = shlex.split(command)
        if toks and (toks[0] in ("evaluate", "eval") or toks[:2] == ["rlpj", "eval"]):
            ckpt = toks[toks.index("--checkpoint") + 1] if "--checkpoint" in toks[:-1] else next((t for t in toks[1:] if "/" in t), None)
            n = toks[toks.index("--n") + 1] if "--n" in toks[:-1] else next((t for t in toks[2:] if t.isdigit()), None)
            if not ckpt:
                return "usage: rlpj eval --checkpoint <checkpoint_dir> [--n N] [--serve <program>] [--rollout <program>]", False
            code = {k: toks[toks.index(f"--{k}") + 1] for k in ("serve", "rollout") if f"--{k}" in toks[:-1]}
            res = self.evaluate(ckpt, int(n) if n else None, **code)
            lines = [json.dumps({k: res.get(k) for k in ("id", "checkpoint", "n", "success", "success_rate", "met", "error")})]
            lines += [f"  episode {e['episode']}: {e['final']} ({e['reason']})" for e in res.get("episodes", [])]
            return "\n".join(lines), bool(res.get("met"))
        if "serve" in toks:
            return "not run: the orchestrator starts and stops the policy server itself during 'rlpj eval --checkpoint <dir>'.", False
        if not self.op.gate(what, False, command):
            return f"the operator {self.op.last} this command.", False
        if toks[:2] == ["rlpj", "vla"] and "--run-dir" not in toks:
            command += f" --run-dir {shlex.quote(str(self.st.dir))}"
        convert = "convert" in toks or "rlpj.vla.convert" in command
        local = self.cfg["hosts"].get("train_host") == "local" and (convert or self._is_train(command))
        if convert and not local:
            self._sync_episodes()
        attempt_dir = Path(ws.next_attempt_dir())
        if local:   # hosts.train_host local: convert and train on this machine (the 5090 kept losing power under load)
            r = self._attempt(config.ROOT, f"PATH={shlex.quote(str(config.ROOT / '.venv/bin'))}:$PATH nice -n 10 {command}",
                              attempt_dir, False, timeout_s=self.cfg["training"].get("timeout_s"), what=what)
        else:
            r = self._ssh(command, attempt_dir, timeout_s=self.cfg["training"].get("timeout_s"), what=what)
        self._note_vla(command, r)
        self._push(ws)
        return f"{_mod('rlpj.agents.runner').digest(attempt_dir)}\n\n(exit code {r.exit_code}, {r.duration:.0f} s)", False

    def _ssh(self, command, attempt_dir, timeout_s=None, what=""):
        """Run a command from the repo root on the agent host (5090) as an attempt logged here ('rlpj ...' via uv run)."""
        host = self.cfg["hosts"]["agent"]
        if command.startswith("rlpj "):
            command = "uv run " + command
        remote = f"cd {shlex.quote(str(config.ROOT))} && {command}"
        return self._attempt(self.st.dir, f"ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "
                             f"{shlex.quote(host)} {shlex.quote(remote)}", attempt_dir, False,
                             timeout_s=timeout_s, what=what)

    def _note_vla(self, command, r):
        """Record convert and train commands in the state (dataset and training run names from the command)."""
        toks = shlex.split(command)

        def opt(name, i=1):
            return toks[toks.index(name) + 1] if name in toks[:-1] else (toks[i] if len(toks) > i else None)

        if self._is_train(command):
            if "train.sh" in command:
                i = next(i for i, t in enumerate(toks) if t.endswith("train.sh"))
                dataset, exp = (toks[i + 1:i + 4] + [None] * 3)[1:3]
            else:
                dataset, exp = opt("--dataset"), opt("--exp-name")
            exp = exp or f"run_{len(self.st.training_runs) + 1}"
            self.st.training_runs[exp] = dict(dataset=dataset, command=command, exit_code=r.exit_code, duration_s=r.duration, t=now_iso(),
                                              checkpoint_dir=str(self.st.dir / "checkpoints" / exp), round=self.st.agent("learn").get("rounds"))
            self.st.ledger("gpu", job="train", name=exp, dataset=dataset, exit_code=r.exit_code, duration_s=r.duration)
        elif "convert" in toks or "rlpj.vla.convert" in command:
            name = opt("--name") or f"dataset_{len(self.st.datasets) + 1}"
            self.st.datasets[name] = dict(command=command, exit_code=r.exit_code, duration_s=r.duration, t=now_iso(),
                                          only_success="--only-success" in toks, episodes=dict(self.st.counters["episodes"]),
                                          path=str(self.st.dir / "datasets" / name))
            self.st.ledger("gpu", job="convert", name=name, exit_code=r.exit_code, duration_s=r.duration)
        self.st.save()

    def _rollout_command(self):
        """rlpj.vla.rollout, or agent 3's rollout program (run from its workspace, same arguments, same RobotSession limits)."""
        h, ev = self.cfg["hosts"], self.cfg["evaluation"]
        prog = self._eval_code.get("rollout")
        head = f"{config.ROOT / '.venv/bin/python'} " + (shlex.quote(prog) if prog else "-m rlpj.vla.rollout")
        return (f"{head} --host {h['policy_server']} --port {h['policy_port']} "
                f"--instruction {shlex.quote(self.task['instruction'])} --max-seconds {ev['max_policy_seconds']}")

    def _policy_host(self):
        """Where the policy server runs: 'local' (this robot machine, the default) or an ssh host."""
        return self.cfg["hosts"].get("policy_host") or "local"

    def _server_start(self, checkpoint_dir, port):
        """bash scripts/serve.sh <ckpt> <port> in the background on the policy host (local: the checkpoint is copied
        from the agent host first, the process group goes to evals/server.pid); log evals/server_<port>.log there.
        False when it cannot start (copy failed, or another server holds the port and would answer instead)."""
        host, ip, log = self._policy_host(), self.cfg["hosts"]["policy_server"], self.st.dir / "evals" / f"server_{port}.log"
        self._server_stop()
        if self._port_busy(ip, port):
            self.op.notify(f"{ip}:{port} is held by another server; stop it first (it would answer for the wrong checkpoint)")
            return False
        if host == "local":
            try:
                ckpt = self._fetch_checkpoint(checkpoint_dir)
            except RuntimeError as e:
                self.op.notify(str(e))
                return False
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, **({"RLPJ_SERVE_PROGRAM": self._serve_program()} if self._eval_code.get("serve") else {})}
            with open(log, "w") as f:
                self._server = subprocess.Popen(["bash", str(config.ROOT / "scripts/serve.sh"), str(ckpt), str(int(port))],
                                                cwd=str(config.ROOT), stdout=f, stderr=subprocess.STDOUT, env=env,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
            (log.parent / "server.pid").write_text(f"{self._server.pid}\n")
        else:
            prog = f"RLPJ_SERVE_PROGRAM={shlex.quote(self._serve_program())} " if self._eval_code.get("serve") else ""
            # only the server goes to the background, inside ( ... ): a backgrounded `a && b && c &` list would keep the ssh
            # channel open for the server's lifetime (2026-11-06: ssh timed out and the orchestrator crashed)
            remote = (f"mkdir -p {shlex.quote(str(log.parent))} && cd {shlex.quote(str(config.ROOT))} && "
                      f"({prog}setsid nohup bash scripts/serve.sh {shlex.quote(str(checkpoint_dir))} {int(port)} "
                      f"> {shlex.quote(str(log))} 2>&1 < /dev/null &)")
            try:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-n", host, remote], timeout=60, check=False,
                                   capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                self.op.notify(f"policy server start on {host} did not return within 60 s")
                return False
            if r.returncode != 0:
                self.op.notify(f"policy server start on {host} failed: {(r.stderr or r.stdout).strip()[-300:]}")
                return False
            self._remote_log = (host, log)
        self.op.notify(f"policy server starting on {host}:{port} for {checkpoint_dir} (log {log} on {host})")
        return True

    def _server_wait(self, ip, port, timeout_s=900):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            p = self._server
            if p is not None and p.poll() is not None:
                self.op.notify(f"policy server exited with code {p.returncode}; see runs/{self.st.run_id}/evals/server_{port}.log")
                return False
            try:
                socket.create_connection((ip, int(port)), timeout=2).close()
                return True
            except OSError:
                time.sleep(5)
            if p is None and self._remote_server_died():
                return False
        return False

    def _remote_server_died(self):
        """True when the remote server's log shows a Python traceback (it died; without this the wait ran 15 min)."""
        host, log = getattr(self, "_remote_log", None) or (None, None)
        if host is None:
            return False
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-n", host, f"grep -q '^Traceback' {shlex.quote(str(log))}"],
                               timeout=20, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False
        if r.returncode == 0:
            self.op.notify(f"policy server on {host} died (Traceback in {log} on {host})")
            return True
        return False

    def _serve_program(self):
        """Absolute path of agent 3's server program (relative paths are in its workspace)."""
        prog = Path(self._eval_code["serve"])
        return str(prog if prog.is_absolute() else Path(self.workspace("learn").path) / prog)

    @staticmethod
    def _port_busy(ip, port, wait_s=15):
        """True when something still accepts connections on ip:port after wait_s (a server shutting down gets time)."""
        t0 = time.time()
        while True:
            try:
                socket.create_connection((ip, int(port)), timeout=2).close()
            except OSError:
                return False
            if time.time() - t0 > wait_s:
                return True
            time.sleep(1)

    def _server_stop(self):
        """Stop the policy server: locally only the process group we started (or the one in evals/server.pid left by an
        earlier orchestrator); on a remote policy host its serve processes."""
        host = self._policy_host()
        if host != "local":
            subprocess.run(["ssh", "-o", "BatchMode=yes", host, "pkill -f '[s]cripts/serve.sh'; pkill -f '[o]penpi_entry serve'; true"],
                           timeout=60, check=False)
            return
        pidfile = self.st.dir / "evals" / "server.pid"
        p, self._server = self._server, None
        if p is not None:
            _stop_group(p.pid, p)
        elif pidfile.exists():
            pgid = int(pidfile.read_text().split()[0])
            if any(b"serve.sh" in c or b"openpi_entry" in c for c in _group_cmdlines(pgid)):
                _stop_group(pgid)
        pidfile.unlink(missing_ok=True)

    def _fetch_checkpoint(self, checkpoint_dir):
        """Local path of a checkpoint to serve: params/ and assets/ copied from the agent host (same path) unless it was
        trained here; earlier copies of this run's checkpoints are removed first (6 GB each), the deployable one kept."""
        ckpt = Path(str(checkpoint_dir))
        if "://" in str(checkpoint_dir):
            return checkpoint_dir
        ckpt = ckpt if ckpt.is_absolute() else config.ROOT / ckpt
        if ((ckpt / "params").is_dir() or (ckpt / "model.safetensors").exists()) and not (ckpt / PULLED).exists():
            return ckpt                                  # trained here (JAX params/ or PyTorch model.safetensors)
        keep = {ckpt.resolve()} | ({Path(self.st.deployable_checkpoint).resolve()} if self.st.deployable_checkpoint else set())
        for m in (self.st.dir / "checkpoints").glob(f"*/*/{PULLED}"):
            if m.parent.resolve() not in keep:
                shutil.rmtree(m.parent, ignore_errors=True)
        host = self.cfg["hosts"]["agent"]
        self.op.notify(f"copying {ckpt} (params, assets) from {host}")
        for sub in ("params", "assets"):
            r = subprocess.run(["rsync", "-a", "--delete", "--mkpath", f"{host}:{ckpt}/{sub}/", f"{ckpt}/{sub}/"],
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"cannot copy {ckpt}/{sub} from {host}: {r.stderr.strip()[-300:]}")
        subprocess.run(["rsync", "-a", f"{host}:{ckpt.parent}/train_info.json", f"{ckpt.parent}/"], capture_output=True, text=True)
        (ckpt / PULLED).write_text(host + "\n")
        return ckpt

    def _sync_episodes(self):
        """Copy runs/<run>/episodes to the agent host (same path): convert runs there, the robot records here."""
        src, host = self.st.dir / "episodes", self.cfg["hosts"]["agent"]
        if not src.is_dir():
            return True
        r = subprocess.run(["rsync", "-a", "--mkpath", f"{src}/", f"{host}:{src}/"], capture_output=True, text=True)
        self.op.notify(f"episodes synced to {host}" if r.returncode == 0 else f"episode sync to {host} FAILED: {r.stderr.strip()[-300:]}")
        return r.returncode == 0

    def evaluate(self, checkpoint_dir, n=None, zero_shot=False, serve=None, rollout=None):
        """Serve checkpoint_dir on the policy host and run n rollout episodes (reset + rollout + verdict); writes evals/<id>.json.
        serve / rollout: agent 3's own server and rollout programs (its workspace, pulled here first)."""
        ev, hosts = self.cfg["evaluation"], self.cfg["hosts"]
        self._eval_code = {k: v for k, v in (("serve", serve), ("rollout", rollout)) if v}
        if self.cfg["evaluation"].get("dagger") and not zero_shot:
            self.deliver_notes()                        # the expert may need its note before it takes over
        if self._eval_code:
            ws = self.workspace("learn")
            try:
                ws.pull()
            except Exception as e:
                self.op.notify(f"learn workspace pull failed: {e!r}")
            self._eval_code["learn_commit"] = self._safe_head(ws)
        n, ckpt = int(n or ev["episodes"]), Path(str(checkpoint_dir))
        eid = f"{len(self.st.evals) + 1:02d}_{'zero_shot_' if zero_shot else ''}{ckpt.parent.name}_{ckpt.name}"
        port, ip = hosts["policy_port"], hosts["policy_server"]
        self.op.notify(f"evaluation {eid}: {checkpoint_dir}, {n} episodes, target {ev['target_success_rate']:.0%}")
        res = dict(id=eid, checkpoint=str(checkpoint_dir), zero_shot=zero_shot, requested=n, n=0, success=0, fail=0, unsure=0,
                   success_rate=0.0, met=False, episodes=[], t_start=now_iso(), port=port)
        if zero_shot:
            res["caveat"] = "base pi05 weights served with norm stats from our dataset; the model never saw this robot or task"
        res.update(self._eval_code)
        started = self._server_start(str(checkpoint_dir), port)
        try:
            if started is False or not self._server_wait(ip, port, ev.get("server_timeout_s", 900)):
                res["error"] = f"policy server {ip}:{port} did not come up"
            else:
                for _ in range(n):
                    idx, ep_dir = self._new_episode_dir()
                    targets = self._sample_targets()
                    before = self._reset_to(targets, save=ep_dir / "scene_before.jpg")
                    ep = self._episode(ep_dir, idx, targets, before, "eval") if before is not None else None
                    if ep is None:
                        shutil.rmtree(ep_dir, ignore_errors=True)
                        res["error"] = "stopped by the operator"
                        break
                    res["episodes"].append(dict(episode=idx, final=ep["final"], reason=ep["reason"], duration=ep["result"].duration))
                    if ev.get("dagger") and not zero_shot and ep["final"] != "success":
                        fix = self._correction(ep)
                        if fix is not None:
                            res.setdefault("dagger", []).append(dict(episode=fix["index"], of=idx, final=fix["final"]))
        finally:
            self._server_stop()
        eps = res["episodes"]
        res.update(n=len(eps), t_end=now_iso(), **{k: sum(e["final"] == k for e in eps) for k in ("success", "fail", "unsure")})
        res["success_rate"] = round(res["success"] / len(eps), 3) if eps else 0.0
        full = max(n, int(ev["episodes"]))              # only a full evaluation (evaluation.episodes rollouts) can meet the target
        res["met"] = len(eps) >= full and res["success_rate"] >= ev["target_success_rate"]
        d = self.st.dir / "evals"
        d.mkdir(exist_ok=True)
        (d / f"{eid}.json").write_text(json.dumps(res, indent=1, default=str))
        self.st.evals[eid] = {k: v for k, v in res.items() if k != "episodes"}
        if res["met"] and not zero_shot:
            self.st.deployable_checkpoint = str(checkpoint_dir)
        if zero_shot:
            self.st.zero_shot = dict(eval=eid, success_rate=res["success_rate"], n=len(eps), caveat=res["caveat"])
        self.st.event("eval", **self.st.evals[eid])
        self.st.save()
        self.st.ledger("eval", id=eid, checkpoint=str(checkpoint_dir), n=res["n"], success_rate=res["success_rate"], met=res["met"],
                       duration_s=round(time.time() - _ts(res["t_start"]), 1))
        self.op.notify(f"evaluation {eid}: {res['success']}/{len(eps)} success ({res['success_rate']:.0%}); target {'met' if res['met'] else 'not met'}")
        return res

    def evaluate_zero_shot(self, n=None):
        """Evaluate the base checkpoint (training.base_checkpoint) the same way; the caveat is recorded in the state."""
        return self.evaluate(self.cfg["training"]["base_checkpoint"], n, zero_shot=True)

    # ------------------------------------------------------------------ whole run
    def check(self):
        """rlpj.robot.check.run(): prints every check; a failure asks the operator whether to go on."""
        res = _mod("rlpj.robot.check").run(log=print)
        if res.get("ok"):
            return True
        bad = [c for c in res.get("checks", []) if c.get("status") == "FAIL"]
        self.op.notify("robot check failed: " + "; ".join(f"{c['name']} ({c.get('detail', '')})" for c in bad))
        return self.op.ask("continue anyway?", ["stop", "continue"]) == "continue"

    # ------------------------------------------------------------------ RL tasks (docs/RL_DESIGN.md)
    def _rl(self):
        return self.task.get("mode") == "rl"

    def _rl_env(self, mode, idx):
        ws = self.workspace("execute")
        (Path(ws.path) / "rl").mkdir(exist_ok=True)
        return {"RLPJ_RL_DIR": str(Path(ws.path) / "rl"), "RLPJ_RL_MODE": mode, "RLPJ_EPISODE": str(idx)}

    def _free_gpu_gb(self, host=None):
        """Free GPU memory (GB) on this machine or on host, 0 when unknown."""
        q = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits"
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, q] if host else q.split()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
            total, used = (float(x) for x in out.split(",")[:2])
            return (total - used) / 1024
        except Exception:
            return 0.0

    def _learner_host(self):
        """(host or None for this machine, device): the GPU that is free (this 4090 first, then the 5090), else CPU here."""
        need = float(self.task["rl"].get("learner_gpu_gb", 4))
        if self._free_gpu_gb() >= need:
            return None, "cuda"
        host = self.cfg["hosts"]["agent"]
        if self._free_gpu_gb(host) >= need:
            return host, "cuda"
        return None, "cpu"

    def _rl_learner(self, idx, check=False, host=None, device=None):
        """programs/learner.py on the free GPU (rl/ goes there first, rl/policy/ comes back); returns the attempt result."""
        ws, rl = self.workspace("execute"), self.task["rl"]
        if host is None and device is None:
            host, device = self._learner_host()
        attempt_dir = Path(ws.next_attempt_dir())
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in {**self._rl_env("train", idx), "RLPJ_DEVICE": device}.items())
        py = shlex.quote(str(config.ROOT / "third_party/openpi/.venv/bin/python"))
        cmd = f"env {env} {py} programs/learner.py" + (" --check" if check else "")
        what = f"learner {'check' if check else f'after episode {idx}'} ({host or 'robot machine'}, {device})"
        timeout = 180 if check else rl.get("learner_timeout_s", 300)
        if host is None:
            r = self._attempt(ws.path, cmd, attempt_dir, False, timeout_s=timeout, what=what, commit=self._safe_head(ws))
        else:
            self._push(ws)
            remote = f"cd {shlex.quote(str(ws.path))} && {cmd}"
            r = self._attempt(ws.path, f"ssh -o BatchMode=yes -o ServerAliveInterval=30 {shlex.quote(host)} {shlex.quote(remote)}",
                              attempt_dir, False, timeout_s=timeout, what=what, commit=self._safe_head(ws))
            if not check:
                subprocess.run(["rsync", "-a", "--delete", "--mkpath", f"{host}:{ws.path}/rl/policy/", f"{ws.path}/rl/policy/"],
                               capture_output=True, text=True, timeout=300)
        self.st.event("rl_learner", episode=idx, host=host or "local", device=device, check=check, exit_code=r.exit_code,
                      duration=r.duration, attempt=str(attempt_dir))
        return r

    def _accept_rl(self):
        """RL agent 'done': its last robot test (if any this session) was a complete episode with transitions, and
        programs/learner.py --check passes on the data recorded so far."""
        last = self._last.get("execute")
        if last is None and not self.st.rl["episodes"]:
            return False, "no robot episode yet: run programs/execute.py on the robot (moves_robot true) first"
        if last is not None:
            r = last["result"]
            if r.exit_code not in (0, 2):
                return False, f"the last robot run did not finish an episode (exit code {r.exit_code})"
            if not int(r.summary.get("transitions") or 0):
                return False, "the last robot run wrote no transitions (summary key 'transitions')"
        r = self._rl_learner(-1, check=True, host=None, device="cpu")
        if r.exit_code != 0:
            return False, f"programs/learner.py --check failed (exit code {r.exit_code}); see {r.attempt_dir.name}"
        return True, ""

    def _rl_episode(self, mode):
        """Reset, then one recorded RL episode (mode rl: training, rl_eval: evaluation); the episode dict or None (stop)."""
        idx, ep_dir = self._new_episode_dir()
        targets = self._sample_targets()
        self._reset_runs = []
        before = self._reset_to(targets, save=ep_dir / "scene_before.jpg")
        if before is None:
            shutil.rmtree(ep_dir, ignore_errors=True)
            return None
        env = self._rl_env("eval" if mode == "rl_eval" else "train", idx)
        return self._episode(ep_dir, idx, targets, before, mode, extra_env=env)

    def _rl_curve(self, **row):
        with open(self.st.dir / "rl_curve.jsonl", "a") as f:
            f.write(json.dumps(dict(t=now_iso(), **row)) + "\n")

    def _rl_eval(self):
        """Fixed evaluation: rl.eval_episodes episodes with RLPJ_RL_MODE=eval; returns the success rate or None (stop)."""
        rl, r = self.task["rl"], self.st.rl
        n, ok = int(rl["eval_episodes"]), 0
        self.op.notify(f"RL evaluation after {r['episodes']} training episodes: {n} episodes")
        for _ in range(n):
            self.restart_point()
            ep = self._rl_episode("rl_eval")
            if ep is None:
                return None
            ok += ep["final"] == "success"
        rate = ok / n
        r["evals"].append(dict(after=r["episodes"], success=ok, n=n, rate=round(rate, 3), t=now_iso()))
        self._rl_curve(kind="eval", after=r["episodes"], success=ok, n=n, rate=round(rate, 3))
        self.st.event("rl_eval", after=r["episodes"], success=ok, n=n)
        self.st.save()
        self.op.notify(f"RL evaluation after {r['episodes']} episodes: {ok}/{n}")
        return rate

    def _rl_review(self, recent):
        """Tuning turn: the RL agent gets the learning curve and recent failed episodes, may change its reward,
        learner, hyperparameters or episode program, and replies done (learner check again)."""
        r = self.st.rl
        evals = ", ".join(f"{e['success']}/{e['n']} after {e['after']}" for e in r["evals"][-6:]) or "none yet"
        train = self.st.counters.get("rl", {})
        head = (f"RL tuning turn after {r['episodes']} training episodes ({train.get('success', 0)} independent successes so "
                f"far); evaluations: {evals}. Look at rl_curve.jsonl, your learner log and these episodes; change the reward, "
                f"hyperparameters, learner or episode program if the curve calls for it (say why in notes/), check offline, "
                f"commit and reply done. The failed or unsure episodes:")
        r["since_review"] = 0
        self.st.save()
        return self._improve("execute", head, recent[-4:], "review")

    def rl_train(self):
        """RL loop: reset -> episode -> independent verdict -> learner, a fixed evaluation every rl.eval_every episodes,
        a tuning turn every rl.review_every; until the target success rate or rl.max_episodes. Returns the status."""
        rl, r = self.task["rl"], self.st.rl
        self.st.phase = "rl_train"
        self.st.save()
        self.deliver_notes()
        recent, crashes = [], 0
        while r["episodes"] < rl["max_episodes"]:
            self.restart_point()
            if r["episodes"] and r["episodes"] % rl["eval_every"] == 0 and not any(e["after"] == r["episodes"] for e in r["evals"]):
                rate = self._rl_eval()
                if rate is None:
                    return "stopped"
                if rate >= rl["target_success_rate"]:
                    return "target_met"
            if r["since_review"] >= rl["review_every"]:
                self._rl_review(recent)
                recent = []
            ep = self._rl_episode("rl")
            if ep is None:
                return "stopped"
            r["episodes"] += 1
            r["since_review"] += 1
            res = ep["result"]
            self._rl_curve(kind="train", episode=ep["index"], n=r["episodes"], final=ep["final"], reason=ep["reason"],
                           duration=res.duration, reward=res.summary.get("return"), transitions=res.summary.get("transitions"))
            if ep["final"] != "success":
                recent.append((ep["attempt_dir"], f"episode {ep['index']}: {ep['final']} - {ep['reason']}"))
            crashes = crashes + 1 if res.exit_code not in (0, 2) else 0
            if crashes:
                self._fails["execute"].append((ep["attempt_dir"], f"episode {ep['index']} crashed (exit {res.exit_code})"))
            if crashes >= 2:
                self._repair("execute")
                crashes = 0
            lr = self._rl_learner(ep["index"])
            r["learner_fails"] = 0 if lr.exit_code == 0 else r["learner_fails"] + 1
            if r["learner_fails"] >= 2:
                self._fails["execute"] = [(lr.attempt_dir, "programs/learner.py failed twice in a row")]
                self._repair("execute")
                r["learner_fails"] = 0
            self.st.save()
        rate = self._rl_eval()
        return "max_episodes" if rate is not None else "stopped"

    def run_all(self):
        """check -> develop execute -> develop reset -> collect -> learn (-> collect more) -> report; resumes from state.phase.
        RL tasks (mode rl): check -> develop reset -> develop execute (the RL agent) -> rl_train -> report."""
        if self._rl():
            return self._run_all_rl()
        col = self.cfg["collection"]
        phases = ("check", "develop_execute", "develop_reset", "collect", "learn", "report")
        phase = self.st.phase if self.st.phase in phases else "check"
        f = self.st.dir / "PHASE"                       # operator redirect: `echo learn > runs/<run>/PHASE`, then RESTART
        if f.exists():
            want = f.read_text().strip()
            f.unlink()
            if want in phases:
                self.st.event("phase_override", phase=want, was=phase)
                self.op.notify(f"operator: phase {phase} -> {want}")
                phase = want
        while True:
            self.st.phase = phase
            self.st.save()
            self.st.event("phase", phase=phase)
            if phase == "check":
                if not self.check():
                    return "check_failed"
                phase = "develop_execute"
            elif phase.startswith("develop_"):
                role = phase.split("_", 1)[1]
                if self.st.agent(role).get("status") != "done" and self.develop_program(role) != "done":
                    self.op.notify(f"{role} program is not ready; fix the situation and rerun run-all")
                    return phase
                phase = "develop_reset" if role == "execute" else "collect"
            elif phase == "collect":
                self.collect(col["round_size"])
                if self.st.collection.get("round_target"):      # round left open: the operator stopped it
                    return "collect"
                phase = "learn"
            elif phase == "learn":
                status = self.learn()
                phase = "collect" if status == "collect_more" and self.st.counters["episodes"]["total"] < col["max_episodes"] else "report"
            else:
                self.report()
                self.st.phase = "done"
                self.st.save()
                return "done"

    def _run_all_rl(self):
        phases = ("check", "develop_reset", "develop_execute", "rl_train", "report")
        phase = self.st.phase if self.st.phase in phases else "check"
        f = self.st.dir / "PHASE"
        if f.exists():
            want = f.read_text().strip()
            f.unlink()
            if want in phases:
                self.st.event("phase_override", phase=want, was=phase)
                phase = want
        while True:
            self.st.phase = phase
            self.st.save()
            self.st.event("phase", phase=phase)
            if phase == "check":
                if not self.check():
                    return "check_failed"
                phase = "develop_reset"
            elif phase.startswith("develop_"):
                role = phase.split("_", 1)[1]
                if self.st.agent(role).get("status") != "done" and self.develop_program(role) != "done":
                    self.op.notify(f"{role} program is not ready; fix the situation and rerun run-all")
                    return phase
                phase = "develop_execute" if role == "reset" else "rl_train"
            elif phase == "rl_train":
                status = self.rl_train()
                self.st.event("rl_status", status=status)
                if status == "stopped":
                    return "rl_train"
                phase = "report"
            else:
                self.report()
                self.st.phase = "done"
                self.st.save()
                return "done"

    def report(self):
        """Write runs/<run>/REPORT.md from the state; returns its path."""
        p = self.st.dir / "REPORT.md"
        p.write_text(report_md(self.st, self.cfg))
        return p

    def relabel(self, ep_dir, label, note=""):
        """Operator label for an episode: label file, meta.json outcome and the counters."""
        ep_dir = Path(ep_dir)
        prev = self.op.label(ep_dir, label, note)
        meta = ep_dir / "meta.json"
        mode = (json.loads(meta.read_text()) if meta.exists() else {}).get("mode")
        c = self.st.counters["evals" if mode == "eval" else "episodes"]
        if prev in c and label in c and prev != label:
            c[prev] -= 1
            c[label] += 1
        self.st.event("label", episode=str(ep_dir), label=label, prev=prev, note=note)
        self.st.save()
