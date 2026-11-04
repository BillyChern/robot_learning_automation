"""The phases of one run: develop_program (agents 1/2), collect, learn/evaluate (agent 3), run_all.
Other modules are imported inside the methods so each phase only needs what it uses."""
import importlib
import json
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from rlpj import config
from . import attempts
from .state import RunState, now_iso, report_md, update_meta

RESTART_MSG = "The orchestrator restarted. Summarize in one line where your work stands, then continue."


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
        return attempts.run_attempt(cwd, command, attempt_dir, moves_robot, record_dir=record_dir, extra_env=extra_env,
                                    timeout_s=timeout_s or self.cfg["collection"].get("attempt_timeout_s"))

    def _sample_targets(self):
        t = _mod("rlpj.perception.layout").sample(self.task, self.rng)
        return {k: [float(v[0]), float(v[1])] for k, v in t.items()}

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
        action, usage = runner.resume(RESTART_MSG) if self._session_id(role) else runner.start()
        status = self._agent_loop(role, action, usage)
        a["ended"] = now_iso()
        self.st.event("agent_status", role=role, status=status)
        self.st.save()
        self.op.notify(f"{role} agent: {status} after {a['attempts']} runs ({a['robot_attempts']} on the robot), ${a['cost_usd']:.2f}")
        return status

    def _agent_loop(self, role, action, usage):
        """Drive one agent session until an accepted done, give_up or agents.max_attempts runs; returns the status."""
        a, runner = self.st.agent(role), self.runner(role)
        n0, limit = a["attempts"], self.cfg["agents"]["max_attempts"]
        while True:
            self._account(role, usage)
            kind = _get(action, "action")
            self.st.event("agent", role=role, action=kind, what=_get(action, "what"), report=_get(action, "report"))
            if kind == "run":
                if a["attempts"] - n0 >= limit:
                    a["status"] = "max_attempts"
                    return "max_attempts"
                msg = self._agent_run(role, action)
            elif kind == "need_operator":
                self.op.notify(f"{role} agent needs you: {_get(action, 'report')}")
                msg = "Operator: " + (self.op.ask("reply to the agent (Enter = done, continue)") or "done, continue")
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
            action, usage = runner.resume(msg)

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
        ws.push()
        return _mod("rlpj.agents.runner").digest(attempt_dir) + (f"\n\northe orchestrator: {note}" if note else "")

    def _safe_head(self, ws):
        try:
            return ws.head()
        except Exception:
            return None

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
        a, ws = self.st.agent(role), self.workspace(role)
        a["repairs"] += 1
        a["status"] = "repairing"
        self._last[role] = None
        try:
            ws.push()
        except Exception as e:
            self.op.notify(f"workspace push failed: {e!r}")
        fails, dg = self._fails[role], _mod("rlpj.agents.runner").digest
        self.op.notify(f"{role} program failed {len(fails)} times in a row; asking the agent to repair it")
        lines = [f"repair your program; it failed {len(fails)} times in a row during "
                 f"{'collection' if role == 'execute' else 'the resets between episodes'}:"]
        for d, why in fails:
            lines += [f"--- {Path(d).name}: {why}", str(dg(d))]
        lines.append(f"Find the cause, fix programs/{role}.py, test it with a run (moves_robot true) and reply done when it works again.")
        self._fails[role] = []
        action, usage = self.runner(role).resume("\n".join(lines))
        status = self._agent_loop(role, action, usage)
        self.st.event("repair", role=role, status=status)
        self.st.save()
        return status

    # ------------------------------------------------------------------ collection
    def collect(self, n_target=None):
        """Episodes until this round has n_target successes or collection.max_episodes; returns the round's successes."""
        col, c = self.cfg["collection"], self.st.collection
        if not c.get("round_target"):
            c.update(round=c.get("round", 0) + 1, round_successes=0, round_target=int(n_target or col["round_size"]))
        self.st.phase = "collect"
        self.st.save()
        cf, ce = self.st.counters["consecutive_failures"], self.st.counters["episodes"]
        self.op.notify(f"collection round {c['round']}: {c['round_successes']}/{c['round_target']} successes, {ce['total']} episodes so far")
        while c["round_successes"] < c["round_target"] and ce["total"] < col["max_episodes"]:
            idx, ep_dir = self._new_episode_dir()
            targets = self._sample_targets()
            before = self._reset_to(targets, save=ep_dir / "scene_before.jpg")
            ep = self._episode(ep_dir, idx, targets, before, "collect") if before is not None else None
            if ep is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                self.op.notify("collection stopped by the operator")
                self.st.save()
                return c["round_successes"]
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
        self._pending_targets, fails = targets, 0
        try:
            while True:
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
                self._record_reset(targets, r, blobs, ok, rep)
                if ok:
                    cf["reset"] = 0
                    self._fails["reset"] = []
                    return blobs
                fails += 1
                cf["reset"] += 1
                self._fails["reset"].append((r.attempt_dir, f"layout check failed: {_reason(rep)}"))
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
        self.st.save()

    def _episode(self, ep_dir, idx, targets, before, mode):
        """One recorded robot execution (collect: the execute program, eval: the policy rollout) with its verdict."""
        if mode == "eval":
            cwd, attempt_dir, command = self.st.dir, ep_dir / "attempt", self._rollout_command()
            timeout, commit = self.cfg["evaluation"]["max_policy_seconds"] + 120, None
        else:
            ws = self.workspace("execute")
            cwd, attempt_dir, timeout, commit = Path(ws.path), Path(ws.next_attempt_dir()), None, self._safe_head(ws)
            command = self.st.agent("execute").get("program_command") or "python programs/execute.py"
        if not self.op.gate(f"{mode} episode {idx}", True, command):
            return None
        r = self._attempt(cwd, command, attempt_dir, True, record_dir=ep_dir, timeout_s=timeout, what=f"{mode} episode {idx}", commit=commit)
        after = self._safe_measure(ep_dir / "scene_after.jpg")
        aborted = r.outcome == "aborted"
        claim = "aborted" if aborted else (r.summary.get("claimed_outcome") or r.outcome)
        v = self._verdict(claim, before, after, _images(ep_dir) or _images(attempt_dir), aborted=aborted)
        outcome = dict(claim=claim, perception=v["perception"], vlm=v["vlm"], operator=None, final=v["final"], reason=v["reason"])
        update_meta(ep_dir, episode=idx, run_id=self.st.run_id, task=self.task["name"], instruction=self.task["instruction"],
                    mode=mode, attempt=str(attempt_dir), exit_code=r.exit_code, duration=r.duration, layout_target=targets,
                    layout_before=self._summ(before), layout_after=self._summ(after), outcome=outcome)
        c = self.st.counters["episodes" if mode == "collect" else "evals"]
        c["total"] += 1
        c[v["final"] if v["final"] in ("success", "unsure") else "fail"] += 1
        self.st.next_episode = idx + 1
        if v["final"] == "unsure" or self._safe("rlpj.success.verdict", "needs_operator", v):
            self.op.queue_review(ep_dir, v)
        self.st.event("episode", episode=idx, mode=mode, final=v["final"], reason=v["reason"], claim=claim,
                      exit_code=r.exit_code, duration=r.duration, attempt=str(attempt_dir))
        self.st.save()
        self.op.notify(f"episode {idx} ({mode}): {v['final']} - {v['reason']}")
        return dict(index=idx, dir=ep_dir, attempt_dir=attempt_dir, final=v["final"], reason=v["reason"], result=r)

    # ------------------------------------------------------------------ agent 3 and evaluation
    def dataset_stats(self):
        c = self.st.counters["episodes"]
        return dict(run_dir=str(self.st.dir), episodes_dir=str(self.st.dir / "episodes"), episodes=dict(c),
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
        a["rounds"] = a.get("rounds", 0) + 1
        self.st.save()
        self.workspace("learn", **self._learn_extras())
        runner = self.runner("learn")
        if self._session_id("learn"):
            action, usage = runner.resume(f"The orchestrator is back with the current data: {json.dumps(self.dataset_stats())}. "
                                          f"Convert the new episodes, train and evaluate again; training runs this round: {tr['max_runs']}.")
        else:
            action, usage = runner.start()
        n_train, status = 0, None
        while status is None:
            self._account("learn", usage)
            kind = _get(action, "action")
            self.st.event("agent", role="learn", action=kind, what=_get(action, "what"), report=_get(action, "report"))
            if kind == "run":
                command = (_get(action, "command") or "").strip()
                if self._is_train(command):
                    if n_train >= tr["max_runs"]:
                        status = "collect_more"
                        continue
                    n_train += 1
                msg, met = self._learn_run(action, command)
                if met:
                    status = "met"
            elif kind == "need_operator":
                self.op.notify(f"learn agent needs you: {_get(action, 'report')}")
                msg = "Operator: " + (self.op.ask("reply to the agent (Enter = done, continue)") or "done, continue")
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
                action, usage = runner.resume(msg)
        a["status"] = {"met": "done", "collect_more": "max_runs", "gave_up": "gave_up"}[status]
        a["ended"] = now_iso()
        self.st.event("agent_status", role="learn", status=status)
        self.st.save()
        self.op.notify(f"learn agent: {status} after {n_train} training runs, ${a['cost_usd']:.2f}")
        return status

    @staticmethod
    def _is_train(command):
        toks = shlex.split(command)
        return "train.sh" in command or ("train" in toks and "vla" in toks)

    def _learn_run(self, action, command):
        """One 'run' of agent 3: evaluations here on the robot, everything else over ssh on the agent host."""
        what, ws = _get(action, "what") or command, self.workspace("learn")
        toks = shlex.split(command)
        if toks and (toks[0] in ("evaluate", "eval") or toks[:2] == ["rlpj", "eval"]):
            ckpt = toks[toks.index("--checkpoint") + 1] if "--checkpoint" in toks[:-1] else next((t for t in toks[1:] if "/" in t), None)
            n = toks[toks.index("--n") + 1] if "--n" in toks[:-1] else next((t for t in toks[2:] if t.isdigit()), None)
            if not ckpt:
                return "usage: rlpj eval --checkpoint <checkpoint_dir> [--n N]", False
            res = self.evaluate(ckpt, int(n) if n else None)
            lines = [json.dumps({k: res.get(k) for k in ("id", "checkpoint", "n", "success", "success_rate", "met", "error")})]
            lines += [f"  episode {e['episode']}: {e['final']} ({e['reason']})" for e in res.get("episodes", [])]
            return "\n".join(lines), bool(res.get("met"))
        if "serve" in toks:
            return "not run: the orchestrator starts and stops the policy server itself during 'rlpj eval --checkpoint <dir>'.", False
        if not self.op.gate(what, False, command):
            return f"the operator {self.op.last} this command.", False
        if toks[:2] == ["rlpj", "vla"] and "--run-dir" not in toks:
            command += f" --run-dir {shlex.quote(str(self.st.dir))}"
        attempt_dir = Path(ws.next_attempt_dir())
        r = self._ssh(command, attempt_dir, timeout_s=self.cfg["training"].get("timeout_s"), what=what)
        self._note_vla(command, r)
        try:
            ws.push()
        except Exception as e:
            self.op.notify(f"workspace push failed: {e!r}")
        return f"{_mod('rlpj.agents.runner').digest(attempt_dir)}\n\n(exit code {r.exit_code}, {r.duration:.0f} s)", False

    def _ssh(self, command, attempt_dir, timeout_s=None, what=""):
        """Run a command from the repo root on the agent host (5090) as an attempt logged here ('rlpj ...' via uv run)."""
        host = self.cfg["hosts"]["agent"]
        if command.startswith("rlpj "):
            command = "uv run " + command
        remote = f"cd {shlex.quote(str(config.ROOT))} && {command}"
        return self._attempt(self.st.dir, f"ssh -o BatchMode=yes {shlex.quote(host)} {shlex.quote(remote)}", attempt_dir, False,
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
        elif "convert" in toks or "rlpj.vla.convert" in command:
            name = opt("--name") or f"dataset_{len(self.st.datasets) + 1}"
            self.st.datasets[name] = dict(command=command, exit_code=r.exit_code, duration_s=r.duration, t=now_iso(),
                                          only_success="--only-success" in toks, episodes=dict(self.st.counters["episodes"]),
                                          path=str(self.st.dir / "datasets" / name))
        self.st.save()

    def _rollout_command(self):
        h, ev = self.cfg["hosts"], self.cfg["evaluation"]
        return (f"{config.ROOT / '.venv/bin/python'} -m rlpj.vla.rollout --host {h['policy_server']} --port {h['policy_port']} "
                f"--instruction {shlex.quote(self.task['instruction'])} --max-seconds {ev['max_policy_seconds']}")

    def _server_start(self, checkpoint_dir, port):
        """nohup bash scripts/serve.sh <ckpt> <port> on the agent host; its log goes to runs/<run>/evals/ there."""
        host, log = self.cfg["hosts"]["agent"], self.st.dir / "evals" / f"server_{port}.log"
        remote = (f"mkdir -p {shlex.quote(str(log.parent))} && cd {shlex.quote(str(config.ROOT))} && "
                  f"nohup bash scripts/serve.sh {shlex.quote(str(checkpoint_dir))} {int(port)} > {shlex.quote(str(log))} 2>&1 < /dev/null &")
        subprocess.run(["ssh", "-o", "BatchMode=yes", host, remote], timeout=60, check=False)
        self.op.notify(f"policy server starting on {host}:{port} for {checkpoint_dir} (log {log})")

    def _server_wait(self, ip, port, timeout_s=900):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                socket.create_connection((ip, int(port)), timeout=2).close()
                return True
            except OSError:
                time.sleep(5)
        return False

    def _server_stop(self):
        host = self.cfg["hosts"]["agent"]
        subprocess.run(["ssh", "-o", "BatchMode=yes", host, "pkill -f '[s]cripts/serve.sh'; pkill -f '[o]penpi_entry serve'; true"],
                       timeout=60, check=False)

    def evaluate(self, checkpoint_dir, n=None, zero_shot=False):
        """Serve checkpoint_dir on the 5090 and run n rollout episodes (reset + rollout + verdict); writes evals/<id>.json."""
        ev, hosts = self.cfg["evaluation"], self.cfg["hosts"]
        n, ckpt = int(n or ev["episodes"]), Path(str(checkpoint_dir))
        eid = f"{len(self.st.evals) + 1:02d}_{'zero_shot_' if zero_shot else ''}{ckpt.parent.name}_{ckpt.name}"
        port, ip = hosts["policy_port"], hosts["policy_server"]
        self.op.notify(f"evaluation {eid}: {checkpoint_dir}, {n} episodes, target {ev['target_success_rate']:.0%}")
        res = dict(id=eid, checkpoint=str(checkpoint_dir), zero_shot=zero_shot, requested=n, n=0, success=0, fail=0, unsure=0,
                   success_rate=0.0, met=False, episodes=[], t_start=now_iso(), port=port)
        if zero_shot:
            res["caveat"] = "base pi05 weights served with norm stats from our dataset; the model never saw this robot or task"
        self._server_start(str(checkpoint_dir), port)
        try:
            if not self._server_wait(ip, port, ev.get("server_timeout_s", 900)):
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
        finally:
            self._server_stop()
        eps = res["episodes"]
        res.update(n=len(eps), t_end=now_iso(), **{k: sum(e["final"] == k for e in eps) for k in ("success", "fail", "unsure")})
        res["success_rate"] = round(res["success"] / len(eps), 3) if eps else 0.0
        res["met"] = len(eps) >= n and res["success_rate"] >= ev["target_success_rate"]
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

    def run_all(self):
        """check -> develop execute -> develop reset -> collect -> learn (-> collect more) -> report; resumes from state.phase."""
        col = self.cfg["collection"]
        phases = ("check", "develop_execute", "develop_reset", "collect", "learn", "report")
        phase = self.st.phase if self.st.phase in phases else "check"
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
                phase = "learn"
            elif phase == "learn":
                status = self.learn()
                phase = "collect" if status == "collect_more" and self.st.counters["episodes"]["total"] < col["max_episodes"] else "report"
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
