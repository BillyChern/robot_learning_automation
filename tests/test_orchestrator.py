"""Offline tests of rlpj.orchestrator: state, attempts (fake programs, Ctrl+C relay), operator, the loop with fake
agents/perception/success modules (sys.modules stubs), cli parsing. No robot, no network."""
import copy
import json
import os
import shlex
import signal
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from rlpj import config
from rlpj.orchestrator import attempts, cli, loop, operator, state

# A robot program stand-in: with RLPJ_MOTION=1 it pops the next outcome from its plan file; dry runs claim "dry".
FAKE_PROG = r'''
import json, os, sys
d = os.environ["RLPJ_ATTEMPT_DIR"]
plan = sys.argv[1]
targets = None
if "--targets" in sys.argv:
    t = sys.argv[sys.argv.index("--targets") + 1]
    targets = json.load(open(t)) if os.path.exists(t) else json.loads(t)
if os.environ.get("RLPJ_MOTION") == "1":
    lines = open(plan).read().split()
    outcome = lines[0] if lines else "fail"
    open(plan, "w").write("\n".join(lines[1:]))
else:
    outcome = "dry"
env = {k: os.environ.get(k) for k in ("RLPJ_MOTION", "RLPJ_RECORD_DIR", "RLPJ_RESET_TARGETS", "RLPJ_ATTEMPT_DIR")}
print("fake program", outcome, json.dumps(env))
if env["RLPJ_RECORD_DIR"]:
    os.makedirs(env["RLPJ_RECORD_DIR"], exist_ok=True)
    json.dump({"program": {"name": "fake"}, "fps": 15}, open(os.path.join(env["RLPJ_RECORD_DIR"], "meta.json"), "w"))
ok = outcome in ("success", "dry")
json.dump({"claimed_outcome": "success" if ok else "fail", "measured": targets, "env": env}, open(os.path.join(d, "summary.json"), "w"))
sys.exit(0 if ok else 2)
'''

SCENE = {}


def scene_defaults():
    SCENE.clear()
    SCENE.update(layout_ok=True, check=(True, {"reason": "bee in ring"}), vlm=True,
                 blobs=[{"xy": [0.45, -0.1], "height": 0.03}, {"xy": [0.40, 0.05], "height": 0.02}], cams_open=0)


class A:
    def __init__(self, action, command="", moves_robot=False, what="", report=""):
        self.action, self.command, self.moves_robot, self.what, self.report = action, command, moves_robot, what or command, report


class U:
    cost_usd, input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 0.5, 1000, 100, 50, 10
    turns, duration_ms, session_id = 2, 3000, "sid-fake"


class FakeWorkspace:
    created = {}

    def __init__(self, path):
        self.path = Path(path)
        self.session = json.loads((self.path / "session.json").read_text())
        self.pushed = self.pulled = 0
        FakeWorkspace.created[self.path.name] = self

    @classmethod
    def create(cls, run_dir, role, task_cfg, condition=None, **extra):
        ws = Path(run_dir) / "agents" / role
        (ws / "programs").mkdir(parents=True)
        (ws / "attempts").mkdir()
        (ws / "session.json").write_text(json.dumps(dict(role=role, condition=condition, session_id=None, extra=sorted(extra))))
        return cls(ws)

    def next_attempt_dir(self):
        n = len(list((self.path / "attempts").glob("attempt_*"))) + 1
        d = self.path / "attempts" / f"attempt_{n}"
        d.mkdir()
        return d

    def push(self):
        self.pushed += 1

    def pull(self):
        self.pulled += 1

    def head(self):
        return "abc1234"


class FakeRunner:
    script, log = {}, {}

    def __init__(self, workspace, model=None, effort=None, max_budget_usd=None):
        self.ws, self.role = workspace, workspace.session["role"]

    def start(self, prompt=None):
        self.ws.session["session_id"] = "sid-" + self.role
        return self._next("start", prompt)

    def resume(self, message):
        assert self.ws.session.get("session_id"), "resume before start"
        return self._next("resume", message)

    def _next(self, kind, msg):
        FakeRunner.log.setdefault(self.role, []).append((kind, msg))
        a = FakeRunner.script[self.role].pop(0)
        return (a(msg) if callable(a) else a), U()


def fake_digest(attempt_dir, **kw):
    a = Path(attempt_dir)
    return f"digest of {a.name}: exit {(a / 'exit_code').read_text().strip()}"


class FakeCam:
    def __init__(self):
        SCENE["cams_open"] += 1

    def depth_median(self, n=5):
        return np.zeros((720, 1280), np.float32)

    def latest(self):
        return np.zeros((720, 1280, 3), np.uint8), None, time.time()

    def close(self):
        SCENE["cams_open"] -= 1


def fake_decide(claim, perception, vlm):
    final = ("fail" if perception is False else "success") if claim == "success" else "fail"
    return dict(final=final, reason=f"fake claim={claim} perception={perception} vlm={vlm and vlm.get('success')}")


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    scene_defaults()
    FakeWorkspace.created.clear()
    FakeRunner.script.clear()
    FakeRunner.log.clear()
    mods = {
        "rlpj.agents.workspace": _mod("rlpj.agents.workspace", Workspace=FakeWorkspace, render=lambda *a, **k: "PROMPT"),
        "rlpj.agents.runner": _mod("rlpj.agents.runner", AgentRunner=FakeRunner, digest=fake_digest),
        "rlpj.robot.cams": _mod("rlpj.robot.cams", SideCam=FakeCam),
        "rlpj.robot.check": _mod("rlpj.robot.check", run=lambda log=None, **k: {"ok": True, "checks": []}),
        "rlpj.robot.recorder": _mod("rlpj.robot.recorder", update_meta=_update_meta),
        "rlpj.perception.tabletop": _mod("rlpj.perception.tabletop", cloud=lambda cam, **k: "P", detect=lambda P, workspace=None, **k: SCENE["blobs"],
                                         summarize=lambda blobs: [{"xy": b["xy"]} for b in blobs]),
        "rlpj.perception.layout": _mod("rlpj.perception.layout", sample=lambda task_cfg, rng=None: {"tape_roll": [0.45, -0.1], "foam_bee": [0.4, 0.05]},
                                       check=lambda task_cfg, targets, blobs: (SCENE["layout_ok"], {"reason": "layout " + ("ok" if SCENE["layout_ok"] else "off")})),
        "rlpj.success.checks": _mod("rlpj.success.checks", run=lambda task_cfg, b, a: SCENE["check"]),
        "rlpj.success.vlm_judge": _mod("rlpj.success.vlm_judge", judge=lambda images, task_cfg: dict(success=SCENE["vlm"], confidence=0.9, reason="vlm", cost_usd=0.01, images=sorted(images))),
        "rlpj.success.verdict": _mod("rlpj.success.verdict", decide=fake_decide, needs_operator=lambda v: v["final"] == "unsure"),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(attempts, "wrist_stream_free", lambda ip, **k: True)
    prog = tmp_path / "fake_prog.py"
    prog.write_text(FAKE_PROG)
    return prog


def _update_meta(ep_dir, **fields):
    p = Path(ep_dir) / "meta.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    for k, v in fields.items():
        meta[k] = {**meta[k], **v} if isinstance(v, dict) and isinstance(meta.get(k), dict) else v
    p.write_text(json.dumps(meta, indent=1, default=str))
    return meta


def _cfg(**collection):
    cfg = copy.deepcopy(config.load("bee_in_tape_roll"))
    cfg["collection"].update(round_size=2, max_episodes=10, refine_after_failures=2, manual_reset_after_failures=2)
    cfg["collection"].update(collection)
    cfg["agents"]["max_attempts"] = 6
    return cfg


def _op():
    op = operator.Operator(hands_free=True, countdown_s=0)
    op.tty = True
    return op


def _plan(tmp_path, name, *outcomes):
    p = tmp_path / f"{name}.txt"
    p.write_text("\n".join(outcomes))
    return p


# ------------------------------------------------------------------ state
def test_state_roundtrip(runs):
    rid = state.new_run_id("bee_in_tape_roll")
    assert rid.endswith("_bee") and len(rid) == len("20261104_1130_bee")
    st = state.RunState.create(rid, "bee_in_tape_roll", "text")
    st.counters["episodes"]["success"] += 1
    st.agent("execute")["attempts"] = 2
    st.phase = "collect"
    st.save()
    st.event("x", a=1)
    st.note("scene rearranged")
    st2 = state.RunState.open(rid)
    assert (st2.phase, st2.condition, st2.counters["episodes"]["success"], st2.agents["execute"]["attempts"]) == ("collect", "text", 1, 2)
    assert st2.operator_notes[0]["text"] == "scene rearranged"
    assert [e["kind"] for e in state.read_jsonl(st2.dir / "events.jsonl")] == ["created", "x"]
    assert not list(st2.dir.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        state.RunState.create(rid, "bee_in_tape_roll", "text")
    assert any("episodes 0" in l for l in state.status_lines(st2))
    assert "# Run" in state.report_md(st2)


# ------------------------------------------------------------------ attempts
def test_run_attempt_env_log_outcome(tmp_path):
    prog = tmp_path / "p.py"
    prog.write_text(FAKE_PROG)
    plan = _plan(tmp_path, "plan", "success", "fail")
    ad = tmp_path / "attempt_1"
    r = attempts.run_attempt(tmp_path, f"python {prog} {plan}", ad, moves_robot=True, record_dir=tmp_path / "ep", extra_env={"X": "y"})
    assert (r.exit_code, r.outcome, r.stopped_by_operator, r.stopped_by) == (0, "success", False, None)
    s = json.loads((ad / "summary.json").read_text())
    assert s["env"]["RLPJ_MOTION"] == "1" and s["env"]["RLPJ_RECORD_DIR"] == str(tmp_path / "ep") and s["env"]["RLPJ_ATTEMPT_DIR"] == str(ad)
    assert "fake program success" in (ad / "log.txt").read_text() and (ad / "console.txt").exists()
    assert (ad / "command.txt").read_text().strip() == f"python {prog} {plan}"
    assert (ad / "exit_code").read_text().strip() == "0" and (ad / "started").exists() and (ad / "ended").exists()
    meta = json.loads((ad / "attempt.json").read_text())
    assert meta["outcome"] == "success" and meta["moves_robot"] is True and "duration_s" in meta
    r = attempts.run_attempt(tmp_path, f"python {prog} {plan}", tmp_path / "a2", moves_robot=True)
    assert (r.exit_code, r.outcome) == (2, "fail")
    r = attempts.run_attempt(tmp_path, f"python {prog} {plan}", tmp_path / "a3", moves_robot=False)
    assert r.summary["env"]["RLPJ_MOTION"] is None and r.outcome == "success"
    r = attempts.run_attempt(tmp_path, "python -c 'raise SystemExit(3)'", tmp_path / "a4", moves_robot=False)
    assert (r.exit_code, r.outcome, r.summary) == (3, "error", {})


WAIT_PROG = r'''
import json, os, signal, sys, time
d = os.environ["RLPJ_ATTEMPT_DIR"]
def stop(*a):
    print("homing", flush=True)
    json.dump({"claimed_outcome": "fail", "stopped": True}, open(os.path.join(d, "summary.json"), "w"))
    sys.exit(130)
signal.signal(signal.SIGINT, stop)
open(os.path.join(d, "ready"), "w").close()
print("waiting", flush=True)
time.sleep(30)
'''


def test_run_attempt_relays_ctrl_c(tmp_path):
    prog = tmp_path / "wait.py"
    prog.write_text(WAIT_PROG)
    ad = tmp_path / "a"

    def ctrl_c():
        t0 = time.time()
        while not (ad / "ready").exists() and time.time() - t0 < 10:
            time.sleep(0.05)
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=ctrl_c, daemon=True).start()
    r = attempts.run_attempt(tmp_path, f"python {prog}", ad, moves_robot=False)
    assert (r.exit_code, r.outcome, r.stopped_by_operator, r.stopped_by) == (130, "aborted", True, "operator")
    assert r.duration < 10 and (ad / "operator_stop").exists() and "homing" in (ad / "log.txt").read_text()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_run_attempt_timeout(tmp_path):
    prog = tmp_path / "wait.py"
    prog.write_text(WAIT_PROG)
    r = attempts.run_attempt(tmp_path, f"python {prog}", tmp_path / "a", moves_robot=False, timeout_s=1)
    assert (r.outcome, r.stopped_by, r.stopped_by_operator, r.exit_code) == ("aborted", "timeout", False, 130)


# ------------------------------------------------------------------ operator
def test_operator_gate_and_ask(monkeypatch, tmp_path):
    op = operator.Operator(countdown_s=0)
    op.tty = True
    answers = iter(["", "n", "s", "", "c"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert op.gate("move the arm", True, "python x.py") is True and op.last == "go"
    assert op.gate("move", True) is False and op.last == "refused"
    assert op.gate("move", True) is False and op.last == "skipped"
    assert op.gate("dry run", False) is True
    assert op.ask("go on?", ["stop", "continue"]) == "continue"
    op.tty = False
    assert op.gate("move", True) is False and op.last == "refused"
    assert op.gate("dry", False) is True
    assert op.ask("q", ["stop", "continue"]) == "stop"
    hf = operator.Operator(hands_free=True, countdown_s=0)
    hf.tty = True
    assert hf.gate("move", True) is True and hf.gate("dry", False) is True and hf.ask("q", ["a", "b"]) == "a"
    assert operator.Operator(yes=True).gate("dry", False) is True
    ep = tmp_path / "runs" / "r" / "episodes" / "ep_000001"
    ep.mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({"outcome": {"claim": "success", "final": "unsure"}}))
    op.queue_review(ep, {"final": "unsure", "reason": "no evidence", "votes": {}})
    assert state.read_jsonl(ep.parent.parent / "operator_queue.jsonl")[0]["verdict"] == {"final": "unsure", "reason": "no evidence"}
    assert op.label(ep, "success", "bee is in") == "unsure"
    meta = json.loads((ep / "meta.json").read_text())
    assert meta["outcome"]["final"] == "success" and meta["outcome"]["operator"] == "success" and meta["outcome"]["claim"] == "success"
    assert state.read_jsonl(ep.parent.parent / "operator_labels.jsonl")[0]["label"] == "success"


# ------------------------------------------------------------------ develop_program
def test_develop_execute(fakes, runs, tmp_path):
    st = state.RunState.create("r1", "bee_in_tape_roll", "text")
    plan = _plan(tmp_path, "exec", "fail", "success", "success")
    cmd = f"python {fakes} {plan}"
    msgs = []

    def check_false(msg):
        SCENE["check"] = (False, {"reason": "bee next to the ring"})
        return A("run", cmd, moves_robot=True, what="place the bee")

    def check_true(msg):
        msgs.append(msg)
        SCENE["check"] = (True, {"reason": "bee in ring"})
        return A("run", cmd, moves_robot=True, what="place the bee again")

    FakeRunner.script["execute"] = [
        A("run", cmd, moves_robot=False, what="dry run"), A("done", report="works"),           # rejected: no robot run
        A("run", cmd, moves_robot=True, what="try"), A("done", report="works"),               # rejected: outcome fail
        check_false, A("done", report="works"),                                               # rejected: independent check says fail
        check_true, A("done", report="works")]                                                # accepted
    p = loop.Pipeline(st, _op(), _cfg())
    assert p.develop_program("execute") == "done"
    a = st.agents["execute"]
    assert (a["attempts"], a["robot_attempts"], a["first_success_attempt"], a["first_success_after_attempts"]) == (4, 3, 3, 4)
    assert a["status"] == "done" and a["program_command"] == cmd and a["session_id"] == "sid-fake"
    assert a["cost_usd"] == pytest.approx(0.5 * 8) and a["input_tokens"] == 8000 and a["cache_read"] == 400 and a["duration_s"] == pytest.approx(24)
    resumes = [m for k, m in FakeRunner.log["execute"] if k == "resume"]
    assert "no robot run" in resumes[1] and "outcome fail" in resumes[3] and "independent verdict says fail" in msgs[0]
    assert resumes[0].startswith("digest of attempt_1") and "the orchestrator" not in resumes[0]
    assert "independent geometric check: True" in resumes[6]
    ws = FakeWorkspace.created["execute"]
    assert ws.pulled == 4 and ws.pushed == 4 and SCENE["cams_open"] == 0
    s1 = json.loads((ws.path / "attempts/attempt_1/summary.json").read_text())
    s2 = json.loads((ws.path / "attempts/attempt_2/summary.json").read_text())
    assert s1["env"]["RLPJ_MOTION"] is None and s2["env"]["RLPJ_MOTION"] == "1"
    assert (ws.path / "attempts/attempt_2/scene_before.jpg").exists() and (ws.path / "attempts/attempt_2/scene_after.jpg").exists()
    assert st.phase == "develop_execute" and state.RunState.open("r1").agents["execute"]["status"] == "done"
    assert [e["kind"] for e in state.read_jsonl(st.dir / "events.jsonl")].count("attempt") == 4


def test_develop_reset_targets_and_acceptance(fakes, runs, tmp_path):
    st = state.RunState.create("r2", "bee_in_tape_roll", "text")
    plan = _plan(tmp_path, "reset", "success", "success")
    cmd = f"python {fakes} {plan}"
    inline = cmd + " --targets " + shlex.quote(json.dumps({"tape_roll": [0.5, -0.1], "foam_bee": [0.4, 0.0]}))

    def fix(msg):
        SCENE["layout_ok"] = True
        return A("run", inline, moves_robot=True, what="place both")

    SCENE["layout_ok"] = False
    FakeRunner.script["reset"] = [A("run", cmd, moves_robot=True, what="place both"), A("done", report="ok"), fix, A("done", report="ok")]
    p = loop.Pipeline(st, _op(), _cfg())
    assert p.develop_program("reset") == "done"
    ws = FakeWorkspace.created["reset"]
    s1 = json.loads((ws.path / "attempts/attempt_1/summary.json").read_text())
    assert s1["measured"] == {"tape_roll": [0.45, -0.1], "foam_bee": [0.4, 0.05]}          # sampled targets were passed
    assert s1["env"]["RLPJ_RESET_TARGETS"] == str(ws.path / "attempts/attempt_1/targets.json")
    s2 = json.loads((ws.path / "attempts/attempt_2/summary.json").read_text())
    assert s2["measured"] == {"tape_roll": [0.5, -0.1], "foam_bee": [0.4, 0.0]}            # the agent's inline targets
    resumes = [m for k, m in FakeRunner.log["reset"] if k == "resume"]
    assert "layout check FAILED" in resumes[0] and "within tolerance" in resumes[1]
    a = st.agents["reset"]
    assert a["program_command"] == cmd and a["first_success_attempt"] == 2 and a["status"] == "done"


def test_develop_give_up_and_max_attempts(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("r3", "bee_in_tape_roll", "text")
    cmd = f"python {fakes} {_plan(tmp_path, 'x', 'fail')}"
    monkeypatch.setattr("builtins.input", lambda *a: "bee is back on the table")
    FakeRunner.script["execute"] = [A("need_operator", report="put the bee back"), A("give_up", report="cannot grasp")]
    p = loop.Pipeline(st, _op(), _cfg())
    assert p.develop_program("execute") == "gave_up" and st.agents["execute"]["last_report"] == "cannot grasp"
    assert FakeRunner.log["execute"][1][1] == "Operator: bee is back on the table"
    st2 = state.RunState.create("r4", "bee_in_tape_roll", "text")
    cfg = _cfg()
    cfg["agents"]["max_attempts"] = 2
    FakeRunner.script["execute"] = [A("run", cmd, what="d1"), A("run", cmd, what="d2"), A("run", cmd, what="d3")]
    assert loop.Pipeline(st2, _op(), cfg).develop_program("execute") == "max_attempts" and st2.agents["execute"]["attempts"] == 2


def test_operator_refusal_message(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("r5", "bee_in_tape_roll", "text")
    cmd = f"python {fakes} {_plan(tmp_path, 'x', 'success')}"
    op = operator.Operator(countdown_s=0)
    op.tty = True
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    FakeRunner.script["execute"] = [A("run", cmd, moves_robot=True, what="swing wildly"), A("give_up", report="ok")]
    loop.Pipeline(st, op, _cfg()).develop_program("execute")
    assert "operator refused" in FakeRunner.log["execute"][1][1] and st.agents["execute"]["attempts"] == 0


# ------------------------------------------------------------------ collect
def _ready_run(rid, fakes, tmp_path, exec_plan, reset_plan):
    st = state.RunState.create(rid, "bee_in_tape_roll", "text")
    for role, plan in (("execute", exec_plan), ("reset", reset_plan)):
        ws = FakeWorkspace.create(st.dir, role, {}, "text")
        ws.session["session_id"] = f"sid-{role}"                 # developed earlier: the session exists
        (ws.path / "session.json").write_text(json.dumps(ws.session))
        st.agent(role).update(status="done", session_id=f"sid-{role}", program_command=f"python {fakes} {plan}", started=state.now_iso())
    st.save()
    return st


def test_collect_refines_after_failures(fakes, runs, tmp_path):
    st = _ready_run("c1", fakes, tmp_path, _plan(tmp_path, "exec", "fail", "fail", "success", "success", "success"),
                    _plan(tmp_path, "reset", *["success"] * 10))
    cmd = st.agents["execute"]["program_command"]
    FakeRunner.script["execute"] = [A("run", cmd, moves_robot=True, what="test the fix"), A("done", report="fixed")]
    p = loop.Pipeline(st, _op(), _cfg())
    assert p.collect(2) == 2
    assert st.counters["episodes"] == dict(total=4, success=2, fail=2, unsure=0)
    assert st.counters["resets"] == dict(total=4, ok=4, fail=0, manual=0)
    assert st.counters["consecutive_failures"] == dict(execute=0, reset=0)
    assert st.agents["execute"]["repairs"] == 1 and st.collection == dict(round=1, round_successes=2, round_target=None)
    repair_msg = FakeRunner.log["execute"][0][1]
    assert repair_msg.startswith("repair your program; it failed 2 times in a row") and "digest of attempt_1" in repair_msg and "digest of attempt_2" in repair_msg
    eps = sorted(p.name for p in (st.dir / "episodes").iterdir())
    assert eps == ["ep_000000", "ep_000001", "ep_000002", "ep_000003"] and st.next_episode == 4
    meta = json.loads((st.dir / "episodes/ep_000000/meta.json").read_text())
    assert meta["outcome"]["final"] == "fail" and meta["outcome"]["claim"] == "fail" and meta["mode"] == "collect" and meta["episode"] == 0
    assert meta["layout_target"] == {"tape_roll": [0.45, -0.1], "foam_bee": [0.4, 0.05]} and meta["layout_before"] and meta["program"] == {"name": "fake"}
    meta3 = json.loads((st.dir / "episodes/ep_000003/meta.json").read_text())
    assert meta3["outcome"]["final"] == "success" and meta3["outcome"]["vlm"]["images"] == ["scene_after", "scene_before"]
    ws = FakeWorkspace.created["execute"]
    s = json.loads((ws.path / "attempts/attempt_1/summary.json").read_text())
    assert s["env"]["RLPJ_RECORD_DIR"] == str(st.dir / "episodes/ep_000000") and s["env"]["RLPJ_MOTION"] == "1"
    assert len(list((ws.path / "attempts").iterdir())) == 5 and FakeWorkspace.created["reset"].pushed == 0
    resets = sorted((st.dir / "resets").glob("reset_*.json"))
    assert len(resets) == 4 and json.loads(resets[0].read_text())["ok"] is True
    assert SCENE["cams_open"] == 0 and not list(st.dir.glob("operator_queue.jsonl"))


def test_collect_manual_reset_and_review_queue(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("c2", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", *["success"] * 5))
    SCENE["layout_ok"] = False
    op = _op()
    calls = []

    def manual(targets):
        calls.append(targets)
        SCENE["layout_ok"] = True
        return True

    monkeypatch.setattr(op, "wait_manual_reset", manual)
    p = loop.Pipeline(st, op, _cfg(manual_reset_after_failures=2, refine_after_failures=5, max_episodes=1))
    monkeypatch.setitem(sys.modules, "rlpj.success.verdict", _mod("rlpj.success.verdict", decide=lambda c, p, v: dict(final="unsure", reason="no evidence"),
                                                                 needs_operator=lambda v: v["final"] == "unsure"))
    assert p.collect(1) == 0                            # the only episode is unsure, not a success; max_episodes=1 ends the round
    assert st.counters["resets"] == dict(total=2, ok=0, fail=2, manual=1) and calls == [{"tape_roll": [0.45, -0.1], "foam_bee": [0.4, 0.05]}]
    assert st.counters["episodes"]["unsure"] == 1
    q = state.pending_reviews(st)
    assert len(q) == 1 and q[0]["episode"].endswith("ep_000000")
    p.relabel(q[0]["episode"], "success", "looked fine")
    assert st.counters["episodes"] == dict(total=1, success=1, fail=0, unsure=0) and state.pending_reviews(st) == []


def test_collect_stops_when_operator_refuses(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("c3", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", "success"))
    op = operator.Operator(countdown_s=0)
    op.tty = True
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert loop.Pipeline(st, op, _cfg()).collect(1) == 0
    assert st.counters["episodes"]["total"] == 0 and not (st.dir / "episodes" / "ep_000000").exists()


# ------------------------------------------------------------------ evaluate / learn
def test_evaluate(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("e1", fakes, tmp_path, _plan(tmp_path, "exec"), _plan(tmp_path, "reset", *["success"] * 5))
    plan = _plan(tmp_path, "rollout", "success", "fail")
    server = []
    monkeypatch.setattr(loop.Pipeline, "_server_start", lambda self, ckpt, port: server.append(("start", ckpt, port)))
    monkeypatch.setattr(loop.Pipeline, "_server_wait", lambda self, ip, port, timeout_s=0: True)
    monkeypatch.setattr(loop.Pipeline, "_server_stop", lambda self: server.append(("stop",)))
    monkeypatch.setattr(loop.Pipeline, "_rollout_command", lambda self: f"python {fakes} {plan}")
    p = loop.Pipeline(st, _op(), _cfg())
    res = p.evaluate(st.dir / "checkpoints/lora1/999", n=2)
    assert (res["n"], res["success"], res["success_rate"], res["met"]) == (2, 1, 0.5, False)
    assert server == [("start", str(st.dir / "checkpoints/lora1/999"), 8000), ("stop",)]
    assert list(st.evals) == ["01_lora1_999"] and (st.dir / "evals/01_lora1_999.json").exists() and st.deployable_checkpoint is None
    assert st.counters["evals"] == dict(total=2, success=1, fail=1, unsure=0) and st.counters["episodes"]["total"] == 0
    meta = json.loads((st.dir / "episodes/ep_000000/meta.json").read_text())
    assert meta["mode"] == "eval" and (st.dir / "episodes/ep_000000/attempt/summary.json").exists()
    z = p.evaluate_zero_shot(n=1)                      # the plan is used up: the rollout fails
    assert z["zero_shot"] and z["met"] is False and st.zero_shot["caveat"] and st.deployable_checkpoint is None
    assert list(st.evals)[1].startswith("02_zero_shot_") and st.counters["evals"]["total"] == 3


def test_rollout_command_from_config(fakes, runs, tmp_path):
    st = state.RunState.create("e2", "bee_in_tape_roll", "text")
    c = loop.Pipeline(st, _op(), _cfg())._rollout_command()
    assert "-m rlpj.vla.rollout --host 100.122.29.80 --port 8000 --instruction 'put the foam bee in the tape roll' --max-seconds 150" in c


def test_learn_runs_commands_and_meets_target(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("l1", "bee_in_tape_roll", "text")
    st.counters["episodes"].update(total=5, success=4, fail=1)
    ssh = []

    def fake_ssh(self, command, attempt_dir, timeout_s=None, what=""):
        ssh.append(command)
        return attempts.run_attempt(tmp_path, f"echo remote: {shlex.quote(command)}", attempt_dir, False)

    def fake_eval(self, ckpt, n=None, zero_shot=False):
        st.deployable_checkpoint = str(ckpt)
        return dict(id="01", checkpoint=str(ckpt), n=n, success=n, success_rate=1.0, met=True, episodes=[dict(episode=0, final="success", reason="r")])

    monkeypatch.setattr(loop.Pipeline, "_ssh", fake_ssh)
    monkeypatch.setattr(loop.Pipeline, "evaluate", fake_eval)
    FakeRunner.script["learn"] = [
        A("done", report="nothing to do"),                                                              # rejected
        A("run", "rlpj vla convert --name bee_v1 --only-success", what="convert"),
        A("run", "rlpj vla serve --checkpoint x", what="serve"),                                        # not run
        A("run", "rlpj vla train --dataset bee_v1 --exp-name lora1 --num-train-steps 100", what="train"),
        A("run", "rlpj eval --checkpoint runs/l1/checkpoints/lora1/99 --n 3", what="evaluate")]
    cfg = _cfg()
    cfg["training"]["max_runs"] = 2
    assert loop.Pipeline(st, _op(), cfg).learn() == "met"
    assert ssh == [f"rlpj vla convert --name bee_v1 --only-success --run-dir {st.dir}",
                   f"rlpj vla train --dataset bee_v1 --exp-name lora1 --num-train-steps 100 --run-dir {st.dir}"]
    resumes = [m for k, m in FakeRunner.log["learn"] if k == "resume"]
    assert "not accepted" in resumes[0] and "digest of attempt_1" in resumes[1] and "not run" in resumes[2]
    assert st.datasets["bee_v1"]["only_success"] and st.training_runs["lora1"]["dataset"] == "bee_v1"
    assert st.deployable_checkpoint == "runs/l1/checkpoints/lora1/99" and st.agents["learn"]["status"] == "done"
    assert json.loads((st.dir / "agents/learn/session.json").read_text())["extra"] == ["DATASET", "EPISODES_DIR", "EPISODES_NOTE", "HISTORY", "N_EPISODES"]


def test_learn_max_runs_means_collect_more(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("l2", "bee_in_tape_roll", "text")
    monkeypatch.setattr(loop.Pipeline, "_ssh", lambda self, command, attempt_dir, timeout_s=None, what="":
                        attempts.run_attempt(tmp_path, "true", attempt_dir, False))
    FakeRunner.script["learn"] = [A("run", "bash scripts/train.sh runs/l2 bee_v1 lora1", what="t1"),
                                  A("run", "bash scripts/train.sh runs/l2 bee_v1 lora2", what="t2")]
    cfg = _cfg()
    cfg["training"]["max_runs"] = 1
    p = loop.Pipeline(st, _op(), cfg)
    assert p.learn() == "collect_more" and list(st.training_runs) == ["lora1"] and st.agents["learn"]["status"] == "max_runs"
    FakeRunner.script["learn"] = [A("give_up", report="no data")]
    assert p.learn() == "gave_up" and FakeRunner.log["learn"][-1][0] == "resume" and st.agents["learn"]["rounds"] == 2


# ------------------------------------------------------------------ run_all
def test_run_all_resumes_from_phase(fakes, runs, monkeypatch):
    st = state.RunState.create("a1", "bee_in_tape_roll", "text")
    p = loop.Pipeline(st, _op(), _cfg())
    calls, learn_results = [], iter(["collect_more", "met"])
    monkeypatch.setattr(p, "develop_program", lambda role: calls.append(("develop", role)) or "done")
    monkeypatch.setattr(p, "collect", lambda n=None: calls.append(("collect", n)) or 2)
    monkeypatch.setattr(p, "learn", lambda: calls.append(("learn",)) or next(learn_results))
    assert p.run_all() == "done"
    assert calls == [("develop", "execute"), ("develop", "reset"), ("collect", 2), ("learn",), ("collect", 2), ("learn",)]
    assert st.phase == "done" and (st.dir / "REPORT.md").exists()
    st.phase, calls[:] = "collect", []
    learn_results = iter(["met"])
    assert p.run_all() == "done" and calls == [("collect", 2), ("learn",)]
    st.phase = "develop_reset"
    monkeypatch.setattr(p, "develop_program", lambda role: "gave_up")
    assert p.run_all() == "develop_reset"


# ------------------------------------------------------------------ cli
def test_cli_parsing_and_new_status(runs, capsys):
    pr = cli.build_parser()
    a = pr.parse_args(["--hands-free", "--yes", "collect", "r1", "--n", "3"])
    assert (a.hands_free, a.yes, a.cmd, a.run, a.n) == (True, True, "collect", "r1", 3)
    a = pr.parse_args(["--task", "cup_down_stacking", "new", "--condition", "one_demo"])
    assert (a.task, a.condition) == ("cup_down_stacking", "one_demo")
    assert pr.parse_args(["eval", "r1", "--checkpoint", "/c", "--n", "5"]).checkpoint == "/c"
    assert cli.split(["--yes", "fit-kin", "--robot", "x"]) == (["--yes", "fit-kin"], ["--robot", "x"])
    assert cli.split(["vla", "train", "--dataset", "d"]) == (["vla"], ["train", "--dataset", "d"])
    assert cli.main(["vla"]) == 2 and cli.main(["vla", "train", "--dataset", "d"]) == 2
    assert pr.parse_args(["develop", "r1", "reset"]).role == "reset"
    with pytest.raises(SystemExit):
        pr.parse_args(["develop", "r1", "learn"])
    capsys.readouterr()
    assert cli.main(["new", "--condition", "text"]) == 0
    rid = capsys.readouterr().out.strip()
    assert rid.endswith("_bee") and (runs / rid / "state.json").exists()
    assert cli.main(["status", rid]) == 0
    out = capsys.readouterr().out
    assert f"run {rid}" in out and "condition text" in out
    assert cli.main(["report", rid]) == 0 and (runs / rid / "REPORT.md").exists()
    assert cli.main(["review", rid]) == 0 and "nothing to review" in capsys.readouterr().out
