"""Offline tests of rlpj.orchestrator: state, attempts (fake programs, Ctrl+C relay), operator, the loop with fake
agents/perception/success modules (sys.modules stubs), cli parsing. No robot, no network."""
import copy
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import cv2
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
env = {k: os.environ.get(k) for k in ("RLPJ_MOTION", "RLPJ_RECORD_DIR", "RLPJ_RESET_TARGETS", "RLPJ_ATTEMPT_DIR", "RLPJ_START_JOINTS", "RLPJ_DAGGER")}
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

    def push(self, full=None):
        self.pushed += 1

    def remote_exists(self):
        return True

    def trust(self):
        pass

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
    if claim is None:                                                              # policy rollout: checks decide
        final = "success" if perception is not False and (vlm or {}).get("success") else "fail"
    else:
        final = ("fail" if perception is False else "success") if claim == "success" else "fail"
    return dict(final=final, reason=f"fake claim={claim} perception={perception} vlm={vlm and vlm.get('success')}")


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(loop.Pipeline, "_archive_transcript", lambda self, role: None)    # offline: no rsync to the agent host
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



def _vlm_says(monkeypatch, *answers):
    """The (fake) VLM judge answers these in turn: a policy rollout's verdict comes from the independent checks."""
    it = iter(answers)
    monkeypatch.setitem(sys.modules, "rlpj.success.vlm_judge", _mod("rlpj.success.vlm_judge", judge=lambda images, task_cfg: dict(
        success=next(it), confidence=0.9, reason="vlm", cost_usd=0.01, images=sorted(images))))

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
    cfg["collection"]["reflect_after_every_run"] = False      # its own test below
    cfg["training"].update(full_finetune_only=False, min_train_steps=0)   # the operator's training policy has its own test
    cfg["hosts"]["train_host"] = None                                     # local training has its own test
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
    assert st.agents["execute"]["repairs"] == 1
    assert {k: st.collection[k] for k in ("round", "round_successes", "round_target")} == dict(round=1, round_successes=2, round_target=None)
    assert st.collection["window"]["n"] == 4 and st.collection["window"]["fail"] == 2
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
    p = loop.Pipeline(st, op, _cfg(manual_reset_after_failures=2, refine_after_failures=5, max_episodes=1, discard_unsure=False))
    monkeypatch.setitem(sys.modules, "rlpj.success.verdict", _mod("rlpj.success.verdict", decide=lambda c, p, v: dict(final="unsure", reason="no evidence"),
                                                                 needs_operator=lambda v: v["final"] == "unsure"))
    assert p.collect(1) == 0                            # the only episode is unsure, not a success; max_episodes=1 ends the round
    assert st.counters["resets"] == dict(total=2, ok=0, fail=2, manual=1) and calls == [{"tape_roll": [0.45, -0.1], "foam_bee": [0.4, 0.05]}]
    assert st.counters["episodes"]["unsure"] == 1
    q = state.pending_reviews(st)
    assert len(q) == 1 and q[0]["episode"].endswith("ep_000000")
    p.relabel(q[0]["episode"], "success", "looked fine")
    assert st.counters["episodes"] == dict(total=1, success=1, fail=0, unsure=0) and state.pending_reviews(st) == []


def test_collect_deletes_and_redoes_unsure_episodes(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("c2b", fakes, tmp_path, _plan(tmp_path, "exec", *["success"] * 4), _plan(tmp_path, "reset", *["success"] * 6))
    finals = iter(["unsure", "success", "unsure", "success"])
    monkeypatch.setitem(sys.modules, "rlpj.success.verdict", _mod("rlpj.success.verdict", decide=lambda c, p, v: dict(final=next(finals), reason="checks"),
                                                                 needs_operator=lambda v: v["final"] == "unsure"))
    p = loop.Pipeline(st, _op(), _cfg(discard_unsure=True))
    assert p.collect(2) == 2
    assert st.counters["episodes"] == dict(total=2, success=2, fail=0, unsure=0) and st.collection["discarded"] == 2
    assert sorted(d.name for d in (st.dir / "episodes").iterdir()) == ["ep_000000", "ep_000001"] and st.next_episode == 2
    assert json.loads((st.dir / "episodes/ep_000000/meta.json").read_text())["outcome"]["final"] == "success"
    assert state.pending_reviews(st) == [] and len(list((FakeWorkspace.created["execute"].path / "attempts").iterdir())) == 4
    gone = [e for e in state.read_jsonl(st.dir / "events.jsonl") if e["kind"] == "episode_discarded"]
    assert [e["episode"] for e in gone] == [0, 1]


def test_discarded_unsure_episodes_count_toward_max_episodes(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("c2c", fakes, tmp_path, _plan(tmp_path, "exec", *["success"] * 5), _plan(tmp_path, "reset", *["success"] * 5))
    monkeypatch.setitem(sys.modules, "rlpj.success.verdict", _mod("rlpj.success.verdict", decide=lambda c, p, v: dict(final="unsure", reason="checks"),
                                                                 needs_operator=lambda v: v["final"] == "unsure"))
    assert loop.Pipeline(st, _op(), _cfg(discard_unsure=True, max_episodes=3)).collect(2) == 0
    assert st.collection["discarded"] == 3 and st.counters["episodes"]["total"] == 0 and not list((st.dir / "episodes").iterdir())


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
    _vlm_says(monkeypatch, True, False)
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
    cfg = _cfg()
    c = loop.Pipeline(st, _op(), cfg)._rollout_command()
    host = cfg["hosts"]["policy_server"]
    secs = cfg["evaluation"]["max_policy_seconds"]
    assert f"-m rlpj.vla.rollout --host {host} --port 8000 --instruction 'put the foam bee in the tape roll' --max-seconds {secs}" in c


def test_fetch_checkpoint_pulls_and_prunes(runs, tmp_path, monkeypatch):
    st = state.RunState.create("e3", "bee_in_tape_roll", "text")
    p, calls = loop.Pipeline(st, _op(), _cfg()), []

    def fake_run(argv, **kw):
        calls.append(argv)
        Path(argv[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    here = st.dir / "checkpoints/local/9"
    (here / "params").mkdir(parents=True)
    assert p._fetch_checkpoint(here) == here and calls == []                       # trained here: served as is
    a, b = st.dir / "checkpoints/lora1/999", st.dir / "checkpoints/lora2/999"
    assert p._fetch_checkpoint(a) == a and (a / loop.PULLED).exists() and len(calls) == 3
    assert calls[2][-2].endswith("lora1/train_info.json")                         # config name and recipe for serving
    assert calls[0][:2] == ["rsync", "-a"] and calls[0][-2].endswith(f"{a}/params/")
    st.deployable_checkpoint = None
    assert p._fetch_checkpoint(b) == b and not a.exists() and here.exists()         # the earlier copy is pruned
    st.deployable_checkpoint = str(b)
    p._fetch_checkpoint(a)
    assert a.exists() and b.exists()                                                 # the deployable copy is kept
    assert p._fetch_checkpoint("gs://bucket/ckpt") == "gs://bucket/ckpt"


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

    synced = []
    monkeypatch.setattr(loop.Pipeline, "_ssh", fake_ssh)
    monkeypatch.setattr(loop.Pipeline, "evaluate", fake_eval)
    monkeypatch.setattr(loop.Pipeline, "_sync_episodes", lambda self: synced.append(len(ssh)) or True)
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
    assert synced == [0]                                                            # episodes pushed before convert
    assert st.datasets["bee_v1"]["only_success"] and st.training_runs["lora1"]["dataset"] == "bee_v1"
    assert st.deployable_checkpoint == "runs/l1/checkpoints/lora1/99" and st.agents["learn"]["status"] == "done"
    assert json.loads((st.dir / "agents/learn/session.json").read_text())["extra"] == ["DATASET", "EPISODES_DIR", "EPISODES_NOTE", "HISTORY", "N_EPISODES"]


def test_restart_during_learn_hands_over_the_last_result(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("l2", "bee_in_tape_roll", "text")
    st.counters["episodes"].update(total=5, success=5)

    def fake_ssh(self, command, attempt_dir, timeout_s=None, what=""):
        (st.dir / "RESTART").write_text("policy server moves to the 5090")       # requested while the run is going
        return attempts.run_attempt(tmp_path, "echo remote", attempt_dir, False)

    monkeypatch.setattr(loop.Pipeline, "_ssh", fake_ssh)
    monkeypatch.setattr(loop.Pipeline, "_sync_episodes", lambda self: True)
    FakeRunner.script["learn"] = [A("run", "rlpj vla train --dataset d --exp-name e --num-train-steps 10", what="train"),
                                  A("give_up", report="stop here")]
    with pytest.raises(SystemExit) as e:
        loop.Pipeline(st, _op(), _cfg()).learn()
    assert e.value.code == loop.RESTART_EXIT and "digest of attempt_1" in st.agents["learn"]["pending_msg"]
    st = state.RunState.open("l2")
    p2 = loop.Pipeline(st, _op(), _cfg())
    p2.workspace("learn").session["session_id"] = "sid-learn"                         # session.json persists for real
    assert p2.learn() == "gave_up"
    resumes = [m for k, m in FakeRunner.log["learn"] if k == "resume"]
    assert resumes[-1].startswith("The orchestrator restarted") and "digest of attempt_1" in resumes[-1]
    assert "pending_msg" not in st.agents["learn"] and st.agents["learn"]["rounds"] == 1


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


def test_local_server_stop_only_touches_our_server(runs, tmp_path):
    st = state.RunState.create("e4", "bee_in_tape_roll", "text")
    cfg = _cfg()
    cfg["hosts"].update(policy_host="local", policy_server="127.0.0.1")
    p = loop.Pipeline(st, _op(), cfg)
    fake = tmp_path / "serve.sh"
    fake.write_text("sleep 60\n")
    (st.dir / "evals").mkdir(exist_ok=True)
    out = subprocess.run(["bash", "-c", f"setsid bash {fake} >/dev/null 2>&1 & echo $!"], capture_output=True, text=True)
    stale = int(out.stdout)                                                        # left by an earlier orchestrator
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)              # unrelated process

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    try:
        (st.dir / "evals/server.pid").write_text(f"{other.pid}\n")
        p._server_stop()
        assert other.poll() is None and not (st.dir / "evals/server.pid").exists()
        (st.dir / "evals/server.pid").write_text(f"{stale}\n")
        t0 = time.time()
        p._server_stop()
        assert not alive(stale) and other.poll() is None and time.time() - t0 < 10
    finally:
        other.kill()
        other.wait()
        if alive(stale):
            os.killpg(stale, signal.SIGKILL)


def test_first_turn_needs_the_workspace_on_the_agent_host(runs):
    st = state.RunState.create("w1", "bee_in_tape_roll", "text")
    p, calls = loop.Pipeline(st, _op(), _cfg()), []

    class WS:
        path = st.dir / "agents/execute"

        def __init__(self, exists):
            self.exists = exists

        def remote_exists(self):
            return self.exists

        def push(self, full=None):
            calls.append(full)

        def trust(self):
            calls.append("trust")

    p._ensure_remote(WS(False))
    p._ensure_remote(WS(True))
    assert calls == [True, "trust", "trust"]


def test_usage_ledger_and_totals(runs):
    st = state.RunState.create("u1", "bee_in_tape_roll", "text")
    p = loop.Pipeline(st, _op(), _cfg())
    p._account("execute", U())
    p._account("execute", U())
    st.vlm.update(calls=1, cost_usd=0.03)
    st.event("attempt", role="execute", moves_robot=True, duration=120.0)
    st.event("attempt", role="execute", moves_robot=False, duration=30.0)
    st.event("episode", episode=0, duration=60.0)
    st.ledger("vlm", cost_usd=0.03)
    recs = state.read_jsonl(st.dir / "usage.jsonl")
    assert [r["kind"] for r in recs] == ["agent", "agent", "vlm"]
    assert recs[0]["cost_usd"] == 0.5 and recs[0]["input_tokens"] == 1000 and recs[0]["duration_s"] == 3.0
    t = recs[-1]["totals"]
    assert t["cost_usd"] == 1.03 and t["agents_usd"] == 1.0 and t["tokens_in"] == 2000 and t["agent_turns"] == 4
    assert t["robot_min"] == 3.0 and t["agent_min"] == 0.1
    line = state.totals_line(st)
    assert "cost $1.03" in line and "judge $0.03 / 1 calls" in line
    assert line in "\n".join(state.status_lines(st)) and line in state.report_md(st)


def test_timeline_stitches_runs_in_time_order(tmp_path):
    import cv2
    from rlpj.orchestrator import timeline
    run = tmp_path / "r"
    for role, n, started, moves, frames in (("reset", 1, 200.0, True, 3), ("execute", 1, 100.0, False, 2)):
        d = run / "agents" / role / "attempts" / f"attempt_{n}"
        d.mkdir(parents=True)
        (d / "attempt.json").write_text(json.dumps(dict(started=started, moves_robot=moves, what=f"{role} test", exit_code=0,
                                                        outcome="success", duration_s=1.0)))
        wr = cv2.VideoWriter(str(d / "dashboard.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (1280, 480))
        for _ in range(frames):
            wr.write(np.full((480, 1280, 3), 128, np.uint8))
        wr.release()
    assert [c[1].parents[1].name for c in timeline.clips(run)] == ["execute", "reset"]
    assert [c[1].parents[1].name for c in timeline.clips(run, robot_only=True)] == ["reset"]
    out, n, secs = timeline.build(run, tmp_path / "t.mp4", card_s=1.0, fps=8.0, log=lambda *a: None)
    cap = cv2.VideoCapture(str(out))
    assert n == 2 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2 * 8 + 3 + 2 and abs(secs - 21 / 8) < 1e-6


def test_transcript_renders_messages_tools_and_results(tmp_path):
    from rlpj.orchestrator import transcript
    d = tmp_path / "r" / "transcripts" / "-home-x-runs-r-agents-execute"
    d.mkdir(parents=True)
    rows = [dict(type="user", timestamp="2026-11-04T20:10:41.000Z", message=dict(content="You are the execute agent\nline 2")),
            dict(type="assistant", timestamp="2026-11-04T20:10:43.000Z", message=dict(content=[
                dict(type="thinking", thinking="", signature="x"), dict(type="text", text="Reading the API."),
                dict(type="tool_use", id="t1", name="Bash", input=dict(command="ls programs"))])),
            dict(type="user", timestamp="2026-11-04T20:10:44.000Z", message=dict(content=[dict(type="tool_result", tool_use_id="t1", content="execute.py")])),
            dict(type="queue-operation", timestamp="2026-11-04T20:10:45.000Z")]
    (d / "s1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    [p] = transcript.build(tmp_path / "r", log=lambda *a: None)
    md = p.read_text()
    assert p.name == "execute.md" and "orchestrator -> execute agent" in md and "You are the execute agent" in md
    assert "execute agent:** Reading the API." in md and "tool Bash:**\n\n```bash\nls programs\n```" in md and "execute.py" in md
    assert "signature" not in md and "queue-operation" not in md


def test_short_evaluation_cannot_meet_the_target(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("e5", fakes, tmp_path, _plan(tmp_path, "exec"), _plan(tmp_path, "reset", "success", "success"))
    plan = _plan(tmp_path, "rollout", "success", "success")
    monkeypatch.setattr(loop.Pipeline, "_server_start", lambda self, ckpt, port: True)
    monkeypatch.setattr(loop.Pipeline, "_server_wait", lambda self, ip, port, timeout_s=0: True)
    monkeypatch.setattr(loop.Pipeline, "_server_stop", lambda self: None)
    monkeypatch.setattr(loop.Pipeline, "_rollout_command", lambda self: f"python {fakes} {plan}")
    cfg = _cfg()
    res = loop.Pipeline(st, _op(), cfg).evaluate(st.dir / "checkpoints/lora1/999", n=2)
    assert cfg["evaluation"]["episodes"] > 2 and (res["n"], res["success_rate"]) == (2, 1.0)
    assert res["met"] is False and st.deployable_checkpoint is None                # 2/2 is a screening, not the target


def _auto_op():
    op = operator.Operator(autonomous=True, countdown_s=0)
    op.tty = True
    return op


def test_autonomous_never_waits_for_a_person(fakes, runs, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("autonomous mode asked for keyboard input"))
    op = _auto_op()
    assert op.hands_free and op.yes and op.autonomous
    st = state.RunState.create("a2", "bee_in_tape_roll", "text")
    FakeRunner.script["execute"] = [A("need_operator", report="put the bee in the bowl"), A("give_up", report="stop")]
    assert loop.Pipeline(st, op, _cfg()).develop_program("execute") == "gave_up"
    assert FakeRunner.log["execute"][1][1] == loop.NO_OPERATOR
    st2 = state.RunState.create("a3", "bee_in_tape_roll", "text")
    FakeRunner.script["execute"] = [A("need_operator", report="help")] * (loop.MAX_ASKS + 1)
    assert loop.Pipeline(st2, op, _cfg()).develop_program("execute") == "gave_up"
    assert "asked for an operator" in st2.agents["execute"]["last_report"]


def test_autonomous_reset_failures_go_to_the_agent_not_a_person(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("a4", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", *["success"] * 20))
    SCENE["layout_ok"] = False
    op = _auto_op()
    monkeypatch.setattr(op, "wait_manual_reset", lambda targets: pytest.fail("manual reset in autonomous mode"))
    repairs = []
    monkeypatch.setattr(loop.Pipeline, "_repair", lambda self, role: repairs.append(role) or "done")
    p = loop.Pipeline(st, op, _cfg(manual_reset_after_failures=2, refine_after_failures=5, max_episodes=1))
    assert p.collect(1) == 0                                      # resets never pass: the run stops instead of waiting
    assert repairs == ["reset"] * loop.AUTONOMOUS_RESET_REPAIRS and st.counters["resets"]["manual"] == 0
    assert any(e["kind"] == "autonomous_stop" for e in state.read_jsonl(st.dir / "events.jsonl"))


def test_restart_point_exits_at_a_safe_point(fakes, runs, tmp_path):
    st = _ready_run("x1", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", "success"))
    (st.dir / "RESTART").write_text("code update")
    with pytest.raises(SystemExit) as e:
        loop.Pipeline(st, _op(), _cfg()).collect(1)
    assert e.value.code == loop.RESTART_EXIT and not (st.dir / "RESTART").exists()
    assert state.read_jsonl(st.dir / "events.jsonl")[-1]["kind"] == "restart" and st.counters["episodes"]["total"] == 0


def test_periodic_review_asks_the_agents_to_improve(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("x2", fakes, tmp_path, _plan(tmp_path, "exec", "fail", "success", "success"), _plan(tmp_path, "reset", *["success"] * 6))
    asked = []
    monkeypatch.setattr(loop.Pipeline, "_improve", lambda self, role, head, items, kind: asked.append((role, kind, len(items), head)) or "done")
    cfg = _cfg(refine_after_failures=5)
    cfg["collection"]["review_every"] = 2
    p = loop.Pipeline(st, _op(), cfg)
    assert p.collect(2) == 2
    assert [(r, k, n) for r, k, n, _ in asked] == [("execute", "review", 1)] and "1 success, 1 fail" in asked[0][3]
    assert st.counters["episodes"]["total"] == 3 and st.counters["resets"]["total"] == 4    # one extra reset for the review


def test_agent3_can_choose_its_own_server_and_rollout(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("x3", "bee_in_tape_roll", "text")
    seen = []
    monkeypatch.setattr(loop.Pipeline, "_ssh", lambda self, command, attempt_dir, timeout_s=None, what="":
                        attempts.run_attempt(tmp_path, "true", attempt_dir, False))
    monkeypatch.setattr(loop.Pipeline, "evaluate", lambda self, ckpt, n=None, zero_shot=False, **code:
                        seen.append((ckpt, n, code)) or dict(id="01", n=n, success_rate=0.0, met=False, episodes=[]))
    FakeRunner.script["learn"] = [A("run", "rlpj eval --checkpoint runs/x3/checkpoints/e/9 --n 5 --serve programs/rtc_server.py "
                                           "--rollout programs/rtc_rollout.py", what="rtc screening"), A("give_up", report="enough")]
    assert loop.Pipeline(st, _op(), _cfg()).learn() == "gave_up"
    assert seen == [("runs/x3/checkpoints/e/9", 5, dict(serve="programs/rtc_server.py", rollout="programs/rtc_rollout.py"))]
    p = loop.Pipeline(st, _op(), _cfg())
    p._eval_code = dict(rollout="programs/rtc_rollout.py")
    c = p._rollout_command()
    assert "programs/rtc_rollout.py --host" in c and "-m rlpj.vla.rollout" not in c


def test_learn_workspace_may_research_the_web(tmp_path):
    from rlpj.agents import workspace
    run = tmp_path / "runs" / "r"
    (run / "agents").mkdir(parents=True)
    allow = lambda role: workspace.settings(run / "agents" / role, run, role, [])["permissions"]["allow"]
    assert {"WebSearch", "WebFetch"} <= set(allow("learn")) and "WebSearch" not in allow("execute")


def test_recipe_and_webview_status(runs, tmp_path):
    from rlpj.orchestrator import webview
    from rlpj.vla import openpi_entry
    (tmp_path / "my_recipe_x4.py").write_text("class C:\n    name = 'mine'\n\ndef configs(run_dir, repo_id):\n    return {'mine': C()}\n")
    assert list(openpi_entry.load_recipe(tmp_path / "my_recipe_x4.py", None, "rlpj/d")) == ["mine"]
    st = state.RunState.create("x4", "bee_in_tape_roll", "text")
    st.event("attempt", role="reset", outcome="success", what="real reset")
    (runs / "live").mkdir()
    (runs / "live" / "latest.json").write_text(json.dumps(dict(t=time.time(), attempt=str(st.dir / "agents/reset/attempts/attempt_3"), phase="place")))
    s = webview.status("x4")
    assert s["live"].startswith("LIVE: reset/attempts/attempt_3 phase place") and "run x4" in s["text"] and "real reset" in s["text"]


def test_agents_reflect_after_every_run_without_the_robot(fakes, runs, tmp_path):
    st = _ready_run("x5", fakes, tmp_path, _plan(tmp_path, "exec", "success", "success"), _plan(tmp_path, "reset", "success", "success"))
    cfg = _cfg()
    cfg["collection"]["reflect_after_every_run"] = True
    FakeRunner.script["execute"] = [A("run", "python programs/execute.py", moves_robot=True, what="try it"),
                                    A("done", report="no change"), A("done", report="faster lift")]
    FakeRunner.script["reset"] = [A("done", report="ok"), A("done", report="ok")]
    p = loop.Pipeline(st, _op(), cfg)
    assert p.collect(2) == 2 and not p._reflections                              # all reflections joined at the end
    ex = [m for k, m in FakeRunner.log["execute"] if k == "resume"]
    rs = [m for k, m in FakeRunner.log["reset"] if k == "resume"]
    assert ex[0].startswith("Reflection after collection episode 0: success") and ex[1] == loop.REFLECT_BUSY
    assert ex[2].startswith("Reflection after collection episode 1") and "no run actions" in ex[0]
    assert len(rs) == 2 and rs[0].startswith("Reflection after the reset for collection episode 0 (1 run(s), ok)")
    assert st.agents["execute"]["reflections"] == 2 and st.agents["reset"]["reflections"] == 2
    assert st.counters["episodes"]["total"] == 2 and st.counters["resets"]["total"] == 2   # reflections never ran the robot
    assert len([e for e in state.read_jsonl(st.dir / "events.jsonl") if e["kind"] == "reflect"]) == 5
    assert FakeWorkspace.created["execute"].pulled >= 2 and FakeWorkspace.created["reset"].pulled >= 2


def test_operator_note_reaches_the_agent_once(fakes, runs, tmp_path):
    st = _ready_run("x6", fakes, tmp_path, _plan(tmp_path, "exec", "success", "success"), _plan(tmp_path, "reset", "success", "success"))
    cfg = _cfg()
    cfg["collection"]["reflect_after_every_run"] = True
    (st.dir / "notes").mkdir()
    (st.dir / "notes" / "reset.md").write_text("the arm reaches 0.72 m tool-down at the table")
    FakeRunner.script["execute"] = [A("done", report="ok"), A("done", report="ok")]
    FakeRunner.script["reset"] = [A("done", report="ok"), A("done", report="ok")]
    assert loop.Pipeline(st, _op(), cfg).collect(2) == 2
    rs = [m for k, m in FakeRunner.log["reset"] if k == "resume"]
    assert rs[0].startswith("Note from the operator") and "0.72 m" in rs[0] and "Note from the operator" not in rs[1]
    assert not (st.dir / "notes" / "reset.md").exists() and len(list((st.dir / "notes").glob("reset.sent_*.md"))) == 1
    assert [e["role"] for e in state.read_jsonl(st.dir / "events.jsonl") if e["kind"] == "operator_note"] == ["reset"]


def test_raised_round_size_extends_an_open_round(fakes, runs, tmp_path):
    st = _ready_run("x6", fakes, tmp_path, _plan(tmp_path, "exec", "success", "success"), _plan(tmp_path, "reset", "success", "success"))
    st.collection.update(round=1, round_successes=1, round_target=1)
    p = loop.Pipeline(st, _op(), _cfg())
    assert p.collect(2) == 2 and st.collection["round"] == 1 and st.counters["episodes"]["total"] == 1


class FakeWristCam:
    def __init__(self, ip=None):
        self.t0 = time.time()

    def latest(self):
        return SCENE.get("wrist_img"), time.time() + 1.0

    def close(self):
        SCENE["wrist_opened"] = SCENE.get("wrist_opened", 0) + 1


def _wrist(monkeypatch, img):
    SCENE["wrist_img"] = img
    monkeypatch.setitem(sys.modules, "rlpj.robot.cams", _mod("rlpj.robot.cams", SideCam=FakeCam, WristCam=FakeWristCam))


def _gripper_cfg(**collection):
    cfg = _cfg(**collection)
    cfg["task"]["gripper_check"] = dict(object="foam_bee", hsv=[[15, 101, 91], [40, 255, 255]], bottom=0.15, max_frac=0.10, stop_after=2)
    return cfg


def test_object_on_the_gripper_skips_the_episode_and_stops_before_it_falls(fakes, runs, tmp_path, monkeypatch):
    stuck = cv2.imread(str(config.ROOT / "tests/data/wrist/stuck_at_home.jpg"))
    st = _ready_run("g1", fakes, tmp_path, _plan(tmp_path, "exec", *["success"] * 4), _plan(tmp_path, "reset", *["success"] * 6))
    _wrist(monkeypatch, stuck)
    p = loop.Pipeline(st, _op(), _gripper_cfg())
    assert p.collect(2) == 0                                   # reset 1 and reset 2 end with the bee on the fingers
    assert st.counters["episodes"]["total"] == 0 and st.counters["resets"]["total"] == 2 and SCENE["wrist_opened"] == 2
    ev = state.read_jsonl(st.dir / "events.jsonl")
    assert [e["in_a_row"] for e in ev if e["kind"] == "gripper_not_clear"] == [1, 2]
    assert any(e["kind"] == "autonomous_stop" and "stuck on the gripper" in e["reason"] for e in ev)
    note = (st.dir / "notes" / "reset.md").read_text()
    assert note.count("hanging on the gripper") == 2 and "put it down on the table" in note
    assert (FakeWorkspace.created["reset"].path / "attempts/attempt_1/gripper_check.jpg").exists()


def test_clear_gripper_lets_collection_go_on(fakes, runs, tmp_path, monkeypatch):
    empty = cv2.imread(str(config.ROOT / "tests/data/wrist/released_below.jpg"))
    st = _ready_run("g2", fakes, tmp_path, _plan(tmp_path, "exec", "success", "success"), _plan(tmp_path, "reset", "success", "success"))
    _wrist(monkeypatch, empty)
    p = loop.Pipeline(st, _op(), _gripper_cfg())
    assert p.collect(2) == 2 and p._stuck == 0 and not (st.dir / "notes" / "reset.md").exists()
    assert SCENE["wrist_opened"] == 4                          # after every robot program: 2 resets + 2 episodes


def test_bee_held_in_closed_jaws_is_not_counted_as_stuck(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("g3", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", "success"))
    _wrist(monkeypatch, cv2.imread(str(config.ROOT / "tests/data/wrist/stuck_at_home.jpg")))
    p = loop.Pipeline(st, _op(), _gripper_cfg())
    a = tmp_path / "held"
    a.mkdir()
    (a / "summary.json").write_text(json.dumps({"gripper_final": {"cmd": 1.0, "reading": 85.0}}))
    p._check_gripper(a, "reset")
    assert p._stuck == 0 and "held in the closed jaws (reading 85.0)" in (st.dir / "notes" / "reset.md").read_text()
    (a / "summary.json").write_text(json.dumps({"gripper_final": {"cmd": 0.0, "reading": 0.4}}))
    p._check_gripper(a, "reset")
    assert p._stuck == 1


def test_pending_notes_are_delivered_before_collection(fakes, runs, tmp_path):
    st = _ready_run("g4", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", "success"))
    (st.dir / "notes").mkdir()
    (st.dir / "notes" / "reset.md").write_text("temporary spots inside the release zone")
    FakeRunner.script["reset"] = [A("done", report="updated temp_spot")]
    p = loop.Pipeline(st, _op(), _cfg())
    p.deliver_notes()
    rs = [m for k, m in FakeRunner.log["reset"] if k == "resume"]
    assert len(rs) == 1 and rs[0].startswith("Note from the operator") and "Before collection continues" in rs[0]
    assert not (st.dir / "notes" / "reset.md").exists() and st.counters["resets"]["total"] == 0


def test_remote_server_start_backgrounds_only_the_server_and_survives_a_hung_ssh(fakes, runs, monkeypatch):
    st = state.RunState.create("s1", "bee_in_tape_roll", "text")
    cfg = _cfg()
    cfg["hosts"].update(policy_host="gpu_host", policy_server="10.0.0.5", policy_port=8000)
    p = loop.Pipeline(st, _op(), cfg)
    monkeypatch.setattr(loop.Pipeline, "_server_stop", lambda self: None)
    monkeypatch.setattr(loop.Pipeline, "_port_busy", lambda self, ip, port: False)
    calls = []

    def ok_run(args, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(loop.subprocess, "run", ok_run)
    assert p._server_start("/ckpt/14999", 8000) is True
    remote = calls[0][-1]
    assert calls[0][:5] == ["ssh", "-o", "BatchMode=yes", "-n", "gpu_host"]
    assert "&& (setsid nohup bash scripts/serve.sh /ckpt/14999 8000 >" in remote and remote.endswith("< /dev/null &)")

    def hung(args, **kw):
        raise subprocess.TimeoutExpired(args, 60)

    monkeypatch.setattr(loop.subprocess, "run", hung)
    assert p._server_start("/ckpt/14999", 8000) is False                     # reported, no crash


def test_dagger_expert_takes_over_after_a_failed_rollout(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("d1", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", *["success"] * 5))
    plan = _plan(tmp_path, "rollout", "success", "fail")
    monkeypatch.setattr(loop.Pipeline, "_server_start", lambda self, ckpt, port: True)
    monkeypatch.setattr(loop.Pipeline, "_server_wait", lambda self, ip, port, timeout_s=0: True)
    monkeypatch.setattr(loop.Pipeline, "_server_stop", lambda self: None)
    monkeypatch.setattr(loop.Pipeline, "_rollout_command", lambda self: f"python {fakes} {plan}")
    monkeypatch.setattr(loop.Pipeline, "_policy_end", lambda self, d: ([25.7, 47.3, 288.0, 178.1, 58.3, 46.9], 99.1))
    cfg = _cfg()
    cfg["evaluation"]["dagger"] = True
    p = loop.Pipeline(st, _op(), cfg)
    _vlm_says(monkeypatch, True, False, True)                               # rollout, rollout, correction
    res = p.evaluate(st.dir / "checkpoints/lora1/999", n=2)
    assert (res["n"], res["success"]) == (2, 1)                                 # the correction does not count as a rollout
    assert res["dagger"] == [dict(episode=2, of=1, final="success")]
    assert st.counters["dagger"] == dict(total=1, success=1, fail=0, unsure=0) and st.counters["evals"]["total"] == 2
    meta = json.loads((st.dir / "episodes/ep_000002/meta.json").read_text())
    assert meta["mode"] == "dagger" and meta["dagger_of"] == 1 and meta["dagger_start"]["gripper"] == 99.1
    s = json.loads(sorted((FakeWorkspace.created["execute"].path / "attempts").glob("attempt_*/summary.json"))[-1].read_text())
    assert json.loads(s["env"]["RLPJ_START_JOINTS"]) == [25.7, 47.3, 288.0, 178.1, 58.3, 46.9] and s["env"]["RLPJ_DAGGER"] == "1"


def test_no_dagger_correction_when_the_rollout_ended_holding_something(fakes, runs, tmp_path, monkeypatch):
    st = _ready_run("d2", fakes, tmp_path, _plan(tmp_path, "exec", "success"), _plan(tmp_path, "reset", "success"))
    p = loop.Pipeline(st, _op(), _cfg())
    monkeypatch.setattr(loop.Pipeline, "_policy_end", lambda self, d: ([0.0] * 6, 85.0))
    assert p._correction(dict(index=7, dir=st.dir / "episodes/ep_000007")) is None and "dagger" not in st.counters


def test_report_lists_dagger_corrections(runs):
    st = state.RunState.create("r9", "bee_in_tape_roll", "text")
    st.counters["dagger"] = dict(total=3, success=2, fail=1, unsure=0)
    st.evals["04_lora_v1_14999"] = dict(checkpoint="c", n=20, success=12, success_rate=0.6, met=False,
                                        dagger=[dict(episode=5, of=4, final="success"), dict(episode=7, of=6, final="fail")])
    md = state.report_md(st, _cfg())
    assert "| 04_lora_v1_14999 | `c` | 20 | 12 | 60% | no | 1/2 |" in md and "DAgger corrections: 3 (success 2" in md
    assert "dagger corrections 2/3 success" in "\n".join(state.status_lines(st, _cfg()))


def test_is_train_counts_only_steps_that_start_a_training():
    t = loop.Pipeline._is_train
    assert t("rlpj vla train --dataset d --exp-name e") and t("cd /r && uv run rlpj vla train --dataset d")
    assert t("cd /r && RLPJ_CONFIG=x RLPJ_RECIPE=/w/recipe.py setsid nohup bash scripts/train.sh /run bee_v2 lora_h30 --overwrite > log 2>&1 &")
    assert t("python -m rlpj.vla.openpi_entry train --run-dir r --dataset d")
    assert not t("cd /r && bash /w/programs/wait_train.sh lora_h30")                 # overnight 2026-11-06: counted 4 times
    assert not t("pkill -f '[o]penpi_entry train'; tail -n 5 logs/train_lora_h30.log")
    assert not t("cd /r && uv run rlpj vla convert --name bee_v2") and not t("rlpj eval r --checkpoint c")


def test_robot_time_limit_does_not_apply_to_gpu_jobs(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("t1", "bee_in_tape_roll", "text")
    cfg = _cfg(attempt_timeout_s=1200)
    p = loop.Pipeline(st, _op(), cfg)
    seen = []
    monkeypatch.setattr(attempts, "run_attempt", lambda cwd, command, attempt_dir, moves_robot, **kw: seen.append((moves_robot, kw["timeout_s"])))
    p._attempt(tmp_path, "true", tmp_path / "a1", True)
    p._attempt(tmp_path, "true", tmp_path / "a2", False)
    p._attempt(tmp_path, "true", tmp_path / "a3", False, timeout_s=7200)
    assert seen == [(True, 1200), (False, None), (False, 7200)]


def test_phase_file_redirects_run_all(fakes, runs, monkeypatch):
    st = state.RunState.create("t2", "bee_in_tape_roll", "text")
    st.phase = "collect"
    (st.dir / "PHASE").write_text("learn\n")
    p = loop.Pipeline(st, _op(), _cfg())
    went = []
    monkeypatch.setattr(loop.Pipeline, "learn", lambda self: went.append("learn") or "gave_up")
    monkeypatch.setattr(loop.Pipeline, "report", lambda self: None)
    p.run_all()
    assert went == ["learn"] and not (st.dir / "PHASE").exists()
    assert any(e["kind"] == "phase_override" and e["was"] == "collect" for e in state.read_jsonl(st.dir / "events.jsonl"))


def test_training_policy_refuses_lora_and_short_runs(runs):
    st = state.RunState.create("tp", "bee_in_tape_roll", "text")
    cfg = _cfg()
    cfg["training"].update(full_finetune_only=True, min_train_steps=30000)
    r = loop.Pipeline(st, _op(), cfg)._train_refusal
    assert r("rlpj vla train --dataset d --exp-name e --num-train-steps 40000") is None             # default config is full
    assert "LoRA" in r("rlpj vla train --dataset d --exp-name e --config pi05_kinova_lora --num-train-steps 40000")
    assert "LoRA" in r("RLPJ_CONFIG=pi05_kinova_lora_v4 setsid nohup bash scripts/train.sh /r d e --num-train-steps 40000")
    assert "at least 30000" in r("RLPJ_CONFIG=pi05_kinova_full bash scripts/train.sh /r d e --num-train-steps 12000")
    assert "at least 30000" in r("rlpj vla train --dataset d --exp-name e --config pi05_kinova_full")   # default 5000 steps
    assert r("rlpj vla train --dataset d --exp-name e --config pi05_kinova_full --num-train-steps 30000") is None


def test_train_host_local_runs_convert_and_train_here(fakes, runs, tmp_path, monkeypatch):
    st = state.RunState.create("th", "bee_in_tape_roll", "text")
    cfg = _cfg()
    cfg["hosts"]["train_host"] = "local"
    p = loop.Pipeline(st, _op(), cfg)
    p.workspace("learn")
    where = []
    monkeypatch.setattr(loop.Pipeline, "_ssh", lambda self, c, a, timeout_s=None, what="": where.append(("ssh", c)) or attempts.run_attempt(tmp_path, "true", a, False))
    monkeypatch.setattr(loop.Pipeline, "_attempt", lambda self, cwd, c, a, m, **kw: where.append(("local", c)) or attempts.run_attempt(tmp_path, "true", a, False))
    monkeypatch.setattr(loop.Pipeline, "_sync_episodes", lambda self: where.append(("sync", "")) or True)
    for cmd in ("rlpj vla convert --name d", "rlpj vla train --dataset d --exp-name e --num-train-steps 30000", "python programs/analyse.py"):
        p._learn_run(dict(action="run", command=cmd, what="x"), cmd)
    assert [w[0] for w in where] == ["local", "local", "ssh"]                # analysis still runs on the agent host
    assert "nice -n 10 rlpj vla train" in where[1][1]


def test_fetch_checkpoint_uses_a_pytorch_checkpoint_trained_here(runs):
    st = state.RunState.create("fp", "bee_in_tape_roll", "text")
    ck = st.dir / "checkpoints/full_v1/29999"
    (ck / "assets").mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"x")
    assert loop.Pipeline(st, _op(), _cfg())._fetch_checkpoint(ck) == ck               # no copy from the agent host
