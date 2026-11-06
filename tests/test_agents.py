"""Offline tests of rlpj.agents: schema, workspace, prompts, runner (through a fake claude binary), digest, sync.sh."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rlpj import config
from rlpj.agents import schema, workspace as wsmod
from rlpj.agents.runner import AgentError, AgentRunner, digest
from rlpj.agents.schema import ACTION_SCHEMA, Action, SchemaError, Usage, parse
from rlpj.agents.workspace import Workspace, render

FAKE_CLAUDE = r'''#!/usr/bin/env python3
"""Fake `claude -p`: records argv/stdin/cwd/env, prints a result JSON like the real one."""
import json, os, sys, time
prompt = sys.stdin.read()
argv = sys.argv[1:]
sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else argv[argv.index("--resume") + 1]
log = os.environ["FAKE_CLAUDE_LOG"]
calls = sum(1 for line in open(log) if json.loads(line)["sid"] == sid) + 1 if os.path.exists(log) else 1
with open(log, "a") as f:
    f.write(json.dumps({"argv": argv, "stdin": prompt, "cwd": os.getcwd(), "sid": sid,
                        "env": {k: v for k, v in os.environ.items() if k.startswith("CLAUDE_CODE")}}) + "\n")
time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "0")))
out = json.loads(os.environ.get("FAKE_CLAUDE_OUTPUT", "null"))
err = os.environ.get("FAKE_CLAUDE_ERROR") == "1"
# like the real CLI: usage/num_turns/duration per invocation, total_cost_usd/modelUsage cumulative over the session
result = {"type": "result", "subtype": "error_during_execution" if err else "success", "is_error": err,
          "duration_ms": 4321, "duration_api_ms": 4000, "num_turns": 3,
          "result": "budget exceeded" if err else "turn text", "session_id": sid, "total_cost_usd": 0.0421 * calls,
          "usage": {"input_tokens": 12, "cache_creation_input_tokens": 3400, "cache_read_input_tokens": 21000,
                    "output_tokens": 250, "server_tool_use": {"web_search_requests": 0}},
          "modelUsage": {"claude-opus-5-5": {"inputTokens": 12 * calls, "outputTokens": 250 * calls,
                                              "cacheReadInputTokens": 21000 * calls, "cacheCreationInputTokens": 3400 * calls,
                                              "costUSD": 0.0421 * calls, "contextWindow": 1000000}},
          "permission_denials": [], "uuid": "0b1c2d3e-0000-4000-8000-000000000000"}
if out is not None:
    result["structured_output"] = out
print(json.dumps(result))
sys.exit(1 if err else 0)
'''

RESULT = {"structured_output": {"action": "run", "command": "python programs/execute.py", "moves_robot": True,
                                "what": "dry run of perception", "report": ""},
          "session_id": "abc", "total_cost_usd": 1.25, "num_turns": 7, "duration_ms": 65000, "is_error": False,
          "usage": {"input_tokens": 10, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 30, "output_tokens": 40},
          "modelUsage": {"claude-opus-5-5": {"inputTokens": 10}}}


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture
def robot(tmp_path):
    calib = tmp_path / "calib"
    calib.mkdir()
    for f in ("side_pose_111.json", "handeye_kinova.json"):
        (calib / f).write_text(json.dumps({"file": f}))
    return {"name": "kinova_t", "ip": "192.168.2.9", "calibration_dir": str(calib), "side_camera_serial": "111",
            "files": ["side_pose_111.json", "handeye_kinova.json"], "table_z_m": -0.023, "side_depth_gain": 0.97,
            "side_depth_reads_long": "3%", "wrist_handeye": "handeye_kinova.json",
            "wrist_intrinsics": "intrinsics_kinova_1280x720.json", "wrist_intrinsics_use": "factory"}


@pytest.fixture
def task_cfg(tmp_path):
    demo = tmp_path / "demos/bee_in_tape_roll/episode_000"
    demo.mkdir(parents=True)
    (demo / "meta.json").write_text("{}")
    reuse = [tmp_path / "skills/battery_placement/kinova_b", tmp_path / "skills/marker_placement/kinova_a",
             tmp_path / "far_runs/battery_placement/baseline/seed_2/kinova_a"]
    for r in reuse:
        r.mkdir(parents=True)
        (r / "INDEX.md").write_text(f"# {r.name}\n")
    return {"name": "bee_in_tape_roll", "instruction": "put the foam bee in the tape roll",
            "goal": "Pick up the bee and put it inside the tape roll's hole.",
            "success": "The bee rests inside the tape roll's hole.",
            "reset": "Take the bee out, then place both. The tape roll is wider than the gripper opens (85 mm): lift it by "
                     "pinching its wall.",
            "objects": ["tape_roll", "foam_bee"], "one_demo": str(demo), "reuse": [str(r) for r in reuse],
            "layout": {"tape_roll": {"x": [0.40, 0.56], "y": [-0.20, 0.05]}, "foam_bee": {"x": [0.32, 0.56], "y": [-0.25, 0.10]},
                       "min_gap": 0.15, "radius": [0.30, 0.58], "tolerance": 0.025}}


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "runs/run_other/agents").mkdir(parents=True)
    d = tmp_path / "runs/run_test"
    (d / "episodes").mkdir(parents=True)
    (d / "state.json").write_text("{}")
    return d


@pytest.fixture
def ws(run_dir, task_cfg, robot):
    return Workspace.create(run_dir, "execute", task_cfg, "one_demo", robot=robot)


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    exe = tmp_path / "bin/claude"
    exe.parent.mkdir()
    exe.write_text(FAKE_CLAUDE)
    exe.chmod(0o755)
    log = tmp_path / "fake_claude.jsonl"
    monkeypatch.setenv("RLPJ_FAKE_CLAUDE", str(exe))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.delenv("FAKE_CLAUDE_OUTPUT", raising=False)
    monkeypatch.delenv("FAKE_CLAUDE_ERROR", raising=False)
    monkeypatch.delenv("FAKE_CLAUDE_SLEEP", raising=False)

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return calls


# ---------------------------------------------------------------------------------------------------- schema
def test_schema_is_strict_object():
    assert ACTION_SCHEMA["type"] == "object" and ACTION_SCHEMA["additionalProperties"] is False
    assert set(ACTION_SCHEMA["required"]) == {"action", "command", "moves_robot", "what", "report"}
    assert json.loads(schema.schema_json()) == ACTION_SCHEMA


def test_parse_run_from_result_json_text():
    a = parse(json.dumps(RESULT))
    assert a == Action("run", "dry run of perception", "python programs/execute.py", True, "")
    assert a.is_run and a.to_dict()["moves_robot"] is True


def test_parse_other_actions():
    a = parse({"action": "done", "command": "", "moves_robot": False, "what": "", "report": "works: attempts 3, 4"})
    assert a.action == "done" and a.report.startswith("works") and not a.is_run
    a = parse({"action": "need_operator", "command": "x", "moves_robot": True, "what": "scene wrong", "report": ""})
    assert a.report == "scene wrong" and a.command == "" and a.moves_robot is False     # report falls back to what
    a = parse({"action": "run", "command": "ls", "moves_robot": False, "what": "", "report": ""})
    assert a.what == "ls"


@pytest.mark.parametrize("bad", [
    {"action": "fly", "command": "", "moves_robot": False, "what": "", "report": ""},
    {"action": "run", "command": "", "moves_robot": False, "what": "x", "report": ""},
    {"action": "run", "command": "ls", "what": "x", "report": ""},
    {"action": "run", "command": "ls", "moves_robot": "yes", "what": "x", "report": ""},
    {"action": "give_up", "command": "", "moves_robot": False, "what": "", "report": ""},
    {"action": "done", "command": "", "moves_robot": False, "what": "", "report": "r", "extra": 1},
    {"is_error": False, "result": "no structured output"},
    [1, 2],
])
def test_parse_rejects(bad):
    with pytest.raises(SchemaError):
        parse(bad)


def test_usage_from_result():
    u = Usage.from_result(RESULT, model="fallback")
    assert (u.input_tokens, u.cache_creation_input_tokens, u.cache_read_input_tokens, u.output_tokens) == (10, 20, 30, 40)
    assert u.total_tokens == 100 and u.cost_usd == 1.25 and u.session_cost_usd == 1.25 and u.turns == 7 and u.duration_ms == 65000
    assert u.model == "claude-opus-5-5" and u.session_id == "abc" and u.is_error is False
    assert Usage.from_result({}, model="m").model == "m" and Usage.from_result({}).total_tokens == 0


# ---------------------------------------------------------------------------------------------------- workspace
def test_workspace_layout(ws, run_dir, task_cfg, robot):
    assert ws.path == (run_dir / "agents/execute").resolve()
    for d in ("programs", "notes", "inputs/calibration", "inputs/reuse", "attempts", "turns", ".claude"):
        assert (ws.path / d).is_dir(), d
    assert sorted(p.name for p in (ws.path / "inputs/calibration").iterdir()) == sorted(robot["files"])
    assert json.loads((ws.path / "inputs/calibration/handeye_kinova.json").read_text()) == {"file": "handeye_kinova.json"}
    demo = ws.path / "inputs/demo"
    assert demo.is_symlink() and demo.resolve() == Path(task_cfg["one_demo"]).resolve()
    links = {p.name: str(p.resolve()) for p in (ws.path / "inputs/reuse").iterdir()}
    assert links == {"battery_placement_kinova_b": task_cfg["reuse"][0], "marker_placement_kinova_a": task_cfg["reuse"][1],
                     "seed_2_kinova_a": task_cfg["reuse"][2]}
    assert all(Path(p).is_symlink() for p in (ws.path / "inputs/reuse").iterdir())
    assert (ws.path / "usage.jsonl").read_text() == ""
    s = ws.session
    assert s["role"] == "execute" and s["condition"] == "one_demo" and s["session_id"] is None and s["turns"] == 0
    assert s["add_dirs"] == [str(config.ROOT), *task_cfg["reuse"], task_cfg["one_demo"]]
    assert s["model"] == config.load()["agents"]["model"] and s["host"] == config.load()["hosts"]["agent"]
    assert ws.add_dirs == s["add_dirs"] and ws.role == "execute" and ws.condition == "one_demo"
    assert Workspace(ws.path).session == s


def test_workspace_git(ws):
    log = subprocess.run(["git", "-C", str(ws.path), "log", "--oneline"], capture_output=True, text=True).stdout
    assert len(log.splitlines()) == 1 and "workspace: execute agent" in log
    status = subprocess.run(["git", "-C", str(ws.path), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert status == ""                                   # attempts/turns/session/usage/links are ignored
    h0 = ws.head()
    assert ws.commit("nothing changed") == h0
    (ws.path / "programs/execute.py").write_text("print('hi')\n")
    h1 = ws.commit("attempt 1: first program")
    assert h1 != h0 and ws.head() == h1
    log = subprocess.run(["git", "-C", str(ws.path), "log", "--format=%s"], capture_output=True, text=True).stdout
    assert log.splitlines()[0] == "attempt 1: first program"


def test_workspace_settings(ws, run_dir, task_cfg):
    s = json.loads((ws.path / ".claude/settings.json").read_text())
    deny, allow = s["permissions"]["deny"], s["permissions"]["allow"]
    for b in ("Bash(*RLPJ_MOTION*)", "Bash(ssh *)", "Bash(scp *)", "Bash(rsync *)", "Bash(*192.168.2.*)"):
        assert b in deny
    root = str(config.ROOT).lstrip("/")
    assert f"Read(//{root}/**)" in allow and "Bash" in allow
    for p in task_cfg["reuse"] + [task_cfg["one_demo"]]:
        assert f"Read(//{p.lstrip('/')}/**)" in allow and f"Edit(//{p.lstrip('/')}/**)" in deny
    assert f"Edit(//{root}/rlpj/**)" in deny and f"Edit(//{root}/config/**)" in deny and f"Edit(//{root}/pyproject.toml)" in deny
    assert not any(r.startswith(f"Edit(//{root}/runs") for r in deny)          # the workspace lives under runs/
    other = str(run_dir.parent / "run_other").lstrip("/")
    assert f"Read(//{other}/**)" in deny and f"Edit(//{other}/**)" in deny
    me = str(ws.path).lstrip("/")
    for sib in ("episodes/**", "state.json", "datasets/**", "checkpoints/**", "evals/**", "calib/**", "agents/reset/**", "agents/learn/**"):
        assert f"Read(//{str(run_dir).lstrip('/')}/{sib})" in deny, sib
    assert not any(me in r for r in deny if r.startswith("Read("))              # this workspace stays readable
    for ro in ("inputs/**", ".claude/**", "attempts/**", "turns/**", "PROMPT.md", "session.json", "usage.jsonl"):
        assert f"Edit(//{me}/{ro})" in deny, ro
    assert not any(f"Edit(//{me}/programs" in r for r in deny)
    assert f"Read(//{str(Path.home()).lstrip('/')}/.claude/projects/**)" in deny
    assert s["claudeMdExcludes"][0] == str(ws.path.parent / "CLAUDE.md") and s["claudeMdExcludes"][-1] == "/CLAUDE.md"


def test_workspace_attempts_and_usage(ws):
    assert ws.attempts() == []
    a1, a2 = ws.next_attempt_dir(), ws.next_attempt_dir()
    assert (a1.name, a2.name) == ("attempt_1", "attempt_2") and a2.is_dir()
    (ws.path / "attempts/attempt_10").mkdir()
    assert ws.next_attempt_dir().name == "attempt_11" and [p.name for p in ws.attempts()][-2:] == ["attempt_10", "attempt_11"]
    ws.record_usage(Usage.from_result(RESULT), turn=1, kind="start")
    ws.record_usage(Usage(output_tokens=5, cost_usd=0.5, turns=2), turn=2, kind="resume")
    recs = [json.loads(line) for line in (ws.path / "usage.jsonl").read_text().splitlines()]
    assert [r["turn"] for r in recs] == [1, 2] and recs[0]["cost_usd"] == 1.25 and recs[0]["role"] == "execute"
    tot = ws.usage_total()
    assert tot["cost_usd"] == 1.75 and tot["output_tokens"] == 45 and tot["turns"] == 9 and tot["calls"] == 2


def test_workspace_sync_args(ws, monkeypatch):
    monkeypatch.setenv("RLPJ_AGENT_HOST", "agenthost")
    p = str(ws.path)
    full = ws.push_args(full=True)
    assert full[-2:] == [f"{p}/", f"agenthost:{p}/"] and "--delete" in full and "--mkpath" in full
    part = ws.push_args(full=False)
    assert part[-2:] == full[-2:] and part[part.index("--exclude") + 1] == "*"
    inc = [part[i + 1] for i, a in enumerate(part) if a == "--include"]
    assert inc == ["attempts/***", "turns/***", "session.json", "usage.jsonl", ".git/***"]
    pull = ws.pull_args()
    assert pull[-2:] == [f"agenthost:{p}/", f"{p}/"] and "--delete" in pull
    assert [pull[i + 1] for i, a in enumerate(pull) if a == "--exclude"] == ["attempts/", "turns/", "session.json", "usage.jsonl"]


def test_workspace_text_condition_and_errors(run_dir, task_cfg, robot):
    w = Workspace.create(run_dir, "execute", task_cfg, "text", robot=robot)
    assert not (w.path / "inputs/demo").exists() and w.session["demo"] is None
    assert w.session["add_dirs"] == [str(config.ROOT), *task_cfg["reuse"]]
    with pytest.raises(FileExistsError):
        Workspace.create(run_dir, "execute", task_cfg, "text", robot=robot)
    with pytest.raises(ValueError):
        Workspace.create(run_dir, "execute", task_cfg, None, robot=robot)
    with pytest.raises(ValueError):
        Workspace.create(run_dir, "driver", task_cfg, "text", robot=robot)
    r = Workspace.create(run_dir, "reset", task_cfg, "one_demo", robot=robot)   # condition is only for execute
    assert r.condition is None and not (r.path / "inputs/demo").exists()
    bad = dict(task_cfg, one_demo=str(run_dir / "missing"))
    with pytest.raises(FileNotFoundError):
        Workspace.create(run_dir.parent / "run_x", "execute", bad, "one_demo", robot=robot)


def test_link_names_unique():
    names = wsmod.link_names(["/a/far_skills/x/kinova_a", "/a/far_runs/x/kinova_a", "/b/y/kinova_b"])
    assert names == {"/a/far_skills/x/kinova_a": "far_skills_x_kinova_a", "/a/far_runs/x/kinova_a": "far_runs_x_kinova_a",
                     "/b/y/kinova_b": "y_kinova_b"}


# ---------------------------------------------------------------------------------------------------- prompts
def test_render_execute_one_demo(task_cfg, robot, ws):
    text = (ws.path / "PROMPT.md").read_text()
    assert text == render("execute", task_cfg, "one_demo", robot=robot, workspace=ws.path)
    assert task_cfg["goal"] in text and task_cfg["success"] in text
    assert "## Task specification: one human demonstration" in text and task_cfg["one_demo"] in text
    assert "_depth.mkv" in text and "side_left" in text and "front" in text and "never by colour" in text
    assert "APPLY" in text and "ADAPT" in text and "REJECT" in text
    assert "`inputs/reuse/battery_placement_kinova_b` -> `" + task_cfg["reuse"][0] in text
    assert "RobotSession" in text and "s.claim(" in text and "RLPJ_MOTION" in text
    assert "`tape_roll`, `foam_bee`" in text and "x [0.32, 0.56] m, y [-0.25, 0.10] m" in text
    assert robot["ip"] in text and robot["side_camera_serial"] in text and str(robot["table_z_m"]) in text
    assert str(ws.path) in text and str(config.ROOT / ".venv/bin/python") in text
    assert "{" + "TASK_GOAL}" not in text and "{CONDITION_BLOCK}" not in text and "{REUSE_BLOCK}" not in text
    assert "exit code 130" in text and "structured action" in text


def test_render_execute_text(task_cfg, robot):
    text = render("execute", task_cfg, "text", robot=robot)
    assert "## Task specification: text only" in text and "inputs/demo" not in text and "_depth.mkv" not in text
    assert task_cfg["goal"] in text
    no_reuse = render("execute", dict(task_cfg, reuse=[]), "text", robot=robot)
    assert "APPLY" not in no_reuse and "inputs/reuse" not in no_reuse


def test_render_reset_and_learn(task_cfg, robot):
    text = render("reset", task_cfg, robot=robot)
    assert "programs/reset.py --targets" in text and "pinching its wall" in text and "0.025" in text
    assert '"tape_roll": [0.48' in text and "Undo" in text or "undo" in text
    assert "## Task specification" not in text
    text = render("learn", task_cfg, robot=robot, DATASET="datasets/bee_v1", N_EPISODES="57", HISTORY="  - run_1: loss 0.1")
    assert "rlpj vla convert" in text and "rlpj vla train" in text and "rlpj vla serve" in text and "rlpj eval" in text
    assert "datasets/bee_v1" in text and "57" in text and "run_1: loss 0.1" in text
    assert task_cfg["instruction"] in text and "programs/execute.py" not in text.split("## Your job")[1]
    default = render("learn", task_cfg, robot=robot)
    assert "no training runs yet" in default
    with pytest.raises(ValueError):
        render("execute", task_cfg, None, robot=robot)


# ---------------------------------------------------------------------------------------------------- runner
def test_runner_start_and_resume(ws, fake_claude, monkeypatch):
    runner = AgentRunner(ws, model="claude-opus-5-5", effort="low", max_budget_usd=7.5, timeout_s=30)
    monkeypatch.setenv("FAKE_CLAUDE_OUTPUT", json.dumps(
        {"action": "run", "command": "python programs/execute.py", "moves_robot": False, "what": "dry run", "report": ""}))
    action, usage = runner.start()
    assert action == Action("run", "dry run", "python programs/execute.py", False, "")
    assert usage.cost_usd == 0.0421 and usage.cache_read_input_tokens == 21000 and usage.turns == 3
    assert usage.model == "claude-opus-5-5" and ws.session["session_cost_usd"] == 0.0421
    sid = ws.session["session_id"]
    assert sid and len(sid) == 36 and ws.session["turns"] == 1 and usage.session_id == sid
    call = fake_claude()[0]
    argv = call["argv"]
    assert call["stdin"] == ws.prompt() and call["cwd"] == str(ws.path)
    assert call["env"] == {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    assert argv[argv.index("--session-id") + 1] == sid and "--resume" not in argv
    assert "-p" in argv and argv[argv.index("--model") + 1] == "claude-opus-5-5" and argv[argv.index("--effort") + 1] == "low"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == ACTION_SCHEMA
    for flag, val in (("--output-format", "json"), ("--permission-mode", "acceptEdits"), ("--permission-prompts", "none"),
                      ("--setting-sources", "project,local"), ("--max-budget-usd", "7.5")):
        assert argv[argv.index(flag) + 1] == val, flag
    assert "--strict-mcp-config" in argv and "--disable-slash-commands" in argv
    i = argv.index("--add-dir")
    assert argv[i + 1:i + 1 + len(ws.add_dirs)] == ws.add_dirs and argv[i + 1 + len(ws.add_dirs)] == "-p"
    assert "--no-session-persistence" not in argv and "--tools" not in argv
    # resume with a digest
    monkeypatch.setenv("FAKE_CLAUDE_OUTPUT", json.dumps(
        {"action": "done", "command": "", "moves_robot": False, "what": "", "report": "success on attempt 2"}))
    action, usage = runner.resume("digest text")
    assert action.action == "done" and action.report == "success on attempt 2"
    assert usage.session_cost_usd == pytest.approx(0.0842) and usage.cost_usd == pytest.approx(0.0421)   # this turn's share
    call = fake_claude()[1]
    assert call["stdin"] == "digest text" and call["argv"][call["argv"].index("--resume") + 1] == sid
    assert "--session-id" not in call["argv"]
    assert ws.session["turns"] == 2 and Workspace(ws.path).session["session_id"] == sid
    recs = [json.loads(line) for line in (ws.path / "usage.jsonl").read_text().splitlines()]
    assert [(r["turn"], r["kind"]) for r in recs] == [(1, "start"), (2, "resume")]
    assert recs[1]["cost_usd"] == pytest.approx(0.0421) and ws.usage_total()["cost_usd"] == pytest.approx(0.0842)
    turns = sorted(p.name for p in (ws.path / "turns").iterdir())
    assert turns == ["turn_001.json", "turn_002.json"]
    raw = json.loads((ws.path / "turns/turn_002.json").read_text())
    assert raw["kind"] == "resume" and raw["prompt"] == "digest text" and raw["result"]["structured_output"]["action"] == "done"
    with pytest.raises(AgentError):
        runner.start()                                      # a started session is only resumed


def test_runner_errors(ws, fake_claude, monkeypatch):
    runner = AgentRunner(ws, timeout_s=30)
    with pytest.raises(AgentError):
        runner.resume("nothing to resume")
    monkeypatch.setenv("FAKE_CLAUDE_ERROR", "1")
    monkeypatch.setenv("FAKE_CLAUDE_OUTPUT", "null")
    with pytest.raises(AgentError, match="budget exceeded"):
        runner.start("hello")
    recs = (ws.path / "usage.jsonl").read_text().splitlines()
    assert len(recs) == 1 and json.loads(recs[0])["is_error"] is True   # cost is recorded even for failed turns
    monkeypatch.delenv("FAKE_CLAUDE_ERROR")
    monkeypatch.setenv("FAKE_CLAUDE_OUTPUT", json.dumps({"action": "run", "command": "", "moves_robot": False, "what": "", "report": ""}))
    with pytest.raises(AgentError, match="invalid action"):
        runner.resume("again")
    monkeypatch.setenv("FAKE_CLAUDE_OUTPUT", "null")
    with pytest.raises(AgentError, match="invalid action"):
        runner.resume("no structured output")
    assert fake_claude()[0]["stdin"] == "hello"


def test_runner_timeout_and_missing_binary(ws, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "5")
    runner = AgentRunner(ws, timeout_s=0.5)
    with pytest.raises(AgentError, match="timed out"):
        runner.start("slow")
    assert ws.session["session_id"] is None                  # a first turn that never came back leaves no session
    monkeypatch.setenv("RLPJ_FAKE_CLAUDE", str(ws.path / "no_such_binary"))
    with pytest.raises(AgentError, match="cannot run"):
        AgentRunner(ws, timeout_s=5).start("x")
    with pytest.raises(AgentError, match="no session"):
        AgentRunner(ws, timeout_s=5).resume("x")


def test_runner_ssh_command(ws, monkeypatch):
    monkeypatch.delenv("RLPJ_FAKE_CLAUDE", raising=False)
    runner = AgentRunner(ws, model="claude-opus-5-5", effort="medium", max_budget_usd=0, claude_bin="claude")
    argv = runner.command("11111111-2222-4333-8444-555555555555")
    assert argv[0] == "claude" and "--max-budget-usd" not in argv
    sh = runner.shell_command(argv)
    assert sh.startswith(f"cd {ws.path} && export PATH=$HOME/.local/bin:$PATH CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 "
                         "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 && exec claude --add-dir ")
    assert "--session-id 11111111-2222-4333-8444-555555555555" in sh
    assert "'" in sh and json.loads(sh.split("--json-schema ")[1].split(" --session-id")[0].strip("'")) == ACTION_SCHEMA
    monkeypatch.setenv("RLPJ_CLAUDE_BIN", "/opt/claude")
    assert AgentRunner(ws).command("x")[0] == "/opt/claude"


# ---------------------------------------------------------------------------------------------------- digest
def test_digest(tmp_path):
    a = tmp_path / "attempt_3"
    (a / "plans").mkdir(parents=True)
    (a / "attempt.json").write_text(json.dumps({"exit_code": 130, "command": "python programs/execute.py", "commit": "abc1234",
                                                "outcome": {"claim": "success", "operator": "stop", "final": "fail"}}))
    (a / "summary.json").write_text("{\n" + "\n".join(f' "k{i}": {i},' for i in range(100)) + "\n}\n")
    (a / "log.txt").write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    (a / "after_side.jpg").write_bytes(b"x" * 3000)
    (a / "plans/segment_1.png").write_bytes(b"y" * 1024)
    (a / "plans/replay.mp4").write_bytes(b"z" * 2048)
    d = digest(a)
    assert d.startswith("# attempt_3: exit code 130; outcome: claim=success, operator=stop, final=fail")
    assert "commit: abc1234" in d and "command: python programs/execute.py" in d
    assert "The operator stopped this attempt: it FAILED" in d
    assert '"k58": 58,' in d and '"k59": 59,' not in d and "(42 more lines)" in d      # 60 lines: "{" + k0..k58
    assert "line 60" in d and "line 59" not in d and "log.txt: 100 lines; last 40" in d
    assert "after_side.jpg  2 KB" in d and "plans/segment_1.png  1 KB" in d and "plans/replay.mp4 (2 KB)" in d
    assert "exit code 130) counts as a failure" in d and "structured action" in d
    # fallbacks: exit_code file, no summary, outcome derived from the code
    b = tmp_path / "attempt_4"
    b.mkdir()
    (b / "exit_code").write_text("2\n")
    d = digest(b)
    assert d.startswith("# attempt_4: exit code 2; outcome: fail") and "summary.json: MISSING" in d and "log.txt: MISSING" in d
    assert "The operator stopped" not in d
    assert digest(b, exit_code=0, outcome="success").startswith("# attempt_4: exit code 0; outcome: success")
    assert "exit code ?" in digest(tmp_path / "attempt_5") if not (tmp_path / "attempt_5").exists() else True


# ---------------------------------------------------------------------------------------------------- sync.sh
def test_sync_sh_usage():
    script = config.ROOT / "scripts/sync.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert r.returncode == 1 and "usage" in r.stdout
    r = subprocess.run(["bash", str(script), "sideways"], capture_output=True, text=True)
    assert r.returncode == 1 and "usage" in r.stdout
    r = subprocess.run(["bash", str(script), "push", "/tmp/outside"], capture_output=True, text=True)
    assert r.returncode == 1 and "outside the repo" in r.stdout
    text = script.read_text()
    for ex in ("/.venv", "/runs", "/third_party", "/.git"):
        assert f"--exclude {ex}" in text


def test_trust_snippet_marks_only_the_workspace(tmp_path):
    import os
    import subprocess
    import sys
    from rlpj.agents.workspace import TRUST
    cj = tmp_path / ".claude.json"
    cj.write_text(json.dumps({"projects": {"/a": {"x": 1}}, "other": 2}))
    cj.chmod(0o600)
    for _ in range(2):
        subprocess.run([sys.executable, "-c", TRUST, "/ws/path"], env={**os.environ, "HOME": str(tmp_path)}, check=True)
    d = json.loads(cj.read_text())
    assert d["projects"]["/ws/path"] == {"hasTrustDialogAccepted": True} and d["projects"]["/a"] == {"x": 1} and d["other"] == 2
    assert oct(cj.stat().st_mode & 0o777) == "0o600" and not (tmp_path / ".claude.json.rlpj.tmp").exists()
