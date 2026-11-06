"""Per-agent workspace runs/<run>/agents/<role>/: git repo, inputs, rendered prompt, Claude Code permissions, rsync."""
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from rlpj import config

ROLES = ("execute", "reset", "learn")
CONDITIONS = ("text", "one_demo")
PROMPTS = Path(__file__).parent / "prompts"
PROGRAM = {"execute": "programs/execute.py", "reset": "programs/reset.py", "learn": ""}
# Written on the robot machine by the runner/orchestrator; everything else is the agent's (edited on the agent machine).
ORCH_OWNED = ("attempts/", "turns/", "session.json", "usage.jsonl")
RUN_SIBLINGS = ("state.json", "episodes", "datasets", "checkpoints", "evals", "calib")   # DESIGN.md section 4
BASH_DENY = ("*RLPJ_MOTION*", "ssh *", "scp *", "rsync *", "*192.168.2.*", "sudo *")
GITIGNORE = "attempts/\nturns/\nsession.json\nusage.jsonl\ninputs/demo\ninputs/reuse/\n__pycache__/\n*.pyc\n"
GIT_ID = ["-c", "user.name=rlpj", "-c", "user.email=rlpj@local"]
# Claude Code applies a folder's .claude/settings.json allow rules only once the folder is trusted:
# projects[<path>].hasTrustDialogAccepted in ~/.claude.json of the agent machine (deny rules apply either way).
TRUST = """import json, os, sys
p = os.path.expanduser("~/.claude.json")
d = json.load(open(p)) if os.path.exists(p) else {}
e = d.setdefault("projects", {}).setdefault(sys.argv[1], {})
if not e.get("hasTrustDialogAccepted"):
    e["hasTrustDialogAccepted"] = True
    t = p + ".rlpj.tmp"
    with open(t, "w") as f:
        json.dump(d, f, indent=2)
    if os.path.exists(p):
        os.chmod(t, os.stat(p).st_mode & 0o777)
    os.replace(t, p)
"""
LEARN_DEFAULTS = dict(N_EPISODES="0", EPISODES_NOTE="none collected yet", EPISODES_DIR="<run>/episodes", DATASET="none yet",
                      HISTORY="  (no training runs yet)")


class SyncError(RuntimeError):
    """rsync or ssh to the agent machine failed."""


def fill(text: str, **kw) -> str:
    """Replace {KEY} placeholders (plain replace, so braces in code samples survive)."""
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def _ranges(layout: dict, objects) -> tuple[str, str]:
    xs = [layout[o]["x"] for o in objects if o in layout]
    ys = [layout[o]["y"] for o in objects if o in layout]
    rng = lambda vs: f"[{min(v[0] for v in vs):.2f}, {max(v[1] for v in vs):.2f}]" if vs else "n/a"
    return rng(xs), rng(ys)


def _example_targets(layout: dict, objects) -> str:
    """Valid-looking example targets: x mid-range, y spread over the objects so they keep apart."""
    objs = [o for o in objects if o in layout]
    out = {}
    for i, o in enumerate(objs):
        f = 0.5 if len(objs) == 1 else 0.15 + 0.7 * i / (len(objs) - 1)
        (x0, x1), (y0, y1) = layout[o]["x"], layout[o]["y"]
        out[o] = [round((x0 + x1) / 2, 2), round(y0 + f * (y1 - y0), 2)]
    return json.dumps(out)


def link_names(paths) -> dict:
    """Short unique link names for reuse paths: the last two components, extended until unique."""
    paths = [Path(p) for p in paths]
    out = {}
    for p in paths:
        n = 2
        name = "_".join(p.parts[-n:])
        while name in out.values() or sum(1 for q in paths if "_".join(q.parts[-n:]) == name) > 1:
            n += 1
            if n > len(p.parts):
                break
            name = "_".join(p.parts[-n:])
        out[str(p)] = name
    return out


def render(role: str, task_cfg: dict, condition: str | None = None, robot: dict | None = None,
           workspace: str | Path | None = None, cfg: dict | None = None, **extra) -> str:
    """First prompt of an agent: prompts/common.md + prompts/<role>.md with the placeholders filled."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if role == "execute" and condition not in CONDITIONS:
        raise ValueError(f"execute needs condition in {CONDITIONS}, got {condition!r}")
    cfg = cfg or config.load()
    robot = robot or config.robot()
    ws = str(workspace) if workspace else f"{config.RUNS}/<run>/agents/{role}"
    objects = list(task_cfg.get("objects", []))
    layout = task_cfg.get("layout", {})
    lx, ly = _ranges(layout, objects)
    demo = config.expand(task_cfg["one_demo"]) if task_cfg.get("one_demo") else ""
    reuse = link_names([config.expand(p) for p in task_cfg.get("reuse", [])])        # {path: link name}
    v = dict(
        ROLE=role, ROLE_TITLE=role.capitalize(), TASK_NAME=task_cfg.get("name", "?"), ROOT=config.ROOT, WORKSPACE=ws,
        PYTHON=config.ROOT / ".venv/bin/python", PROGRAM=PROGRAM[role],
        AGENT_HOST=cfg["hosts"]["agent"], POLICY_PORT=cfg["hosts"].get("policy_port", 8000),
        ROBOT_NAME=robot.get("name", "kinova"), ROBOT_IP=robot.get("ip", "?"), SIDE_SERIAL=robot.get("side_camera_serial", "?"),
        TABLE_Z=robot.get("table_z_m", "?"), WRIST_HANDEYE=robot.get("wrist_handeye", "handeye_kinova.json"),
        WRIST_INTRINSICS=robot.get("wrist_intrinsics", "intrinsics_kinova_1280x720.json"),
        WRIST_INTRINSICS_USE=robot.get("wrist_intrinsics_use", "factory"), SIDE_DEPTH_LONG=robot.get("side_depth_reads_long", "3%"),
        TASK_GOAL=task_cfg.get("goal", ""), TASK_SUCCESS=task_cfg.get("success", ""), TASK_RESET=task_cfg.get("reset", ""),
        TASK_INSTRUCTION=task_cfg.get("instruction", ""), OBJECTS=", ".join(f"`{o}`" for o in objects) or "(see goal)",
        LAYOUT=json.dumps({o: layout[o] for o in objects if o in layout}), LAYOUT_X=lx, LAYOUT_Y=ly,
        RADIUS=layout.get("radius", "?"), MIN_GAP=layout.get("min_gap", "?"), TOLERANCE=layout.get("tolerance", "?"),
        EXAMPLE_TARGETS=_example_targets(layout, objects), DEMO=demo,
        TARGET_SUCCESS=cfg.get("evaluation", {}).get("target_success_rate", "?"),
        EVAL_EPISODES=cfg.get("evaluation", {}).get("episodes", "?"), MAX_RUNS=cfg.get("training", {}).get("max_runs", "?"),
        DAGGER="on" if cfg.get("evaluation", {}).get("dagger") else "off",
    )
    if role == "learn":
        v.update(LEARN_DEFAULTS)
    v.update(extra)
    if role == "execute":
        v["CONDITION_BLOCK"] = fill((PROMPTS / f"condition_{condition}.md").read_text().strip(), **v)
    lines = [f"- `inputs/reuse/{n}` -> `{p}`" for p, n in reuse.items()]
    v["REUSE_BLOCK"] = fill((PROMPTS / "reuse.md").read_text(), REUSE_LIST="\n".join(lines), **v).strip() if lines else ""
    text = fill((PROMPTS / "common.md").read_text().strip() + "\n\n" + (PROMPTS / f"{role}.md").read_text().strip(), **v)
    text = re.sub(r"\n{3,}", "\n\n", text) + "\n"
    left = sorted(set(re.findall(r"\{[A-Z][A-Z_]+\}", text)))
    if left:
        raise ValueError(f"unfilled placeholders in the {role} prompt: {left}")
    return text


def _pat(p: Path) -> str:
    """Claude Code absolute path pattern: //dir/** for folders, //file for files (FAR convention)."""
    is_file = p.is_file() or (not p.is_dir() and p.suffix != "")
    return f"//{str(p).lstrip('/')}" if is_file else f"//{str(p).lstrip('/')}/**"


def settings(ws: Path, run_dir: Path, role: str, allowed: list) -> dict:
    """Claude Code project settings: no robot/remote access, repo read-only, other runs/agents hidden, inputs read-only."""
    root = config.ROOT
    read_only = [p for p in sorted(root.iterdir()) if p.name not in ("runs", ".venv")]
    hidden = [p for p in sorted(run_dir.parent.iterdir()) if p != run_dir]              # other runs
    hidden += [run_dir / n for n in dict.fromkeys(RUN_SIBLINGS + tuple(p.name for p in run_dir.iterdir()
                                                                              if p.name != "agents"))]
    hidden += [run_dir / "agents" / r for r in ROLES if r != role]                        # the other agents
    hidden.append(Path.home() / ".claude/projects")                                        # earlier sessions
    read_only += [ws / x for x in ("inputs", ".claude", "attempts", "turns", "PROMPT.md", "session.json", "usage.jsonl")]
    read_only += [Path(p) for p in allowed]
    deny = [f"Bash({b})" for b in BASH_DENY] + [f"Read({_pat(p)})" for p in hidden] + \
           [f"Edit({_pat(p)})" for p in hidden + read_only]
    allow = ["Bash", f"Read({_pat(root)})"] + [f"Read({_pat(Path(p))})" for p in allowed]
    if role == "learn":
        allow += ["WebSearch", "WebFetch"]          # research on fine-tuning practice (pi0.5, openpi, RTC, ...)
    excludes = [str(p / "CLAUDE.md") for p in ws.parents]
    return {"permissions": {"allow": list(dict.fromkeys(allow)), "deny": list(dict.fromkeys(deny))},
            "claudeMdExcludes": excludes}


class Workspace:
    """One agent's directory. `Workspace(path)` opens an existing one, `Workspace.create(...)` makes a new one."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.session = json.loads((self.path / "session.json").read_text())

    @classmethod
    def create(cls, run_dir, role: str, task_cfg: dict, condition: str | None = None, robot: dict | None = None,
               cfg: dict | None = None, **extra) -> "Workspace":
        """Make runs/<run>/agents/<role>/ with inputs, PROMPT.md, .claude/settings.json, session.json and a first commit."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if role == "execute" and condition not in CONDITIONS:
            raise ValueError(f"execute needs condition in {CONDITIONS}, got {condition!r}")
        condition = condition if role == "execute" else None
        cfg = cfg or config.load()
        robot = robot or config.robot()
        run_dir = Path(run_dir).resolve()
        ws = run_dir / "agents" / role
        if (ws / "session.json").exists():
            raise FileExistsError(f"{ws} already exists")
        for d in ("programs", "notes", "inputs/calibration", "inputs/reuse", "attempts", "turns", ".claude"):
            (ws / d).mkdir(parents=True, exist_ok=True)
        calib = Path(robot["calibration_dir"])
        for f in robot.get("files", []):
            shutil.copy2(calib / f, ws / "inputs/calibration" / f)
        demo = None
        if condition == "one_demo":
            demo = config.expand(task_cfg["one_demo"])
            if not demo.is_dir():
                raise FileNotFoundError(f"demo not found: {demo}")
            (ws / "inputs/demo").symlink_to(demo)
        reuse = {}
        for target, name in link_names([config.expand(p) for p in task_cfg.get("reuse", [])]).items():
            if not Path(target).exists():
                raise FileNotFoundError(f"reuse path not found: {target}")
            (ws / "inputs/reuse" / name).symlink_to(target)
            reuse[name] = target
        allowed = list(reuse.values()) + ([str(demo)] if demo else [])
        (ws / "PROMPT.md").write_text(render(role, task_cfg, condition, robot=robot, workspace=ws, cfg=cfg, **extra))
        (ws / ".claude/settings.json").write_text(json.dumps(settings(ws, run_dir, role, allowed), indent=1))
        (ws / ".gitignore").write_text(GITIGNORE)
        (ws / "usage.jsonl").touch()
        session = dict(role=role, condition=condition, task=task_cfg.get("name"), run_dir=str(run_dir), path=str(ws),
                       host=cfg["hosts"]["agent"], robot=robot.get("name"), model=cfg["agents"]["model"],
                       effort=cfg["agents"]["effort"], created=time.strftime("%Y-%m-%d %H:%M:%S"), session_id=None,
                       turns=0, demo=str(demo) if demo else None, reuse=reuse, add_dirs=[str(config.ROOT), *allowed])
        (ws / "session.json").write_text(json.dumps(session, indent=1))
        _git(ws, "init", "-q")
        _git(ws, "config", "user.name", "rlpj")
        _git(ws, "config", "user.email", "rlpj@local")
        _git(ws, "add", "-A")
        _git(ws, "commit", "-qm", f"workspace: {role} agent, task {task_cfg.get('name')}, condition {condition}")
        return cls(ws)

    @property
    def role(self) -> str:
        return self.session["role"]

    @property
    def condition(self):
        return self.session.get("condition")

    @property
    def host(self) -> str:
        return os.environ.get("RLPJ_AGENT_HOST") or self.session["host"]

    @property
    def add_dirs(self) -> list:
        return list(self.session.get("add_dirs") or [str(config.ROOT)])

    def prompt(self) -> str:
        return (self.path / "PROMPT.md").read_text()

    def save_session(self):
        (self.path / "session.json").write_text(json.dumps(self.session, indent=1))

    def attempts(self) -> list:
        """Existing attempt dirs in numeric order."""
        ds = [p for p in (self.path / "attempts").glob("attempt_*") if p.is_dir() and p.name[8:].isdigit()]
        return sorted(ds, key=lambda p: int(p.name[8:]))

    def next_attempt_dir(self) -> Path:
        """Create and return attempts/attempt_<N> with the next free N."""
        n = max([int(p.name[8:]) for p in self.attempts()] + [0]) + 1
        d = self.path / "attempts" / f"attempt_{n}"
        d.mkdir(parents=True)
        return d

    def commit(self, msg: str) -> str:
        """Commit everything that changed (if anything); returns the short HEAD hash."""
        _git(self.path, "add", "-A")
        if _git(self.path, "status", "--porcelain"):
            _git(self.path, "commit", "-qm", msg)
        return self.head()

    def head(self) -> str:
        return _git(self.path, "rev-parse", "--short", "HEAD", check=False) or "none"

    def record_usage(self, usage, **extra):
        """Append one turn's usage to usage.jsonl."""
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "role": self.role, **usage.to_dict(), **extra}
        with open(self.path / "usage.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def usage_total(self) -> dict:
        """Sum of cost, tokens and turns over usage.jsonl."""
        tot = dict(cost_usd=0.0, input_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                   output_tokens=0, turns=0, duration_ms=0, calls=0)
        for line in (self.path / "usage.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                for k in tot:
                    tot[k] += r.get(k, 0) if k != "calls" else 1
        return tot

    # ---- sync with the agent machine (same path there)
    def remote(self) -> str:
        return f"{self.host}:{self.path}/"

    def push_args(self, full: bool = False) -> list:
        """rsync argv: robot machine -> agent machine; our files (attempts, turns, session, usage) and .git (so commits
        made here after a pull reach the agent), or everything when full."""
        if full:
            return ["rsync", "-a", "--delete", "--mkpath", f"{self.path}/", self.remote()]
        inc = [a for p in ORCH_OWNED + (".git/",) for a in ("--include", p + "***" if p.endswith("/") else p)]
        return ["rsync", "-a", "--delete", "--mkpath", *inc, "--exclude", "*", f"{self.path}/", self.remote()]

    def pull_args(self) -> list:
        """rsync argv: agent machine -> robot machine; the agent's files (everything but what we write here)."""
        ex = [a for p in ORCH_OWNED for a in ("--exclude", p)]
        return ["rsync", "-a", "--delete", *ex, self.remote(), f"{self.path}/"]

    def push(self, full: bool | None = None):
        """Sync to the agent machine: everything on the first push (and the demo recording), later only our files."""
        if full is None:
            full = not self.remote_exists()
        if full and self.session.get("demo"):
            demo = self.session["demo"]
            _run(["rsync", "-a", "--mkpath", f"{demo}/", f"{self.host}:{demo}/"])
        _run(self.push_args(full))

    def pull(self):
        """Sync the agent's files from the agent machine."""
        _run(self.pull_args())

    def trust(self):
        """Mark this workspace trusted for Claude Code on the agent machine, so its permission rules apply."""
        _run(["ssh", "-o", "BatchMode=yes", self.host, f"python3 -c {shlex.quote(TRUST)} {_q(self.path)}"])

    def remote_exists(self) -> bool:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, f"test -f {_q(self.path / 'PROMPT.md')}"],
                           capture_output=True, text=True)
        if r.returncode not in (0, 1):
            raise SyncError(f"ssh {self.host} failed ({r.returncode}): {r.stderr.strip()[-500:]}")
        return r.returncode == 0


def _q(p) -> str:
    return shlex.quote(str(p))


def _run(argv):
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode:
        raise SyncError(f"{' '.join(argv[:3])}... failed ({r.returncode}): {r.stderr.strip()[-800:]}")
    return r.stdout


def _git(d, *args, check=True) -> str:
    r = subprocess.run(["git", "-C", str(d), *GIT_ID, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()
