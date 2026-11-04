"""Pipeline config (config/pipeline.yaml), the task it names (config/tasks/<task>.yaml) and the arm's robots.json entry."""
import functools
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def expand(p) -> Path:
    return Path(os.path.expanduser(str(p)))


@functools.lru_cache(None)
def load(task: str | None = None) -> dict:
    cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
    name = task or os.environ.get("RLPJ_TASK") or cfg["task"]
    cfg["task"] = yaml.safe_load((ROOT / "config/tasks" / f"{name}.yaml").read_text())
    return cfg


@functools.lru_cache(None)
def robot() -> dict:
    """robots.json entry of the arm in use (RLPJ_ROBOT_IP overrides pipeline.yaml)."""
    cfg = load()
    ip = os.environ.get("RLPJ_ROBOT_IP") or cfg["robot"]["ip"]
    entry = dict(json.loads(expand(cfg["robot"]["robots_json"]).read_text())[ip])
    entry.update(ip=ip, calibration_dir=str(expand(entry["calibration_dir"])))
    return entry
