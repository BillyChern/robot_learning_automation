"""The structured action that ends every agent turn, and the usage record of a turn (DESIGN.md section 7)."""
import json
from dataclasses import asdict, dataclass

ACTIONS = ("run", "done", "need_operator", "give_up")

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string", "enum": list(ACTIONS),
            "description": "run: the orchestrator executes `command` and resumes you with its digest; done: the job is "
                           "finished; need_operator: a person must do something first (say what in `report`); "
                           "give_up: no further progress is possible (say why in `report`)"},
        "command": {
            "type": "string",
            "description": "run only: the shell command, executed from the workspace root on the machine it belongs to "
                           "(robot programs: `<python> programs/<file>.py ...`); empty string otherwise"},
        "moves_robot": {
            "type": "boolean",
            "description": "run only: true if the command may move the arm or the gripper (the operator is asked first); "
                           "false for dry runs and offline commands, which run with motion disabled"},
        "what": {
            "type": "string",
            "description": "one line: what this run should do (run), or a one-line status (other actions)"},
        "report": {
            "type": "string",
            "description": "done, give_up, need_operator: short text for the operator; empty string for run"},
    },
    "required": ["action", "command", "moves_robot", "what", "report"],
    "additionalProperties": False,
}


def schema_json() -> str:
    """ACTION_SCHEMA as the one-line string passed to `claude --json-schema`."""
    return json.dumps(ACTION_SCHEMA, separators=(",", ":"))


class SchemaError(ValueError):
    """The agent's structured output is not a valid action."""


@dataclass
class Action:
    """One agent decision; `command`/`moves_robot` matter for run, `report` for the other actions."""
    action: str
    what: str = ""
    command: str = ""
    moves_robot: bool = False
    report: str = ""

    @property
    def is_run(self) -> bool:
        return self.action == "run"

    def to_dict(self) -> dict:
        return asdict(self)


def parse(result_json) -> Action:
    """Validated Action from a claude result JSON (dict or text) or from the structured_output object itself."""
    obj = json.loads(result_json) if isinstance(result_json, (str, bytes)) else result_json
    if not isinstance(obj, dict):
        raise SchemaError(f"expected a JSON object, got {type(obj).__name__}")
    if "structured_output" in obj or "action" not in obj:
        obj = obj.get("structured_output")
    if not isinstance(obj, dict):
        raise SchemaError("the result has no structured_output object")
    action = obj.get("action")
    if action not in ACTIONS:
        raise SchemaError(f"unknown action {action!r}; expected one of {ACTIONS}")
    extra = set(obj) - set(ACTION_SCHEMA["properties"])
    if extra:
        raise SchemaError(f"unexpected keys {sorted(extra)}")
    for key in ("command", "what", "report"):
        if not isinstance(obj.get(key, ""), str):
            raise SchemaError(f"{key} must be a string")
    if not isinstance(obj.get("moves_robot", False), bool):
        raise SchemaError("moves_robot must be a boolean")
    a = Action(action, obj.get("what", "").strip(), obj.get("command", "").strip(),
               bool(obj.get("moves_robot", False)), obj.get("report", "").strip())
    if a.is_run:
        if not a.command:
            raise SchemaError("run needs a non-empty command")
        if "moves_robot" not in obj:
            raise SchemaError("run needs moves_robot")
        a.what = a.what or a.command
    else:
        a.report = a.report or a.what
        if not a.report:
            raise SchemaError(f"{action} needs a report")
        a.command, a.moves_robot = "", False
    return a


@dataclass
class Usage:
    """Tokens by kind, cost, agent turns and wall time of one claude invocation (cost_usd: this invocation's share)."""
    # Verified on Claude Code 2.1.318: `usage`, `num_turns`, `duration_ms` are per invocation; `total_cost_usd` and
    # `modelUsage` are cumulative over a resumed session. session_cost_usd keeps the reported cumulative value.
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    session_cost_usd: float = 0.0
    turns: int = 0
    duration_ms: int = 0
    model: str = ""
    session_id: str = ""
    is_error: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_result(cls, result: dict, model: str = "") -> "Usage":
        """Usage from the result JSON of `claude -p --output-format json` (missing fields count as zero)."""
        u = result.get("usage") or {}
        models = sorted(result.get("modelUsage") or {})
        return cls(
            input_tokens=int(u.get("input_tokens") or 0),
            cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            cost_usd=float(result.get("total_cost_usd") or 0.0),
            session_cost_usd=float(result.get("total_cost_usd") or 0.0),
            turns=int(result.get("num_turns") or 0),
            duration_ms=int(result.get("duration_ms") or 0),
            model=",".join(models) or model,
            session_id=str(result.get("session_id") or ""),
            is_error=bool(result.get("is_error", False)),
        )
