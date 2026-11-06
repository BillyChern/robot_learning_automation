"""Readable agent conversations for demos: runs/<run>/transcripts/<role>/*.jsonl (Claude Code session files, copied
from the agent host) -> runs/<run>/transcripts/<role>.md with local times, the orchestrator's messages, the agent's
text, every tool call (command, file) and a short excerpt of each result. Thinking is stored encrypted: not shown.

    python -m rlpj.orchestrator.transcript <run>      (also: rlpj transcripts <run>)"""
import argparse
import calendar
import json
import sys
import time
from pathlib import Path

from rlpj import config

RESULT_CHARS, INPUT_CHARS, FIRST_PROMPT_LINES = 1200, 800, 12


def _local(ts):
    """'2026-11-04T20:10:41.989Z' -> '15:10:41' local time."""
    try:
        return time.strftime("%H:%M:%S", time.localtime(calendar.timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))))
    except (TypeError, ValueError):
        return "--:--:--"


def _cut(text, n):
    text = str(text).strip()
    return text if len(text) <= n else text[:n] + f"\n... ({len(text) - n} more characters)"


def _tool_input(name, inp):
    """One readable line or block for a tool call."""
    if name == "Bash":
        return f"```bash\n{_cut(inp.get('command', ''), INPUT_CHARS)}\n```"
    if name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit"):
        extra = f" ({len(str(inp.get('content', '')).splitlines())} lines)" if name == "Write" else ""
        return f"`{inp.get('file_path', '')}`{extra}"
    return f"```json\n{_cut(json.dumps(inp, indent=1), INPUT_CHARS)}\n```"


def _result_text(content):
    if isinstance(content, list):
        return "\n".join(b.get("text", "[image]" if b.get("type") == "image" else "") for b in content if isinstance(b, dict))
    return str(content or "")


def render(jsonl, role):
    """Markdown of one session file."""
    out, first = [], True
    for line in Path(jsonl).read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        t, c = _local(d.get("timestamp")), d["message"]["content"]
        if d["type"] == "user" and isinstance(c, str):
            lines = c.strip().splitlines()
            if first:
                c = "\n".join(lines[:FIRST_PROMPT_LINES]) + f"\n\n... (full prompt: agents/{role}/PROMPT.md, {len(lines)} lines)"
                first = False
            out += [f"### {t} orchestrator -> {role} agent", "", "```text", _cut(c, 4000), "```", ""]
            continue
        for b in c if isinstance(c, list) else []:
            k = b.get("type")
            if k == "text" and b.get("text", "").strip():
                out += [f"**{t} {role} agent:** {b['text'].strip()}", ""]
            elif k == "tool_use":
                inp = _tool_input(b.get("name"), b.get("input") or {})
                out += [f"**{t} tool {b.get('name')}:**" + ("\n\n" if inp.startswith("```") else " ") + inp, ""]
            elif k == "tool_result":
                r = _result_text(b.get("content"))
                tag = " (error)" if b.get("is_error") else ""
                out += [f"<details><summary>{t} result{tag}</summary>", "", "```text", _cut(r, RESULT_CHARS), "```", "</details>", ""]
    return out


def build(run_dir, log=print):
    """Write transcripts/<role>.md for every archived role; returns the paths."""
    run_dir, paths = Path(run_dir), []
    for d in sorted(p for p in (run_dir / "transcripts").glob("*") if p.is_dir()):
        role = d.name.rsplit("-", 1)[-1] if d.name.startswith("-") else d.name
        files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            continue
        body = [f"# {role} agent: conversation", "", f"run `{run_dir.name}`, sessions: " + ", ".join(f.stem for f in files), ""]
        for f in files:
            body += render(f, role)
        p = run_dir / "transcripts" / f"{role}.md"
        p.write_text("\n".join(body) + "\n")
        paths.append(p)
        log(f"transcript: {role} -> {p} ({p.stat().st_size // 1024} KB)")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="run id (runs/<run>)")
    build(config.RUNS / p.parse_args(argv).run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
