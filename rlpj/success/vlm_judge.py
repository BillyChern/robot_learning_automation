"""VLM judge: headless Claude Code on the agent host reads before/after images and returns a strict judgement.

CLI: python -m rlpj.success.vlm_judge before_side.jpg after_side.jpg [before_wrist.jpg after_wrist.jpg] [--dry]"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

from rlpj import config

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["success", "confidence", "reason"],
          "properties": {"success": {"type": ["boolean", "null"]},
                         "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                         "reason": {"type": "string"}}}
REMOTE_ROOT = "/tmp/rlpj_judge"
KEYS = ("before_side", "after_side", "before_wrist", "after_wrist")


def judge(images, task_cfg, extra="", dry=False, timeout=300, host=None, model=None, effort=None):
    """images {name: path} (e.g. before_side, after_side, before_wrist, after_wrist) -> dict(success: bool|None,
    confidence, reason, cost_usd, usage, ...). Images are rsynced to the agent host and judged by `claude -p`
    with a JSON schema; any failure gives success None with the reason. dry=True returns the prompt and command."""
    cfg = config.load()
    host = host or cfg["hosts"]["agent"]
    model = model or cfg["vlm"]["model"]
    effort = effort or cfg["vlm"]["effort"]
    names = {k: f"{k}{os.path.splitext(str(p))[1].lower() or '.jpg'}" for k, p in images.items()}
    remote = f"{REMOTE_ROOT}/{uuid.uuid4().hex[:12]}"
    text = prompt(names, task_cfg, extra)
    cmd = command(remote, model, effort)
    out = dict(success=None, confidence=0.0, reason="", cost_usd=0.0, usage=None, model=model, host=host, remote=remote)
    if dry:
        return dict(out, prompt=text, command=cmd, reason="dry run")
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for k, p in images.items():
                os.symlink(os.path.abspath(str(p)), os.path.join(tmp, names[k]))
            _run(["rsync", "-aqL", "--rsync-path", f"mkdir -p {shlex.quote(remote)} && rsync", tmp + "/",
                  f"{host}:{remote}/"], timeout=60)
        r = subprocess.run(["ssh", host, cmd], input=text, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(out, reason=f"timeout after {timeout} s", duration_s=round(time.time() - t0, 1))
    except (OSError, RuntimeError) as e:
        return dict(out, reason=f"judge failed: {e}", duration_s=round(time.time() - t0, 1))
    out["duration_s"] = round(time.time() - t0, 1)
    if r.returncode != 0:
        return dict(out, reason=f"claude exit {r.returncode}: {(r.stderr or r.stdout)[-400:].strip()}")
    return dict(out, **parse(r.stdout))


def parse(stdout):
    """Fields from the `claude -p --output-format json` result (structured_output, total_cost_usd, usage, is_error)."""
    try:
        res = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
    except json.JSONDecodeError:
        return dict(success=None, confidence=0.0, reason=f"unparsable judge output: {stdout[-300:].strip()}")
    so = res.get("structured_output") or {}
    out = dict(cost_usd=float(res.get("total_cost_usd") or 0), usage=res.get("usage"), is_error=bool(res.get("is_error")))
    if res.get("is_error") or "success" not in so:
        return dict(out, success=None, confidence=0.0,
                    reason=f"judge error: {str(res.get('result') or res)[:300]}")
    return dict(out, success=so["success"], confidence=float(so.get("confidence") or 0), reason=str(so.get("reason", "")))


def prompt(names, task_cfg, extra=""):
    """Judge prompt: the image files to Read, the task goal and success sentence, strict rules, the answer format."""
    files = "\n".join(f"- {f}: {k.replace('_', ' ')} the attempt" for k, f in names.items())
    return f"""You judge whether a robot arm completed a tabletop task. Read every image file listed below (they are in the
current directory), compare before and after, then answer in the required JSON.

Task: {task_cfg['goal'].strip()}
Success means: {task_cfg['success'].strip()}

Images:
{files}

Rules: judge only from what the after images show. Count success only when the success condition is clearly visible;
objects knocked over, resting on a rim or edge instead of inside, or still held by the gripper are not success.
If the view does not allow a judgement (occlusion, missing image), answer success null with low confidence.
{extra.strip()}
Answer: success (true/false/null), confidence 0..1, reason (one sentence naming what you saw change)."""


def command(remote, model, effort):
    """Remote shell command: cd to the image directory, run claude -p with the schema, remove the directory."""
    args = ["claude", "-p", "--model", model, "--effort", str(effort), "--output-format", "json", "--json-schema",
            json.dumps(SCHEMA), "--permission-mode", "acceptEdits", "--tools", "Read", "--strict-mcp-config",
            "--disable-slash-commands", "--no-session-persistence"]
    q = shlex.quote(remote)
    return (f"export PATH=$HOME/.local/bin:$PATH; cd {q} && " + " ".join(shlex.quote(a) for a in args)
            + f"; rc=$?; cd / && rm -rf {q}; exit $rc")


def _run(cmd, timeout):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exit {r.returncode}: {r.stderr.strip()[-300:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="VLM success judge (images in order: " + ", ".join(KEYS) + ")")
    ap.add_argument("images", nargs="+", help="image files; names like before_side.jpg are used as their role")
    ap.add_argument("--task", default=None, help="task name (default: config)")
    ap.add_argument("--extra", default="", help="extra text for the prompt")
    ap.add_argument("--dry", action="store_true", help="print the prompt and command without calling the model")
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args(argv)
    images = {}
    for i, p in enumerate(a.images):
        stem = os.path.splitext(os.path.basename(p))[0]
        images[stem if stem in KEYS else (KEYS[i] if i < len(KEYS) else stem)] = p
    res = judge(images, config.load(a.task)["task"], extra=a.extra, dry=a.dry, timeout=a.timeout)
    if a.dry:
        print(res.pop("prompt"), "\n\n$ ssh", res["host"], res.pop("command"))
    print(json.dumps(res, indent=1, default=str))
    return 0 if res.get("success") is not None or a.dry else 1


if __name__ == "__main__":
    sys.exit(main())
