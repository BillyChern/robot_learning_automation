"""Policy server that picks the right config + recipe for a checkpoint even when its training has not finished
(train_info.json is only written at the end of training, so serve.sh falls back to pi05_kinova_lora: wrong transforms).
Started by serve.sh as: python serve_v.py --checkpoint <dir> --port <p> --config <name> (openpi venv)."""
import argparse
import json
import os
import pathlib
import sys

WS = pathlib.Path(__file__).resolve().parent
KNOWN = {"lora_v4": ("pi05_kinova_lora_v4", WS / "recipe_v4.py"),
         "lora_v3": ("pi05_kinova_lora_v3", WS / "recipe_v3.py"),
         "lora_h30": ("pi05_kinova_lora_h30", WS / "recipe_h30.py")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default="")
    a = p.parse_args()
    ckpt = pathlib.Path(a.checkpoint)
    info = ckpt.parent / "train_info.json"
    exp = ckpt.parent.name
    if info.exists():
        d = json.loads(info.read_text())
        config, recipe = d.get("config"), d.get("recipe")
    elif exp in KNOWN:
        config, recipe = KNOWN[exp][0], str(KNOWN[exp][1])
    else:
        config, recipe = a.config or "pi05_kinova_lora", None
    if recipe:
        os.environ["RLPJ_RECIPE"] = recipe
    print(f"[serve_v] {ckpt}: config {config}, recipe {recipe}", flush=True)
    from rlpj.vla import openpi_entry as oe
    oe.cmd_serve(["--port", str(a.port), "policy:checkpoint", "--policy.config", config, "--policy.dir", str(ckpt)])


if __name__ == "__main__":
    sys.exit(main())
