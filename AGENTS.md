# Working in this repo

- Read `docs/DESIGN.md` before changing interfaces; it is the contract between modules.
- Robot venv: `.venv/bin/python` (uv, `--extra robot`). Tests: `.venv/bin/python -m pytest -q tests/` (offline only).
- Never set `RLPJ_MOTION=1` by hand. Robot motion goes through `rlpj.orchestrator.attempts.run_attempt` with the
  operator present. Read-only robot and camera access is fine (`python -m rlpj.robot.check`).
- Cameras are single-owner: do not hold the side camera or the wrist stream while a robot program runs.
- Config comes from `rlpj.config.load()` and `rlpj.config.robot()`; no hard-coded IPs, serials or paths.
- Style: concise, one-line docstrings, plain names, no emojis. Keep files focused and short.
- Training code runs on the 5090 inside `third_party/openpi/.venv` (`uv run`), with this repo on `PYTHONPATH`.
