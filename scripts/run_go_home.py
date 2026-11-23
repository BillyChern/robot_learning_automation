"""Run scripts/go_home.py through run_attempt (recorded, model-checked)."""
import os, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlpj.orchestrator.attempts import run_attempt  # noqa: E402
work = ROOT / "runs/drawers_setup/home"; work.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "scripts/go_home.py", work / "go_home.py")
r = run_attempt(str(work), "python go_home.py", str(work / time.strftime("home_%Y%m%d_%H%M%S")), True,
                extra_env={"RLPJ_TASK": os.environ.get("RLPJ_TASK", "drawer_lying")}, timeout_s=600)
print("home:", r.exit_code, r.outcome)
