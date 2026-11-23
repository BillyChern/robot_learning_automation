"""Run scripts/measure_lying_cabinet.py through run_attempt (recorded, model-checked): python scripts/run_measure_lying.py [--dry]"""
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlpj.orchestrator.attempts import run_attempt  # noqa: E402

dry = "--dry" in sys.argv
work = ROOT / "runs/drawers_setup/lying"
work.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "scripts/measure_lying_cabinet.py", work / "measure_lying_cabinet.py")
name = time.strftime("measure_%Y%m%d_%H%M%S") + ("_dry" if dry else "")
r = run_attempt(str(work), "python measure_lying_cabinet.py", str(work / name), not dry,
                extra_env={"RLPJ_TASK": os.environ.get("RLPJ_TASK", "drawer_lying")}, timeout_s=900)
print("measure:", r.exit_code, r.outcome, work / name)
