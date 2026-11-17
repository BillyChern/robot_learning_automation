"""Full fine-tuning on the RTX 5090 (sm_120) with openpi's PyTorch trainer, via rlpj.vla.openpi_entry, using the
torch 2.7.1+cu128 in <workspace>/pylib/torch128 (the openpi venv's cu126 torch has no sm_120 kernels).
Same steps as scripts/train.sh: norm stats for (config, dataset), then train-pytorch with the given overrides.
usage (openpi venv python, cwd third_party/openpi, PYTHONPATH=<repo>):
  python train_5090.py --run-dir R --dataset NAME --exp-name EXP [--config pi05_kinova_full] [tyro overrides...]
Used because the 4090's GPU was held by other users' jobs (2026-11-09 01:47, 19.7 of 24.5 GB)."""
import importlib
import os
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "pylib" / "torch128"
for m in ("numpy", "PIL", "typing_extensions", "jinja2", "markupsafe", "fsspec", "filelock", "sympy", "networkx", "setuptools"):
    try:
        importlib.import_module(m)
    except Exception:
        pass
sys.path.insert(0, str(TARGET))
os.environ["PYTHONPATH"] = os.pathsep.join([str(TARGET)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
import torch  # noqa: E402

from rlpj.vla import openpi_entry as oe  # noqa: E402


def main():
    argv = sys.argv[1:]

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv[:-1] else default

    run_dir, dataset, config = opt("--run-dir"), opt("--dataset"), opt("--config", "pi05_kinova_full")
    rest = list(argv)
    if "--config" in rest:
        i = rest.index("--config")
        del rest[i:i + 2]
    print(f"[train_5090] torch {torch.__version__} archs {torch.cuda.get_arch_list()[-1:]} "
          f"device {torch.cuda.get_device_name(0)}; config {config}", flush=True)
    if os.environ.get("SKIP_NORM") != "1":
        oe.main(["norm-stats", "--run-dir", run_dir, "--dataset", dataset, "--config-name", config])
    # train-pytorch: positional config name first, then our flags (run-dir/dataset parsed by openpi_entry)
    head = ["--run-dir", run_dir, "--dataset", dataset]
    others = [a for a in rest]
    for k in ("--run-dir", "--dataset"):
        i = others.index(k)
        del others[i:i + 2]
    oe.main(["train-pytorch", *head, config, *others])


if __name__ == "__main__":
    main()
