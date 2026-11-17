"""Policy server for PyTorch checkpoints on the RTX 5090 (sm_120).

The 5090's openpi venv has torch 2.7.1+cu126 (kernels up to sm_90): PyTorch policies fail there with "no kernel image
is available". This server uses torch 2.7.1+cu128 / torchvision 0.22.1+cu128 installed in <workspace>/pylib/torch128
(uv pip install --target), everything else from the openpi venv: the generic packages the target also contains are
imported from the venv first, then the target goes first on sys.path for torch, torchvision, triton and the
nvidia-* libraries. Otherwise like openpi's serve_policy (our configs + recipe registered, metadata from the policy).
Started by scripts/serve.sh as: python serve_torch128.py --checkpoint <dir> --port <port> --config <name>.
The checkpoint dir must contain params/ for serve.sh (use the agents/learn/ckpt/<exp>/<step> shim: params/ +
model.safetensors symlink; openpi takes the PyTorch path when model.safetensors exists)."""
import argparse
import importlib
import logging
import socket
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "pylib" / "torch128"
for m in ("numpy", "PIL", "typing_extensions", "jinja2", "markupsafe", "fsspec", "filelock", "sympy", "networkx",
          "setuptools"):
    try:
        importlib.import_module(m)
    except Exception:
        pass
if TARGET.is_dir():
    sys.path.insert(0, str(TARGET))

import torch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default="pi05_kinova_full")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("torch %s cuda %s archs %s device %s", torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(),
                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    from rlpj.vla import openpi_entry as oe
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    import openpi.training.config as _config
    repo = oe.asset_id_from_checkpoint(a.checkpoint)
    oe._register(repo_id=repo)
    policy = policy_config.create_trained_policy(_config.get_config(a.config), a.checkpoint)
    logging.info("serving %s (config %s, dataset %s) on port %d", a.checkpoint, a.config, repo, a.port)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=a.port,
                                                           metadata=policy.metadata)
    logging.info("host %s", socket.gethostname())
    server.serve_forever()


if __name__ == "__main__":
    main()
