#!/bin/bash
# Install openpi into third_party/openpi, check JAX on the GPU, download pi05_base (training machine only).
#   bash scripts/setup_openpi.sh               5090: training + serving (log: third_party/openpi_setup.log)
#   bash scripts/setup_openpi.sh --serve-only  4090: serving fine-tuned checkpoints (no pi05_base download)
#   OPENPI_REF=<commit> bash scripts/setup_openpi.sh   another revision (default: third_party/openpi.commit, else main)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"
REF="${OPENPI_REF:-$(cat "$ROOT/third_party/openpi.commit" 2>/dev/null || echo main)}"
SERVE_ONLY=0; [ "${1:-}" = "--serve-only" ] && SERVE_ONLY=1

if [ ! -d "$OPENPI/.git" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
fi
cd "$OPENPI"
git fetch -q origin
git checkout -q "$REF"
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync

SERVE_ONLY=$SERVE_ONLY uv run python - <<'PY'
import os
import jax, jax.numpy as jnp
print("jax", jax.__version__, jax.devices())
print("matmul ok:", float((jnp.ones((512, 512)) @ jnp.ones((512, 512))).sum().block_until_ready()))
if os.environ["SERVE_ONLY"] != "1":
    from openpi.shared import download
    print("pi05_base:", download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
PY

git rev-parse HEAD > "$ROOT/third_party/openpi.commit"
echo "openpi ready at $(cat "$ROOT/third_party/openpi.commit")"

# PyTorch full fine-tuning / serving (2026-11-07). openpi's lock pins torch 2.7.1 built for CUDA 12.6, which has no
# kernels for Blackwell (RTX 5090, sm_120): install the CUDA 12.8 build, and apply openpi's transformers patch as fresh
# files (cp --remove-destination) so the hard-linked uv cache is not modified. Our scripts then run with UV_NO_SYNC=1.
(cd third_party/openpi && \
  uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchvision \
    "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" --index-url https://download.pytorch.org/whl/cu128 && \
  cp -r --remove-destination src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/)
