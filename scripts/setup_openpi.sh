#!/bin/bash
# Training machine (5090): install openpi into third_party/openpi, check JAX on the GPU, download pi05_base.
#   bash scripts/setup_openpi.sh            (log: third_party/openpi_setup.log)
#   OPENPI_REF=<commit> bash scripts/setup_openpi.sh   to pin another revision
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"
REF="${OPENPI_REF:-main}"

if [ ! -d "$OPENPI/.git" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
fi
cd "$OPENPI"
git fetch -q origin
git checkout -q "$REF"
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync

uv run python - <<'EOF'
import jax, jax.numpy as jnp
print("jax", jax.__version__, jax.devices())
print("matmul ok:", float((jnp.ones((512, 512)) @ jnp.ones((512, 512))).sum().block_until_ready()))
from openpi.shared import download
print("pi05_base:", download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
EOF

git rev-parse HEAD > "$ROOT/third_party/openpi.commit"
echo "openpi ready at $(cat "$ROOT/third_party/openpi.commit")"
