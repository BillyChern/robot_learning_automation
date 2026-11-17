"""Offline policy check on recorded frames: serve-style inference (letterboxed 224 RGB like the client) vs recorded
actions. usage (openpi venv, cwd third_party/openpi): python offline_check.py <ckpt_dir> <config> [recipe] [episodes]"""
import glob
import io
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

ckpt, cfg_name = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3 and sys.argv[3] != "-":
    os.environ["RLPJ_RECIPE"] = sys.argv[3]
eps = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 40, 120]
from rlpj.vla import openpi_entry as oe            # noqa: E402
from rlpj.vla.policy_client import bgr_to_model    # noqa: E402
from openpi.policies import policy_config          # noqa: E402
import openpi.training.config as _config           # noqa: E402

repo = oe.asset_id_from_checkpoint(ckpt)
oe._register(repo_id=repo)
cfg = _config.get_config(cfg_name)
pol = policy_config.create_trained_policy(cfg, ckpt)
ds = sorted(glob.glob(f"/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/datasets/{repo}/data/*/*.parquet"))
INSTR = "pick up the toy bee and put it into the bowl"


def img(b):
    rgb = np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))
    return bgr_to_model(rgb[..., ::-1].copy(), 224)


for e in eps:
    d = pd.read_parquet(ds[e])
    n = len(d)
    S, A = np.stack(d.state.values), np.stack(d.actions.values)
    errs, gerr, rows = [], [], []
    for i in np.linspace(0, n - 40, 12).astype(int):
        r = d.iloc[i]
        out = pol.infer({"image": img(r.image), "wrist_image": img(r.wrist_image), "state": S[i].astype(np.float32), "prompt": INSTR})
        a = np.asarray(out["actions"])
        k = min(len(a), 30)
        gt = A[i:i + k]
        dq = (a[:k, :6] - gt[:, :6] + np.pi) % (2 * np.pi) - np.pi
        errs.append(np.degrees(np.abs(dq)).max(0))
        gerr.append(np.abs(a[:k, 6] - gt[:, 6]).mean())
        mv = np.degrees(np.abs((gt[-1, :6] - S[i, :6] + np.pi) % (2 * np.pi) - np.pi)).max()
        rows.append(f"  t={i / 15:5.1f}s gt_move={mv:5.1f}deg pred_err_max={np.degrees(np.abs(dq)).max():5.1f}deg "
                    f"grip gt={gt[-1, 6]:.2f} pred={a[k - 1, 6]:.2f}")
    print(f"episode file {os.path.basename(ds[e])}: per-joint max err deg (median over samples) "
          f"{np.round(np.median(errs, 0), 1).tolist()}, gripper mae {np.mean(gerr):.2f}")
    print("\n".join(rows))
