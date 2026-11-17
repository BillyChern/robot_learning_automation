"""Does the policy use the images? Same state, images from own / other episode / black: compare predicted chunks.
usage (openpi venv): python vision_check.py <ckpt> <config> <recipe|->"""
import glob, io, os, sys
import numpy as np, pandas as pd
from PIL import Image
ckpt, cfg_name = sys.argv[1], sys.argv[2]
if sys.argv[3] != "-":
    os.environ["RLPJ_RECIPE"] = sys.argv[3]
from rlpj.vla import openpi_entry as oe
from rlpj.vla.policy_client import bgr_to_model
from openpi.policies import policy_config
import openpi.training.config as _config
repo = oe.asset_id_from_checkpoint(ckpt); oe._register(repo_id=repo)
pol = policy_config.create_trained_policy(_config.get_config(cfg_name), ckpt)
ds = sorted(glob.glob(f"/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/datasets/{repo}/data/*/*.parquet"))
I = "pick up the toy bee and put it into the bowl"
def img(b): return bgr_to_model(np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))[..., ::-1].copy(), 224)
def end_q(a, s): return np.degrees((a[-1, :6] - s[:6] + np.pi) % (2*np.pi) - np.pi)
pairs = [(0, 5), (40, 90), (130, 20)]
for ea, eb in pairs:
    A, B = pd.read_parquet(ds[ea]), pd.read_parquet(ds[eb])
    for frac in (0.15, 0.3):
        ia, ib = int(frac * len(A)), int(frac * len(B))
        s = np.asarray(A.state.values[ia], np.float32)
        own = pol.infer({"image": img(A.image.iloc[ia]), "wrist_image": img(A.wrist_image.iloc[ia]), "state": s, "prompt": I})["actions"]
        oth = pol.infer({"image": img(B.image.iloc[ib]), "wrist_image": img(B.wrist_image.iloc[ib]), "state": s, "prompt": I})["actions"]
        blk = pol.infer({"image": np.zeros((224,224,3),np.uint8), "wrist_image": np.zeros((224,224,3),np.uint8), "state": s, "prompt": I})["actions"]
        gt = np.stack(A.actions.values[ia:ia+30])
        print(f"ep{ea} t={ia/15:.0f}s: gt dq_end {np.round(end_q(gt,s),1).tolist()}\n   own {np.round(end_q(np.asarray(own),s),1).tolist()}\n"
              f"   other(ep{eb}) {np.round(end_q(np.asarray(oth),s),1).tolist()}\n   black {np.round(end_q(np.asarray(blk),s),1).tolist()}")
