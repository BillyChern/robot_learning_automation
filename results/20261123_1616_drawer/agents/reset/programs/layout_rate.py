"""No motion: repeat the orchestrator's exact layout measurement to see how often it finds the drawer as its own blob."""
import argparse
import time

import numpy as np

from rlpj import config
from rlpj.perception import layout
from rlpj.perception import tabletop as tt
from rlpj.robot.session import RobotSession

ap = argparse.ArgumentParser()
ap.add_argument("--xy", type=float, nargs=2, required=True)
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()
TASK = config.load("drawer_lying")["task"]
with RobotSession(program="reset") as s:
    s.phase("rate")
    res = []
    for k in range(a.n):
        blobs = tt.detect(tt.cloud(s.side))
        ok, rep = layout.check(TASK, {"drawer": a.xy}, blobs)
        near = [b for b in blobs if np.linalg.norm(np.subtract(b["xy"], a.xy)) < 0.09]
        res.append(dict(ok=bool(ok), dist=rep["objects"]["drawer"]["dist"],
                        near=[dict(xy=np.round(b["xy"], 3).tolist(), h=b["height"], ext=b["extent"], n=b["n"]) for b in near]))
        s.log(f"{k}: ok={ok} {res[-1]['dist']} {res[-1]['near']}")
        time.sleep(0.25)
    n_ok = sum(r["ok"] for r in res)
    d = [r["dist"] for r in res if r["dist"] is not None]
    s.summary(layout_pass=f"{n_ok}/{a.n}", dist_cm=[round(min(d) * 100, 2), round(max(d) * 100, 2)] if d else None)
    s.claim("success" if n_ok == a.n else "fail")
