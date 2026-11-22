"""Near-goal start check: the drawer front's distance out of the bottom slot from side-camera points (offline)."""
import types

import numpy as np

from rlpj import config
from rlpj.orchestrator.loop import Pipeline


def _blob(out_m, task):
    fx = task["fixtures"]["cabinet"]
    n = np.array(fx["front_normal"]); t = np.array([-n[1], n[0]])
    ts = np.linspace(fx["slot_t"][0] + 0.01, fx["slot_t"][1] - 0.01, 30)
    hs = np.linspace(fx["slot_h"][0] + 0.01, fx["slot_h"][1] - 0.01, 10)
    P = [np.r_[(fx["front_offset"] - out_m) * n + tt * t, h] for tt in ts for h in hs]
    return dict(points=np.array(P), z_top=0.07, height=0.07)


def test_slot_state():
    task = config.load("drawer_rl")["task"]
    o = types.SimpleNamespace(task=task)
    ok, rep = Pipeline._slot_state(o, {"drawer_out": 0.04}, [_blob(0.04, task)])
    assert ok and abs(rep["out_m"] - 0.04) < 1e-3
    ok, rep = Pipeline._slot_state(o, {"drawer_out": 0.04}, [_blob(0.09, task)])
    assert not ok
    ok, rep = Pipeline._slot_state(o, {"drawer_out_range": [0.005, 0.09]}, [_blob(0.02, task)])
    assert ok
