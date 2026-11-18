"""drawer_rl success check on synthetic side-depth points (offline)."""
import numpy as np
import yaml

from rlpj import config
from rlpj.success import checks

TASK = yaml.safe_load((config.ROOT / "config/tasks/drawer_rl.yaml").read_text())
FX = TASK["fixtures"]["cabinet"]
N = np.asarray(FX["front_normal"])
T_DIR = np.array([-N[1], N[0]])
TABLE = -0.016


def slot_points(offset_d, n=400, seed=0):
    """Points spread over the bottom-slot window at signed distance offset_d from the front plane."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(*FX["slot_t"], n)
    h = rng.uniform(*FX["slot_h"], n)
    d = offset_d + rng.normal(0, 0.003, n)
    xy = np.outer(FX["front_offset"] + d, N) + np.outer(t, T_DIR)
    return np.c_[xy, TABLE + h]


def blob(points, height=0.27, extent=(0.12, 0.25)):
    xy = points[:, :2].mean(0)
    return dict(xy=list(xy), z_top=TABLE + height, height=height, extent=list(extent), n=len(points), points=points)


def test_flush_is_success():
    ok, rep = checks.drawer_rl([], [blob(slot_points(0.0))], TASK)
    assert ok is True, rep


def test_empty_slot_fails():
    ok, rep = checks.drawer_rl([], [blob(slot_points(0.08))], TASK)
    assert ok is False and "empty" in rep["reason"], rep


def test_sticking_out_fails():
    ok, rep = checks.drawer_rl([], [blob(slot_points(-0.04))], TASK)
    assert ok is False and "sticks out" in rep["reason"], rep


def test_drawer_on_table_fails():
    rng = np.random.default_rng(1)
    on_table = np.c_[rng.uniform(0.35, 0.45, 300), rng.uniform(-0.15, -0.05, 300), TABLE + rng.uniform(0, 0.06, 300)]
    ok, rep = checks.drawer_rl([], [blob(slot_points(0.0)), blob(on_table, height=0.06, extent=(0.10, 0.2))], TASK)
    assert ok is False and rep["drawer_on_table"], rep
