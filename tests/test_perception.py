"""Offline perception tests: synthetic side depth images (z-buffer splats of table + objects), layout, wrist."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from rlpj import config
from rlpj.perception import layout, tabletop, wrist
from rlpj.robot import calib

TZ = -0.023
K_SIDE = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
DIST0 = np.zeros(5)
RING = dict(xy=(0.48, -0.08), r_out=0.050, r_in=0.038, h=0.025)
BEE = dict(xy=(0.40, 0.06), sx=0.05, sy=0.03, h=0.02)
CUP = dict(xy=(0.52, -0.12), r_top=0.022, r_rim=0.036, h=0.095)


def task(name):
    return yaml.safe_load((config.ROOT / "config/tasks" / f"{name}.yaml").read_text())


def look_at(pos, aim):
    z = np.subtract(aim, pos); z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x = x / np.linalg.norm(x)
    T = np.eye(4); T[:3, :3] = np.c_[x, np.cross(z, x), z]; T[:3, 3] = pos
    return T


T_SIDE = look_at([1.18, 0.05, 0.58], [0.5, -0.1, TZ])


def grid2(xa, xb, ya, yb, step):
    gx, gy = np.meshgrid(np.arange(xa, xb, step), np.arange(ya, yb, step), indexing="ij")
    return np.c_[gx.ravel(), gy.ravel()]


def table_pts(step=0.002):
    xy = grid2(0.15, 1.0, -0.5, 0.3, step)
    return np.c_[xy, np.full(len(xy), TZ)]


def wall(curve_xy, h, step):
    """Vertical wall points along a closed/open curve (K,2) from the table up to height h."""
    zs = np.arange(0, h, step)
    return np.c_[np.repeat(curve_xy, len(zs), 0), TZ + np.tile(zs, len(curve_xy))]


def circle(c, r, step):
    a = np.arange(0, 2 * np.pi, step / r)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


def ring_pts(xy, r_out, r_in, h, step=0.0007):
    top = grid2(xy[0] - r_out, xy[0] + r_out, xy[1] - r_out, xy[1] + r_out, step)
    r = np.linalg.norm(top - xy, axis=1)
    top = top[(r <= r_out) & (r >= r_in)]
    return np.vstack([np.c_[top, np.full(len(top), TZ + h)], wall(circle(xy, r_out, step), h, step),
                      wall(circle(xy, r_in, step), h, step)])


def box_pts(xy, sx, sy, h, step=0.0007):
    top = grid2(xy[0] - sx / 2, xy[0] + sx / 2, xy[1] - sy / 2, xy[1] + sy / 2, step)
    x0, x1, y0, y1 = xy[0] - sx / 2, xy[0] + sx / 2, xy[1] - sy / 2, xy[1] + sy / 2
    xs, ys = np.arange(x0, x1, step), np.arange(y0, y1, step)
    edge = np.vstack([np.c_[xs, np.full(len(xs), y0)], np.c_[xs, np.full(len(xs), y1)],
                      np.c_[np.full(len(ys), x0), ys], np.c_[np.full(len(ys), x1), ys]])
    return np.vstack([np.c_[top, np.full(len(top), TZ + h)], wall(edge, h, step)])


def cup_pts(xy, r_top, r_rim, h, step=0.0007):
    """Upside-down tapered cup: closed disc on top, cone wall widening to the rim on the table."""
    top = grid2(xy[0] - r_top, xy[0] + r_top, xy[1] - r_top, xy[1] + r_top, step)
    top = top[np.linalg.norm(top - xy, axis=1) <= r_top]
    pts = [np.c_[top, np.full(len(top), TZ + h)]]
    for z in np.arange(0, h, step):
        c = circle(xy, r_rim + (r_top - r_rim) * z / h, step)
        pts.append(np.c_[c, np.full(len(c), TZ + z)])
    return np.vstack(pts)


def render(point_sets, T_base_cam=T_SIDE, K=K_SIDE, shape=(720, 1280), noise=0.002, gain=(1.01, 0.005), seed=0):
    """Z-buffer splat of base-frame surface points -> depth image (m) with noise and a range-dependent long error."""
    rng = np.random.default_rng(seed)
    P = np.vstack(point_sets)
    Pc = calib.apply(np.linalg.inv(T_base_cam), P)
    Pc = Pc[Pc[:, 2] > 0.1]
    uv = np.rint(calib.project(Pc, K, DIST0)).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    D = np.full(shape, np.inf)
    np.minimum.at(D, (uv[ok, 1], uv[ok, 0]), Pc[ok, 2])
    D[~np.isfinite(D)] = 0
    m = D > 0
    D[m] = D[m] * (gain[0] + gain[1] * D[m]) + rng.normal(0, noise, m.sum())
    return D.astype(np.float32)


class FakeCam:
    def __init__(self, D):
        self.D = D

    def depth_median(self, n=5, timeout=3.0):
        return self.D


def scene_cloud(objects, **kw):
    D = render([table_pts()] + objects, **kw)
    return tabletop.cloud(FakeCam(D), n=1, K=K_SIDE, dist=DIST0, T=T_SIDE, table_z=TZ)


@pytest.fixture(scope="module")
def bee_scene():
    return scene_cloud([ring_pts(**RING), box_pts(**BEE)])


def test_depth_correction_flattens_table(bee_scene):
    D = render([table_pts()])
    P, _ = calib.deproject(D, K_SIDE, DIST0)
    k, info = tabletop.depth_correction(P, T_SIDE, table_z=TZ)
    assert info["n"] > 1000 and info["table_resid_mm"] < 3.0
    assert 0.9 < k[0] < 1.0                       # the synthetic depth reads long
    assert abs(tabletop.table_fit(bee_scene, z0=TZ) - TZ) < 0.002
    assert abs(tabletop.table_fit(bee_scene) - TZ) < 0.002


def test_detect_ring_and_bee(bee_scene):
    blobs = tabletop.detect(bee_scene, table_z=TZ)
    assert len(blobs) == 2
    assert [np.hypot(*b["xy"]) for b in blobs] == sorted(np.hypot(*b["xy"]) for b in blobs)
    ring = max(blobs, key=lambda b: b["outer_r"] or 0)
    bee = min(blobs, key=lambda b: b["outer_r"] or 0)
    assert ring["hollow"] and not bee["hollow"]
    assert abs(ring["outer_r"] - RING["r_out"]) < 0.006
    assert abs(ring["inner_r"] - RING["r_in"]) < 0.006
    assert np.linalg.norm(np.subtract(ring["xy"], RING["xy"])) < 0.006
    assert np.linalg.norm(np.subtract(ring["hole_xy"], RING["xy"])) < 0.006
    assert abs(ring["height"] - RING["h"]) < 0.006
    assert np.linalg.norm(np.subtract(bee["xy"], BEE["xy"])) < 0.01
    assert abs(bee["height"] - BEE["h"]) < 0.006
    assert abs(max(bee["extent"]) - BEE["sx"]) < 0.012 and abs(min(bee["extent"]) - BEE["sy"]) < 0.012
    assert bee["inner_r"] < 0.01 and len(ring["table_xy"]) > 0
    assert ring["points"].dtype == np.float32 and ring["points"].shape[1] == 3


def test_detect_filled_ring_is_not_hollow():
    P = scene_cloud([ring_pts(**RING), box_pts(RING["xy"], 0.05, 0.04, 0.02)])
    blobs = tabletop.detect(P, table_z=TZ)
    assert len(blobs) == 1
    b = blobs[0]
    assert not b["hollow"] and abs(b["outer_r"] - RING["r_out"]) < 0.006
    assert np.linalg.norm(np.subtract(b["xy"], RING["xy"])) < 0.008


def test_detect_cups_and_priors():
    c2 = (CUP["xy"][0] - 0.15, CUP["xy"][1] + 0.01)
    P = scene_cloud([cup_pts(**CUP), cup_pts(c2, CUP["r_top"], CUP["r_rim"], CUP["h"])])
    blobs = tabletop.detect(P, table_z=TZ)
    assert len(blobs) == 2 and all(not b["hollow"] for b in blobs)
    assert all(abs(b["height"] - CUP["h"]) < 0.006 for b in blobs)
    assert np.linalg.norm(np.subtract(blobs[0]["xy"], c2)) < 0.012     # nearer the robot first
    assert np.linalg.norm(np.subtract(blobs[1]["xy"], CUP["xy"])) < 0.012
    pick = layout.assign(task("cup_down_stacking"), blobs)
    assert np.linalg.norm(np.subtract(pick["move_cup"]["xy"], c2)) < 0.012
    assert np.linalg.norm(np.subtract(pick["base_cup"]["xy"], CUP["xy"])) < 0.012


def test_summarize_json(bee_scene):
    s = tabletop.summarize(tabletop.detect(bee_scene, table_z=TZ))
    txt = json.dumps(s)
    assert "points" not in txt and len(s) == 2 and set(s[0]) == set(tabletop.BLOB_KEYS)


def test_detect_empty_and_workspace():
    assert tabletop.detect(np.zeros((0, 3)), table_z=TZ) == []
    P = scene_cloud([ring_pts(**RING)])
    assert tabletop.detect(P, workspace=dict(x=(0.6, 0.9), y=(-0.4, 0.2)), table_z=TZ) == []


@pytest.mark.parametrize("name", ["bee_in_tape_roll", "cup_down_stacking", "bee_in_bowl"])
def test_layout_sample(name):
    cfg = task(name)
    lay = cfg["layout"]
    rng = np.random.default_rng(1)
    for _ in range(500):
        t = layout.sample(cfg, rng)
        assert set(t) == set(cfg["objects"])
        pts = {o: np.array(v) for o, v in t.items()}
        for o, s in ((o, lay[o]) for o in cfg["objects"]):
            if "from" in s:
                v = pts[o] - pts[s["from"]]
                assert s["distance"][0] - 1e-3 <= np.hypot(*v) <= s["distance"][1] + 1e-3
                a = np.degrees(np.arctan2(v[1], v[0])) % 360
                assert s["angle"][0] - 0.5 <= a <= s["angle"][1] + 0.5
            else:
                assert s["x"][0] <= pts[o][0] <= s["x"][1] and s["y"][0] <= pts[o][1] <= s["y"][1]
        names = list(pts)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert np.linalg.norm(pts[names[i]] - pts[names[j]]) >= lay["min_gap"] - 1e-3
            assert lay["radius"][0] - 1e-3 <= np.hypot(*pts[names[i]]) <= lay["radius"][1] + 1e-3
    assert layout.sample(cfg, 3) == layout.sample(cfg, 3)


def fake_blob(xy, height, extent, hollow=False, inner_r=None, outer_r=None):
    return dict(xy=list(xy), z_top=TZ + height, height=height, extent=list(extent), n=300, hollow=hollow,
                hole_xy=list(xy) if hollow else None, inner_r=inner_r, outer_r=outer_r,
                points=np.zeros((0, 3), np.float32), table_xy=np.zeros((0, 2), np.float32))


def test_assign_and_check_bee():
    cfg = task("bee_in_tape_roll")
    ring = fake_blob((0.48, -0.08), 0.025, (0.10, 0.10), hollow=True, inner_r=0.038, outer_r=0.05)
    bee = fake_blob((0.40, 0.06), 0.02, (0.05, 0.03), inner_r=0.003, outer_r=0.02)
    tall = fake_blob((0.60, -0.3), 0.10, (0.07, 0.07), inner_r=0.0, outer_r=0.022)
    pick = layout.assign(cfg, [tall, bee, ring])
    assert pick["tape_roll"] is ring and pick["foam_bee"] is bee
    assert layout.assign(cfg, [tall])["tape_roll"] is None
    ok, rep = layout.check(cfg, {"tape_roll": [0.49, -0.07], "foam_bee": [0.41, 0.05]}, [ring, bee, tall])
    assert ok and rep["extra_blobs"] == 1 and rep["objects"]["tape_roll"]["dist"] < 0.02
    ok, rep = layout.check(cfg, {"tape_roll": [0.49, -0.07], "foam_bee": [0.30, 0.05]}, [ring, bee])
    assert not ok and "foam_bee" in rep["reason"] and rep["objects"]["foam_bee"]["ok"] is False
    ok, rep = layout.check(cfg, {"tape_roll": [0.49, -0.07], "foam_bee": [0.30, 0.05]}, [ring])
    assert not ok and rep["objects"]["foam_bee"]["found"] is None


def test_assign_cups_by_targets_and_relation():
    cfg = task("cup_down_stacking")
    a = fake_blob((0.50, -0.10), 0.095, (0.07, 0.07), outer_r=0.022)
    b = fake_blob((0.35, -0.09), 0.095, (0.07, 0.07), outer_r=0.022)
    pick = layout.assign(cfg, [a, b])
    assert pick["base_cup"] is a and pick["move_cup"] is b          # move_cup lies towards the robot
    pick = layout.assign(cfg, [a, b], targets={"base_cup": [0.35, -0.09], "move_cup": [0.50, -0.10]})
    assert pick["base_cup"] is b and pick["move_cup"] is a          # targets override the relation
    ok, rep = layout.check(cfg, {"base_cup": [0.505, -0.10], "move_cup": [0.35, -0.085]}, [a, b])
    assert ok


def T_tool_down(x, y, z):
    T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0]); T[:3, 3] = [x, y, z]
    return T


T_TC = np.array([[-1, 0, 0, 0], [0, -1, 0, 0.05], [0, 0, 1, -0.12], [0, 0, 0, 1.0]])
K_W = np.array([[1300.0, 0, 640], [0, 1300.0, 360], [0, 0, 1]])


def test_wrist_projection_and_circle_refine():
    T_bt = T_tool_down(0.50, -0.10, 0.25)
    c_true, z, r = np.array([0.50, -0.10]), TZ + 0.025, 0.05
    uv = wrist.project_base_to_wrist([[c_true[0], c_true[1], z]], T_bt, K=K_W, dist=DIST0, T_tool_cam=T_TC)[0]
    T_bc = T_bt @ T_TC
    depth = (T_bc[:3, 3] - [c_true[0], c_true[1], z]) @ T_bc[:3, 2] * -1
    assert depth > 0.2
    rpx = K_W[0, 0] * r / depth
    img = np.full((720, 1280, 3), 230, np.uint8)
    cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), int(round(rpx)), (40, 40, 60), -1)
    cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), int(round(0.76 * rpx)), (230, 230, 230), -1)
    guess = c_true + [0.01, -0.008]
    xy, info = wrist.circle_refine(img, T_bt, guess, z, r, K=K_W, dist=DIST0, T_tool_cam=T_TC)
    assert xy is not None, info
    assert np.linalg.norm(xy - c_true) < 0.002 and info["coverage"] > 0.8
    assert wrist.roi(img, uv, 50).shape == (100, 100, 3)
    behind = wrist.project_base_to_wrist([[0.5, -0.1, 0.5]], T_bt, K=K_W, dist=DIST0, T_tool_cam=T_TC)
    assert np.isnan(behind).all()
    xy2, info2 = wrist.circle_refine(np.full((720, 1280, 3), 230, np.uint8), T_bt, guess, z, r, K=K_W, dist=DIST0, T_tool_cam=T_TC)
    assert xy2 is None and info2["err"]


def test_gripper_load_on_real_wrist_images():
    """run 20261104_1510_bee: the foam bee stuck on the fingers vs released below them vs an empty gripper."""
    chk = config.load("bee_in_bowl")["task"]["gripper_check"]
    d = config.ROOT / "tests/data/wrist"
    f = {p.stem: wrist.gripper_load(cv2.imread(str(p)), chk) for p in d.glob("*.jpg")}
    assert min(f["stuck_after_release"], f["stuck_at_home"], f["stuck_orange_at_home"]) > 1.5 * chk["max_frac"]
    assert f["released_below"] < chk["max_frac"] and f["empty_at_home"] < 0.02
    assert wrist.gripper_load(None, chk) is None
