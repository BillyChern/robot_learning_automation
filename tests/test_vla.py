"""Offline tests for rlpj.vla (4090 venv; no openpi, no lerobot, no robot)."""
import json

import msgpack
import numpy as np
import pytest

from rlpj.robot.kin import Kin
from rlpj.robot.plan import DT, NORMAL, tool_down_T
from rlpj.robot import sim
from rlpj.vla import convert, images, openpi_entry, policy_client, rollout
from rlpj.vla.synthetic_episodes import Q_HOME, make_episode

TABLE_Z = -0.023


def _tagged(a):
    return {b"__ndarray__": True, b"data": a.tobytes(), b"dtype": a.dtype.str, b"shape": a.shape}


def test_msgpack_numpy_matches_openpi_layout():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    state = np.linspace(-3, 3, 7).astype(np.float32)
    obs = {"image": img, "wrist_image": img[::-1].copy(), "state": state, "prompt": "put the foam bee in the tape roll"}
    expected = {"image": _tagged(img), "wrist_image": _tagged(obs["wrist_image"]), "state": _tagged(state),
                "prompt": obs["prompt"]}
    assert policy_client.packb(obs) == msgpack.packb(expected, use_bin_type=True)
    assert policy_client.Packer().pack(obs) == policy_client.packb(obs)
    back = policy_client.unpackb(policy_client.packb(obs))
    assert back["prompt"] == obs["prompt"] and back["state"].dtype == np.float32
    np.testing.assert_array_equal(back["image"], img)
    np.testing.assert_array_equal(back["state"], state)
    scalar = {"x": np.float32(1.5), "n": np.int64(3)}
    exp_scalar = {"x": {b"__npgeneric__": True, b"data": 1.5, b"dtype": "<f4"}, "n": {b"__npgeneric__": True, b"data": 3, b"dtype": "<i8"}}
    assert policy_client.packb(scalar) == msgpack.packb(exp_scalar, use_bin_type=True)
    assert policy_client.unpackb(policy_client.packb(scalar))["x"] == np.float32(1.5)
    with pytest.raises(ValueError):
        policy_client.packb({"bad": np.array([1 + 2j])})


def test_unpack_server_response():
    actions = np.random.randn(10, 7).astype(np.float32)
    resp = {"actions": _tagged(actions), "state": _tagged(np.zeros(32, np.float32)),
            "server_timing": {"infer_ms": 12.5}, "policy_timing": {"infer_ms": 11.0}}
    out = policy_client.unpackb(msgpack.packb(resp, use_bin_type=True))
    assert out["actions"].shape == (10, 7) and out["actions"].dtype == np.float32
    np.testing.assert_array_equal(out["actions"], actions)
    assert out["server_timing"]["infer_ms"] == 12.5


def test_resize_with_pad_letterbox():
    rgb = np.full((360, 640, 3), 200, np.uint8)
    out = images.resize_with_pad(rgb, 224, 224)
    rows = np.where(out.max(axis=(1, 2)) > 0)[0]
    assert out.shape == (224, 224, 3) and len(rows) == 126 and rows[0] == 49
    assert images.resize_with_pad(out, 224, 224) is out
    bgr = np.zeros((720, 1280, 3), np.uint8); bgr[..., 0] = 255          # blue in BGR
    m = images.bgr_to_model(bgr)
    assert m.shape == (224, 224, 3) and m.dtype == np.uint8 and m[112, 112].tolist() == [0, 0, 255]


def test_load_episode_state_actions(tmp_path):
    ep = tmp_path / "episodes" / "ep_000000"
    make_episode(ep, n_steps=40, fps=15, seed=3)
    d = convert.load_episode(ep)
    st = np.load(ep / "steps.npz")
    assert d["n"] == 40 and d["fps"] == 15 and d["instruction"] and d["notes"] == []
    assert d["state"].shape == (40, 7) and d["actions"].shape == (40, 7) and d["state"].dtype == np.float32
    assert d["image"].shape == (40, 224, 224, 3) and d["image"].dtype == np.uint8
    assert d["wrist_image"].shape == (40, 224, 224, 3)
    np.testing.assert_allclose(d["state"][:, :6], convert.wrap_pi(np.deg2rad(st["q"])), atol=1e-6)
    assert np.all(d["state"][:, :6] > -np.pi) and np.all(d["state"][:, :6] <= np.pi)
    np.testing.assert_allclose(d["state"][:, 6], st["gripper"] / 100.0, atol=1e-6)
    np.testing.assert_allclose(d["actions"][:, 6], st["gripper_cmd"], atol=1e-6)
    delta = d["actions"][:, :6] - d["state"][:, :6]
    np.testing.assert_allclose(delta, np.deg2rad(convert.wrap180(st["q_cmd"] - st["q"])), atol=1e-6)
    assert np.abs(delta).max() < 0.1                                   # one tick of motion, no 2 pi jumps
    assert d["image"][0, :40].max() == 0 and d["image"][0, 112].max() > 0   # letterbox bars, content in the middle
    row = d["image"][5, 112]                                            # red block (BGR source) is red after BGR -> RGB
    assert row[:, 0].max() > 200 and row[row[:, 0].argmax(), 2] < 60


def test_state_actions_wraps_and_handles_nan():
    q = np.array([[359.0, 20.0, 265.0, 0.0, 302.0, 180.0]])
    state, actions = convert.state_actions(q, [np.nan], [[1.0, 20.0, 266.0, 0.0, 302.0, 180.0]], [np.nan])
    np.testing.assert_allclose(state[0, :6], np.deg2rad([-1, 20, -95, 0, -58, 180]), atol=1e-6)
    assert state[0, 6] == 0.0 and actions[0, 6] == 0.0
    np.testing.assert_allclose(actions[0, :6] - state[0, :6], np.deg2rad([2, 0, 1, 0, 0, 0]), atol=1e-6)


def test_select_episodes(tmp_path):
    for i, outcome in enumerate(["success", "fail", "success"]):
        make_episode(tmp_path / "episodes" / f"ep_{i:06d}", n_steps=8, seed=i, outcome=outcome,
                     mode="eval" if i == 2 else "collect")
    (tmp_path / "episodes" / "ep_000003").mkdir()
    used, excluded, fps = convert.select_episodes(tmp_path)
    assert [u.name for u in used] == ["ep_000000"] and fps == 15
    reasons = dict((str(d).rsplit("/", 1)[1], r) for d, r in excluded)
    assert reasons["ep_000001"].startswith("outcome.final") and reasons["ep_000002"].startswith("mode")
    assert reasons["ep_000003"].startswith("missing")
    used, _, _ = convert.select_episodes(tmp_path, all_outcomes=True, modes=())
    assert len(used) == 3
    assert convert.main(["--run-dir", str(tmp_path), "--name", "x", "--dry-run"]) == 0


def test_features_spec():
    f = convert.features()
    assert f["image"] == {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}
    assert f["state"]["shape"] == (7,) and f["actions"]["dtype"] == "float32"
    assert convert.features(videos=True)["wrist_image"]["dtype"] == "video"


def test_state_from_feedback_matches_convert():
    q = np.array([359.0, 20.0, 265.0, 10.0, 302.0, 0.0])
    s = rollout.state_from_feedback(q, 80.0)
    ref, _ = convert.state_actions(q, [80.0], q, [0.8])
    np.testing.assert_allclose(s, ref[0], atol=1e-6)
    assert s.dtype == np.float32 and rollout.state_from_feedback(q, np.nan)[6] == 0.0


def test_actions_to_targets_unwrap_near_current():
    q_now = np.array([359.0, 20.0, 265.0, 0.0, 302.0, 0.0])
    acts = np.deg2rad([[1.0, 20.0, -95.0, 0.0, -58.0, 0.0], [3.0, 21.0, -94.0, 0.0, -57.0, 0.0]])
    t = rollout.actions_to_targets_deg(acts, q_now)
    np.testing.assert_allclose(t[0], [361.0, 20.0, 265.0, 0.0, 302.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(t[1], [363.0, 21.0, 266.0, 0.0, 303.0, 0.0], atol=1e-9)


def test_make_policy_segment_speed_clamp_and_check():
    kin = Kin()
    dt = 1 / 15
    fast = Q_HOME + np.outer(np.arange(1, 11), [5.0, 0, 0, 0, 0, 0])          # 75 deg/s requested on joint 1
    seg = rollout.make_policy_segment(kin, Q_HOME, fast, dt)
    P = np.array([kin.fk(q)[:3, 3] for q in np.vstack([Q_HOME, fast])])
    v_lin = np.linalg.norm(np.diff(P, axis=0), axis=1).max() / dt
    assert seg.scale == pytest.approx(max(75.0 / (NORMAL["jv"] * 0.92), v_lin / (NORMAL["lin"] * 0.92)), rel=1e-6)
    assert seg.scale > 1.0 and seg.T == pytest.approx(10 * dt * seg.scale)
    assert np.abs(np.diff(seg.q, axis=0)).max() / DT <= NORMAL["jv"] * 1.05
    np.testing.assert_allclose(seg.q[-1], fast[-1]); np.testing.assert_allclose(seg.q[0], Q_HOME)
    q, qd = seg.ref(seg.T / 2)
    assert q.shape == (6,) and qd.shape == (6,) and qd[0] > 0
    ok, rep = sim.check(seg.q, kin, NORMAL, table_z=TABLE_Z)
    assert ok and rep["joint_speed_ok"] and rep["min_tip_above_table"] > 0.1
    assert rep["max_lin_speed"] <= NORMAL["lin"] * 1.15
    slow = Q_HOME + np.outer(np.arange(1, 11), [0.5, 0.3, 0, 0, 0, 0])
    seg2 = rollout.make_policy_segment(kin, Q_HOME, slow, dt)
    assert seg2.scale == 1.0 and seg2.T == pytest.approx(10 * dt)
    assert sim.check(seg2.q, kin, NORMAL, table_z=TABLE_Z)[0]


def test_sim_check_rejects_target_below_table():
    kin = Kin()
    q_low, ep, _ = kin.ik(tool_down_T([0.45, 0.0, TABLE_Z - 0.05], 0.0), Q_HOME)
    assert ep < 1e-3
    targets = Q_HOME + (q_low - Q_HOME) * np.linspace(0.1, 1.0, 10)[:, None]
    seg = rollout.make_policy_segment(kin, Q_HOME, targets, 1 / 15)
    ok, rep = sim.check(seg.q, kin, NORMAL, table_z=TABLE_Z, tip_margin=0.003)
    assert not ok and rep["min_tip_above_table"] < 0.003


def test_gripper_latch():
    g = rollout.GripperLatch(closed=False)
    assert g.update(0.55) is None and g.update(0.8) == 0.8 and g.closed
    assert g.update(0.45) is None and g.update(0.2) == 0.2 and not g.closed
    assert rollout.GripperLatch(closed=True).update(0.9) is None


def test_asset_id_from_checkpoint(tmp_path):
    d = tmp_path / "assets" / "rlpj" / "bee_v1"
    d.mkdir(parents=True)
    (d / "norm_stats.json").write_text(json.dumps({"norm_stats": {}}))
    assert openpi_entry.asset_id_from_checkpoint(tmp_path) == "rlpj/bee_v1"
    assert openpi_entry.asset_id_from_checkpoint(tmp_path / "none") is None
    assert openpi_entry._repo_id("bee_v1") == "rlpj/bee_v1" and openpi_entry._repo_id("a/b") == "a/b"
