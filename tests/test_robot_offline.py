"""Offline tests of rlpj.robot: no robot, no cameras, no network. Fakes replace kortex.Robot and the cameras."""
import importlib.util
import json
import os
import socket
import threading
import time

import cv2
import numpy as np
import pytest

from rlpj.robot import calib, cams, kortex, sim
from rlpj.robot.executor import Executor
from rlpj.robot.home import plan_return
from rlpj.robot.kin import Kin, wrap180
from rlpj.robot.plan import DT, NORMAL, SLOW, Planner, TimedSegment, tool_down_T

REF_KIN = os.path.expanduser("~/Documents/aaron/agentic_robotics/far_skills/cup_down_stacking/kinova_a/lib/seed_2/cds/kin.py")
KIN = Kin()
TZ = calib.table_z()
Q_HOME = KIN.ik(tool_down_T([0.45, -0.05, 0.30], 0.0), np.array([6, 21, -83, 0, -75, 6.]))[0]
Q_LOW = KIN.ik(tool_down_T([0.45, -0.05, TZ + 0.05], 0.0), Q_HOME)[0]


# ---------------------------------------------------------------- fakes
class FakeRobot:
    """Static arm (reads only; every command raises like kortex.Robot in a dry run)."""

    def __init__(self, ip=None, dry=False, q=None, pose_offset=0.0):
        self.q = np.array(Q_HOME if q is None else q, float)
        self.pose_offset, self.dry = pose_offset, True
        self.gripper_cmd = float("nan")
        self.closed = False

    def feedback(self):
        pose = calib.T_to_pose(KIN.fk(self.q)); pose[0] += self.pose_offset
        return dict(q=self.q.copy(), dq=np.zeros(6), pose=pose, gripper=0.9, t=time.time(), state="ARMSTATE_SERVOING_READY")

    def arm_state(self):
        return "ARMSTATE_SERVOING_READY"

    def fk_oracle(self, q):
        return calib.T_to_pose(KIN.fk(q))

    def keepalive(self):
        pass

    def _no(self, *a):
        raise RuntimeError("motion is not allowed (fake)")

    send_joint_speeds = stop = gripper_position = _no

    def close(self):
        self.closed = True


class MovingFakeRobot(FakeRobot):
    """Perfect arm: integrates joint-speed commands and moves the gripper, so the executor loop runs offline."""

    def __init__(self, ip=None, dry=False, q=None):
        super().__init__(q=q)
        self.v, self.t_last, self.g, self.g_target = np.zeros(6), time.time(), 0.9, None

    def _advance(self):
        t = time.time(); dt = t - self.t_last; self.t_last = t
        self.q = self.q + self.v * dt
        if self.g_target is not None:
            self.g += float(np.clip(self.g_target - self.g, -150 * dt, 150 * dt))

    def feedback(self):
        self._advance()
        fb = super().feedback(); fb["gripper"], fb["dq"] = self.g, self.v.copy()
        return fb

    def send_joint_speeds(self, v):
        self._advance(); self.v = np.asarray(v, float); time.sleep(0.004)

    def stop(self):
        self._advance(); self.v = np.zeros(6)

    def gripper_position(self, value):
        self.g_target, self.gripper_cmd = 100.0 * float(value), float(value)


class FakeSideCam:
    def __init__(self, serial=None):
        self.color = np.zeros((720, 1280, 3), np.uint8); self.color[:, :, 1] = np.linspace(0, 255, 1280, dtype=np.uint8)
        self.depth = np.full((720, 1280), 0.8, np.float32)
        self.closed = False

    def latest(self):
        return self.color, self.depth, time.time()

    def depth_median(self, n=5, timeout=3.0):
        return self.depth

    def close(self):
        self.closed = True


class FakeWristCam:
    def __init__(self, ip=None):
        self.color = np.zeros((720, 1280, 3), np.uint8); self.color[:, :, 2] = 180
        self.closed = False

    def latest(self):
        return self.color, time.time()

    def close(self):
        self.closed = True


class FakeSession:
    """What EpisodeRecorder reads."""

    def __init__(self, A):
        self.A, self.robot, self.side, self.wrist = A, FakeRobot(), FakeSideCam(), FakeWristCam()
        self.executor = Executor(self.robot, KIN)
        self.phases, self.phase_idx = ["start"], 0
        self.program, self.mode, self.commit, self.outcome = "fake", "collect", "abc123", "success"

    def feedback(self):
        return self.robot.feedback()


def free_port():
    s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(kortex, "Robot", FakeRobot)
    monkeypatch.setattr(cams, "SideCam", FakeSideCam)
    monkeypatch.setattr(cams, "WristCam", FakeWristCam)
    monkeypatch.delenv("RLPJ_MOTION", raising=False)
    monkeypatch.delenv("RLPJ_RECORD_DIR", raising=False)
    monkeypatch.delenv("RLPJ_ATTEMPT_DIR", raising=False)


# ---------------------------------------------------------------- kinematics / planning
@pytest.mark.skipif(not os.path.exists(REF_KIN), reason="reference seed_2 code not present")
def test_fk_matches_reference():
    spec = importlib.util.spec_from_file_location("ref_kin", REF_KIN)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    rng = np.random.default_rng(0)
    delta = json.load(open(os.path.join(os.path.dirname(os.path.dirname(REF_KIN)), "model", "kin_fit.json")))["delta"]
    for ours, theirs in ((Kin(), ref.Kin()), (Kin(delta), ref.Kin(delta))):
        for _ in range(50):
            q = rng.uniform(-180, 180, 6)
            assert np.allclose(ours.fk(q), theirs.fk(q), atol=1e-9)
            assert np.allclose(ours.frames(q)[3], theirs.frames(q)[3], atol=1e-9)


def test_segment_timing_and_speed_limits():
    pl = Planner(KIN)
    q_b = KIN.ik(tool_down_T([0.50, 0.05, 0.25], 20.0), Q_HOME)[0]
    for speed in (NORMAL, SLOW):
        seg = pl.joint_move("jm", Q_HOME, q_b, speed)
        v = np.abs(np.diff(seg.q, axis=0)) / DT
        assert seg.T >= 0.5 and v.max() <= speed["jv"] * 1.01
        assert np.allclose(seg.q[0], Q_HOME) and np.allclose(seg.q[-1], q_b)
        qr, vr = seg.ref(seg.T / 2)
        assert qr.shape == (6,) and vr.shape == (6,)
        line = pl.cart_line("ln", Q_HOME, tool_down_T([0.45, -0.05, 0.20], 0.0), speed, exact_goal=False)
        P = np.array([KIN.fk(q)[:3, 3] for q in line.q])
        assert (np.linalg.norm(np.diff(P, axis=0), axis=1) / DT).max() <= speed["lin"] * 1.05
        assert np.linalg.norm(P[-1] - [0.45, -0.05, 0.20]) < 1e-3
    ts = TimedSegment("chunk", np.linspace(Q_HOME, Q_HOME + np.array([2, 0, 0, 0, 0, 2.]), 11), 1 / 15, NORMAL, KIN)
    assert abs(ts.T - 10 / 15) < 1e-9 and np.allclose(ts.q[0], Q_HOME) and np.allclose(ts.q[-1], Q_HOME + [2, 0, 0, 0, 0, 2])
    assert abs(ts.ref(ts.T / 2)[0][0] - (Q_HOME[0] + 1.0)) < 1e-6   # linear timing


def test_sim_check_rejects_table_and_speed():
    pl = Planner(KIN)
    bad = pl.cart_line("down_bad", Q_HOME, tool_down_T([0.45, -0.05, TZ - 0.02], 0.0), SLOW, exact_goal=False)
    ok, rep = sim.check(bad.q, KIN, bad.speed)
    assert not ok and rep["min_tip_above_table"] < 0
    good = pl.cart_line("down_ok", Q_HOME, tool_down_T([0.45, -0.05, TZ + 0.06], 0.0), SLOW, exact_goal=False)
    ok, rep = sim.check(good.q, KIN, good.speed)
    assert ok and rep["min_tip_above_table"] > 0.008
    ok, rep = sim.check(good.q, KIN, dict(SLOW, jv=0.5), tip_margin=0.008)   # same path, declared too slow a class
    assert not ok and not rep["joint_speed_ok"]
    ok, rep = sim.check(good.q, KIN, good.speed, max_radius=0.40)
    assert not ok and rep["max_radius"] > 0.40


def test_plan_return_produces_checked_segments():
    pl = Planner(KIN)
    segs, reps, ok = plan_return(pl, Q_LOW, Q_HOME)
    assert ok and [s.name for s in segs] == ["return_lift", "return_home"] and all(r["ok"] for r in reps)
    assert KIN.fk(segs[0].q[-1])[2, 3] > TZ + 0.19 and np.allclose(wrap180(segs[-1].q[-1]), wrap180(Q_HOME), atol=1e-6)
    segs, reps, ok = plan_return(pl, Q_HOME + np.array([5, 0, 0, 0, 0, 0.]), Q_HOME)
    assert ok and [s.name for s in segs] == ["return_home"]


def test_fit_kin_recovers_synthetic_perturbation():
    from rlpj.robot import fit_kin
    rng = np.random.default_rng(3)
    d = np.zeros((7, 6)); d[:, :3] = rng.normal(0, 0.003, (7, 3)); d[:, 3:] = rng.normal(0, 0.005, (7, 3))
    kt = Kin(d)
    pairs = [dict(q=(q := fit_kin.sample_joints(rng, near_work=i % 2 == 0)).tolist(), pose=calib.T_to_pose(kt.fk(q)).tolist(), src="oracle")
             for i in range(60)]
    pairs[0]["src"] = "measured"
    res = fit_kin.fit(pairs, max_nfev=40, log=lambda m: None)
    assert res["nominal"]["pos_mm_median"] > 3 and res["heldout_pos_mm_median"] < 0.2 and res["heldout_pos_mm_max"] < 1.0
    assert np.asarray(res["delta"]).shape == (7, 6)


def test_watchdog_protocol():
    from rlpj.robot.watchdog import Protocol
    p = Protocol()
    assert p.feed(b"HOME 1 2 3 4 5 6\nHB moving 0.80\n") == ["HOME", "HB"]
    assert np.allclose(p.q_home, [1, 2, 3, 4, 5, 6]) and p.state == "moving" and p.grip_frac == 0.8 and not p.bye
    assert not p.starved() and p.starved(now=p.last_hb + 1.0)
    assert p.feed(b"HB id") == [] and p.feed(b"le 0\nBYE\n") == ["HB", "BYE"] and p.state == "idle" and p.bye


def _run_watchdog(tmp_path, lines, close_without_bye):
    """Run watchdog.main in a thread on a pipe with a fake (read-only) robot; returns (exit code, log, robot)."""
    from rlpj.robot import watchdog
    A = tmp_path / "wd"; A.mkdir()
    (A / "summary.json").write_text('{"claimed_outcome": "aborted", "step": "lift"}\n')
    r, w = os.pipe()
    robot = FakeRobot(q=Q_LOW)
    out = {}
    th = threading.Thread(target=lambda: out.update(code=watchdog.main([str(A), "999999999", "dry"], robot=robot, fd=r, kin=KIN)))
    th.start()
    for line in lines:
        os.write(w, line); time.sleep(0.1)
    if close_without_bye:
        os.close(w)
    th.join(20)
    assert not th.is_alive()
    os.close(r)
    if not close_without_bye:
        os.close(w)
    return out.get("code"), (A / "watchdog_log.txt").read_text(), robot, json.load(open(A / "summary.json"))


def test_watchdog_bye_exits_cleanly(tmp_path):
    home = " ".join(f"{v:.4f}" for v in Q_HOME)
    code, log, robot, S = _run_watchdog(tmp_path, [f"HOME {home}\n".encode(), b"HB idle 0.00\n", b"HB moving 0.00\n", b"BYE\n"], False)
    assert code == 0 and "BYE received" in log and "TAKEOVER" not in log and robot.closed and S["claimed_outcome"] == "aborted"


def test_watchdog_takeover_on_pipe_close(tmp_path):
    home = " ".join(f"{v:.4f}" for v in Q_HOME)
    code, log, robot, S = _run_watchdog(tmp_path, [f"HOME {home}\n".encode(), b"HB moving 0.80\n"], True)
    assert code == 0 and "TAKEOVER: main program pipe closed" in log and "return plan ok=True" in log
    assert S["claimed_outcome"] == "aborted" and S["watchdog"]["return_checks_ok"] is True
    assert S["watchdog"]["result"] == "return incomplete" and robot.closed   # fake robot refuses commands, like a dry run


# ---------------------------------------------------------------- recorder
def test_recorder_writes_valid_episode(tmp_path):
    from rlpj.robot.recorder import EpisodeRecorder, update_meta
    A = tmp_path / "attempt"; A.mkdir()
    (A / "summary.json").write_text('{"claimed_outcome": "success"}\n'); (A / "log.txt").write_text("hello\n")
    s = FakeSession(str(A))
    ep = tmp_path / "run_x" / "episodes" / "ep_000007"
    rec = EpisodeRecorder(s, ep, fps=15, h264=False).start()
    rec.snapshot("before")
    time.sleep(0.5)
    s.phases.append("grasp"); s.phase_idx = 1
    s.executor.running = True; s.executor.last = dict(q_ref=s.robot.q + 1.0)
    s.robot.gripper_cmd = 0.8
    time.sleep(0.5)
    s.executor.running = False
    s.phases.append("home"); s.phase_idx = 2
    time.sleep(0.3)
    rec.snapshot("after")
    rec.close(layout_target={"a": [0.4, 0.0]})
    assert rec.errors == 0
    z = np.load(ep / "steps.npz")
    n = len(z["t"])
    assert 14 <= n <= 24, n
    assert set(z.files) == {"t", "q", "dq", "tool_pose", "gripper", "q_cmd", "gripper_cmd", "phase"}
    for k in ("q", "dq", "tool_pose", "q_cmd"):
        assert z[k].shape == (n, 6)
    assert z["gripper"].shape == z["gripper_cmd"].shape == z["phase"].shape == (n,)
    assert np.all(np.diff(z["t"]) > 0)
    grasp = z["phase"] == 1
    assert grasp.sum() >= 4 and np.allclose(z["q_cmd"][grasp], z["q"][grasp] + 1.0) and np.allclose(z["gripper_cmd"][grasp], 0.8)
    before = z["phase"] == 0
    assert np.allclose(z["q_cmd"][before], z["q"][before]) and np.allclose(z["gripper_cmd"][before], 0.009)
    for name in ("side", "wrist"):
        c = cv2.VideoCapture(str(ep / f"{name}.mp4"))
        assert int(c.get(cv2.CAP_PROP_FRAME_COUNT)) == n and (int(c.get(3)), int(c.get(4))) == (640, 360) and round(c.get(cv2.CAP_PROP_FPS)) == 15
    ph = json.load(open(ep / "phases.json"))
    assert [p["name"] for p in ph] == ["start", "grasp", "home"] and ph[0]["i0"] == 0 and ph[-1]["i1"] == n
    assert all(ph[i]["i1"] == ph[i + 1]["i0"] for i in range(len(ph) - 1))
    m = json.load(open(ep / "meta.json"))
    for k in ("episode", "run_id", "task", "instruction", "mode", "program", "robot", "fps", "layout_target", "layout_before", "layout_after",
              "t_start", "t_end", "n_steps", "outcome", "calibration_hashes"):
        assert k in m, k
    assert m["episode"] == "ep_000007" and m["run_id"] == "run_x" and m["n_steps"] == n and m["fps"] == 15
    assert m["program"] == dict(name="fake", commit="abc123", command=m["program"]["command"]) and m["outcome"]["claim"] == "success"
    assert m["layout_target"] == {"a": [0.4, 0.0]} and m["mode"] == "collect"
    m2 = update_meta(ep, outcome=dict(final="success", reason="judge"), layout_after={"a": [0.41, 0.01]})
    assert m2["outcome"]["claim"] == "success" and m2["outcome"]["final"] == "success" and json.load(open(ep / "meta.json"))["layout_after"] == {"a": [0.41, 0.01]}
    for f in ("before_side.jpg", "before_wrist.jpg", "after_side.jpg", "after_wrist.jpg", "summary.json", "log.txt"):
        assert (ep / f).exists(), f
    assert cv2.imread(str(ep / "before_side.jpg")).shape == (360, 640, 3)


# ---------------------------------------------------------------- session (dry run with fakes)
def test_session_dry_run_success(fakes, tmp_path):
    from rlpj.robot.session import RobotSession
    A, ep = tmp_path / "attempt", tmp_path / "ep"
    with RobotSession(program="test", attempt_dir=A, record_dir=ep, view_port=0, exit_on_close=False, kin=KIN) as s:
        assert s.dry and s.wd is None and s.view.port > 0
        assert np.allclose(s.q_home, wrap180(Q_HOME)) and s.phases == ["start"]
        s.phase("perceive")
        s.view.set_scene(objects=[dict(xy=[0.45, -0.05], z_top=TZ + 0.04, r=0.03, label="cup")], targets=[("grasp", [0.45, -0.05, TZ + 0.02])])
        s.decide("go")
        s.phase("approach")
        s.move_joint("pre", KIN.ik(tool_down_T([0.45, -0.05, 0.20], 0.0), Q_HOME)[0], "normal")
        s.move_line("descend", tool_down_T([0.45, -0.05, TZ + 0.06], 0.0), "slow")
        fb = s.feedback()
        assert fb["sim"] and abs(fb["pose"][2] - (TZ + 0.06)) < 1e-3
        g = s.gripper(0.8, "grasp")
        assert g == 80.0 and s.feedback()["gripper"] == 80.0
        s.phase("chunk")
        r = s.follow("policy", np.linspace(fb["q"], fb["q"] + np.array([0, -2, 0, 0, 0, 0.]), 10), 1 / 15)
        assert r["ok"] and r["dry"]
        s.phase("home")
        s.home()
        assert np.allclose(wrap180(s.feedback()["q"]), wrap180(Q_HOME), atol=1e-6)
        s.summary(verify=dict(ok=True))
        s.require(True, "fine")
        s.claim("success")
        time.sleep(0.3)
    assert s.exit_code == 0
    S = json.load(open(A / "summary.json"))
    assert S["claimed_outcome"] == "success" and S["step"] == "home" and S["verify"] == {"ok": True} and S["exit_code"] == 0
    assert [x[0] for x in S["segments"]] == ["pre", "descend", "policy", "home"] and S["decisions"] == ["go"]
    plans = sorted(os.listdir(A / "plans"))
    assert "01_pre.npz" in plans and "02_descend.json" in plans and "04_home.npz" in plans
    assert any(p.endswith(".mp4") for p in plans)   # replay videos (MuJoCo EGL)
    assert (A / "dashboard.mp4").exists() and (A / "log.txt").exists()
    assert json.load(open(A / "plans" / "02_descend.json"))["check"]["ok"] is True
    z = np.load(ep / "steps.npz")
    ph = json.load(open(ep / "phases.json"))
    assert [p["name"] for p in ph] == ["start", "perceive", "approach", "chunk", "home"] and len(z["t"]) == ph[-1]["i1"] > 0
    m = json.load(open(ep / "meta.json"))
    assert m["outcome"]["claim"] == "success" and m["program"]["name"] == "test" and m["mode"] == "collect"
    assert (ep / "summary.json").exists() and (ep / "before_side.jpg").exists() and (ep / "after_wrist.jpg").exists()
    assert s.robot is None and s.view is None   # closed


def test_session_abort_and_exception_codes(fakes, tmp_path):
    from rlpj.robot.session import Abort, RobotSession
    A = tmp_path / "a1"
    with RobotSession(program="test", attempt_dir=A, view_port=0, exit_on_close=False, kin=KIN) as s:
        s.phase("check")
        s.require(1 > 2, "nothing on the table")
        raise AssertionError("not reached")
    assert s.exit_code == 2
    S = json.load(open(A / "summary.json"))
    assert S["claimed_outcome"] == "fail" and S["stop"]["kind"] == "failed check" and S["stop"]["step"] == "check"
    assert S["first_failed_check"] == "nothing on the table" and (A / "stop_side.jpg").exists()

    A = tmp_path / "a2"
    with RobotSession(program="test", attempt_dir=A, view_port=0, exit_on_close=False, kin=KIN) as s:
        s.move_line("bad", tool_down_T([0.45, -0.05, TZ - 0.02], 0.0), "slow")   # model check must refuse
    assert s.exit_code == 2 and "model check failed" in json.load(open(A / "summary.json"))["stop"]["why"]

    A = tmp_path / "a3"
    with RobotSession(program="test", attempt_dir=A, view_port=0, exit_on_close=False, kin=KIN) as s:
        raise ValueError("bug")
    assert s.exit_code == 3 and json.load(open(A / "summary.json"))["stop"]["kind"] == "exception"

    A = tmp_path / "a4"
    with RobotSession(program="test", attempt_dir=A, view_port=0, exit_on_close=False, kin=KIN) as s:
        pass   # no claim -> fail
    assert s.exit_code == 2 and json.load(open(A / "summary.json"))["claimed_outcome"] == "fail"

    A = tmp_path / "a5"
    with pytest.raises(SystemExit) as e:
        with RobotSession(program="test", attempt_dir=A, view_port=0, kin=KIN) as s:
            s.claim("success")
    assert e.value.code == 0 and s.exit_code == 0


def test_session_start_checks(fakes, tmp_path, monkeypatch):
    from rlpj.robot.session import Abort, RobotSession
    monkeypatch.setattr(kortex, "Robot", lambda ip=None, dry=False: FakeRobot(pose_offset=0.010))
    with pytest.raises(Abort):
        with RobotSession(program="test", attempt_dir=tmp_path / "fk", view_port=0, exit_on_close=False, kin=KIN):
            raise AssertionError("not reached")
    S = json.load(open(tmp_path / "fk" / "summary.json"))
    assert S["claimed_outcome"] == "fail" and "FK disagrees" in S["stop"]["why"] and 9 < S["fk_check_mm"] < 11


def test_session_stop_signal(fakes, tmp_path):
    import signal
    from rlpj.robot.session import RobotSession
    A = tmp_path / "stop"
    with RobotSession(program="test", attempt_dir=A, view_port=0, exit_on_close=False, kin=KIN) as s:
        s.phase("perceive")
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        assert s.stop["flag"] and s.stop["sig"] == "SIGINT"
        s.phase("approach")   # raises StopRequested
        raise AssertionError("not reached")
    assert s.exit_code == 130
    S = json.load(open(A / "summary.json"))
    assert S["claimed_outcome"] == "aborted" and S["stop"]["kind"] == "operator stop" and S["stop"]["signal"] == "SIGINT"
    assert signal.getsignal(signal.SIGINT) is not s._on_signal   # handlers restored


def test_session_motion_path_with_moving_fake(fakes, tmp_path, monkeypatch):
    """RLPJ_MOTION=1 against a fake arm that integrates speed commands: tracking, gripper, Abort -> checked return home."""
    from rlpj.robot.session import RobotSession
    monkeypatch.setenv("RLPJ_MOTION", "1")
    monkeypatch.setattr(kortex, "Robot", MovingFakeRobot)
    A, ep = tmp_path / "motion", tmp_path / "ep"
    with RobotSession(program="test", attempt_dir=A, record_dir=ep, view_port=0, watchdog=False, exit_on_close=False, kin=KIN) as s:
        assert not s.dry and s.wd is None
        s.phase("move")
        s.move_joint("small", Q_HOME + np.array([3, 0, 0, 0, 0, 3.]), "normal")
        assert np.abs(s.feedback()["q"] - (Q_HOME + [3, 0, 0, 0, 0, 3])).max() < 0.3 and s.executor.running is False
        g = s.gripper(0.5, "half")
        assert abs(g - 50) < 3 and s.robot.gripper_cmd == 0.5
        s.require(False, "stop here")
    assert s.exit_code == 2
    S = json.load(open(A / "summary.json"))
    assert S["segments"][0][0] == "small" and S["segments"][0][1] is True and S["segments"][0][2] < 1.0   # name, ok, max dev deg
    assert S["segments"][-1][0] == "return_home" and S["segments"][-1][1] is True and S["return_home"] == [["return_home", True]]
    assert S["stop"]["kind"] == "failed check" and "q_plan" in S["stop"] and S["stop"]["gripper"] == pytest.approx(50, abs=3)
    z = np.load(ep / "steps.npz")
    assert len(z["t"]) >= 15 and z["gripper_cmd"][-1] == 0.5 and np.isclose(z["gripper_cmd"][0], 0.009)
    moving = np.abs(z["dq"]).max(1) > 0.1
    assert moving.sum() >= 3 and np.all(np.isfinite(z["q_cmd"]))
    ph = json.load(open(ep / "phases.json"))
    assert [p["name"] for p in ph] == ["start", "move", "return_home"]


def test_check_offline():
    from rlpj.robot import check
    res = check.run(robot=False, cameras=False, port=free_port())
    names = [c["name"] for c in res["checks"]]
    assert names[0] == "calibration files" and names[1] == "kin fit" and names[-1].startswith("port ")
    assert res["checks"][-1]["status"] == "PASS" and all(c["status"] in ("PASS", "FAIL", "WARN") for c in res["checks"])
    srv = socket.socket(); srv.bind(("", 0)); srv.listen(1)
    try:
        assert not check.port_free(srv.getsockname()[1])
    finally:
        srv.close()


def test_planner_survives_a_refusing_fk_oracle():
    """Kortex sometimes rejects ComputeForwardKinematics for a joint set: plan with the fitted model instead of crashing."""
    import numpy as np
    from rlpj.robot.kin import Kin
    from rlpj.robot.plan import Planner, SLOW

    def refuse(q):
        raise RuntimeError("KServerException('Sending request for forward kinematics calculations failed: Failed')")

    kin, logs = Kin(), []
    q0 = np.array([0, 15, 230, 0, 55, 90.0])
    goal = kin.fk(q0)
    goal[2, 3] -= 0.04
    seg = Planner(kin, oracle=refuse, log=logs.append).cart_line("down", q0, goal, SLOW)
    assert np.linalg.norm(kin.fk(seg.q[-1])[:3, 3] - goal[:3, 3]) < 1e-3 and any("refused" in m for m in logs)


def test_session_refuses_to_let_go_outside_the_release_zone(fakes, tmp_path):
    from rlpj.robot.session import RobotSession
    zone = dict(reading=[60, 92], x=[0.36, 0.60], y=[-0.22, 0.08], max_height=0.20)
    A = tmp_path / "r1"
    with RobotSession(program="test", attempt_dir=A, view_port=0, exit_on_close=False, kin=KIN) as s:
        s.release_zone = zone
        s.move_joint("pre", KIN.ik(tool_down_T([0.45, 0.0, 0.20], 0.0), Q_HOME)[0], "normal")
        s.move_line("low", tool_down_T([0.45, 0.0, TZ + 0.10], 0.0), "slow")
        s.gripper(1.0, "grasp")
        s.grip_reading = 85.0                                   # dry run: as if the bee were in the jaws
        s.gripper(0.0, "release inside the zone")               # allowed
        s.gripper(1.0, "grasp again")
        s.grip_reading = 94.3                                   # the bowl rim: not a held bee, any place is fine
        s.gripper(0.3, "release the rim")
        s.gripper(1.0, "grasp the bee")
        s.grip_reading = 85.0
        s.move_line("edge", tool_down_T([0.45, 0.12, TZ + 0.10], 0.0), "slow")
        s.gripper(0.0, "release near the curtain-side edge")    # refused: the bee stays held
        raise AssertionError("not reached")
    S = json.load(open(A / "summary.json"))
    assert s.exit_code == 2 and "release refused" in S["stop"]["why"] and "(0.450, 0.120)" in S["stop"]["why"]
    assert S["gripper_final"] == {"cmd": 1.0, "reading": 85.0}
