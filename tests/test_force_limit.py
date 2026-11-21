"""Executor stops a segment when the estimated tool force changes by more than force_limit (offline fake robot)."""
import numpy as np

from rlpj.robot.executor import Executor


class Seg:
    name, T, speed = "push", 2.0, {"jv": 10.0}
    q = np.zeros((2, 6))

    def ref(self, t):
        return np.zeros(6), np.zeros(6)


class Kin:
    def fk(self, q):
        return np.eye(4)


class Robot:
    def __init__(self, force_at):
        self.t0, self.force_at = None, force_at

    def feedback(self):
        import time
        now = time.time()
        self.t0 = self.t0 or now
        f = [0, 0, 40.0] if now - self.t0 > self.force_at else [0, 0, 3.0]
        return dict(q=np.zeros(6), t=now, state="ARMSTATE_SERVOING_READY", force=np.array(f))

    def send_joint_speeds(self, v):
        pass

    def stop(self):
        pass


def test_force_stop():
    ex = Executor(Robot(force_at=0.5), Kin(), log=lambda *a: None)
    ex.force_limit = 25
    r = ex.run(Seg())
    assert not r["ok"] and "contact force" in r["reason"], r


def test_no_limit_no_stop():
    ex = Executor(Robot(force_at=0.5), Kin(), log=lambda *a: None)
    r = ex.run(Seg())
    assert "contact force" not in r["reason"], r


class Release(Robot):
    """Starts pressed at 45 N, then lets go (a retreat after a push)."""

    def feedback(self):
        fb = super().feedback()
        import time
        fb["force"] = __import__("numpy").array([0, 0, 2.0 if time.time() - self.t0 > self.force_at else 45.0])
        return fb


def test_release_does_not_stop():
    ex = Executor(Release(force_at=0.5), Kin(), log=lambda *a: None)
    ex.force_limit = 25
    r = ex.run(Seg())
    assert "contact force" not in r["reason"], r


def test_force_cap():
    """A program may lower the limit per move but never raise or clear the task's cap."""
    ex = Executor(Robot(force_at=0.5), Kin(), log=lambda *a: None)
    ex.force_cap = 25
    for lim in (None, 100, 25):
        ex.force_limit = lim
        assert ex.limit() == 25
    ex.force_limit = 10
    assert ex.limit() == 10
    ex.force_limit = 100
    r = ex.run(Seg())
    assert "contact force" in r["reason"]


class PreloadRobot(Robot):
    """Starts the move already pressing (40 N), then rises to 75 N: a change of 35 N, a total over the 65 N cap."""
    def feedback(self):
        fb = super().feedback()
        fb["force"] = np.array([0, 0, 75.0 if time.time() - self.t0 > self.force_at else 40.0])
        return fb


def test_force_stop_total_over_cap_with_preload():
    import time as _t
    globals()["time"] = _t
    ex = Executor(PreloadRobot(force_at=0.5), Kin(), log=lambda *a: None)
    ex.force_limit = ex.force_cap = 65
    r = ex.run(Seg())
    assert "total" in r["reason"], r


class EarlyJamRobot(Robot):
    """The total jumps from 40 N to 85 N within the first 0.1 s, inside the baseline window."""
    def feedback(self):
        fb = super().feedback()
        fb["force"] = np.array([0, 0, 85.0 if time.time() - self.t0 > 0.1 else 40.0])
        return fb


def test_force_stop_total_inside_baseline_window():
    import time as _t
    globals()["time"] = _t
    ex = Executor(EarlyJamRobot(force_at=0.0), Kin(), log=lambda *a: None)
    ex.force_limit = ex.force_cap = 65
    r = ex.run(Seg())
    assert "total" in r["reason"] and float(r["reason"].split("at t=")[1][:-1]) < 0.3, r
