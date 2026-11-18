"""Executor ignores one bad joint reading but stops on a deviation that persists (offline fake robot)."""
import time

import numpy as np

from rlpj.robot.executor import Executor
from test_force_limit import Kin, Seg


class Robot:
    def __init__(self, bad):
        self.k, self.bad, self.sent = 0, bad, []

    def feedback(self):
        self.k += 1
        q = np.zeros(6)
        if self.k in self.bad:
            q[4] = 47.0
        return dict(q=q, t=time.time(), state="ARMSTATE_SERVOING_READY", force=np.zeros(3))

    def send_joint_speeds(self, v):
        self.sent.append(np.asarray(v))

    def stop(self):
        pass


def test_one_bad_sample_is_ignored():
    rob = Robot(bad={5})
    r = Executor(rob, Kin(), log=lambda *a: None).run(Seg())
    assert r["ok"] and r["feedback_glitches"] == 1, r
    assert max(np.abs(v).max() for v in rob.sent) < 1.0      # no correction toward the bad reading


def test_persistent_deviation_stops():
    r = Executor(Robot(bad={5, 6, 7}), Kin(), log=lambda *a: None).run(Seg())
    assert not r["ok"] and "tracking deviation" in r["reason"], r
