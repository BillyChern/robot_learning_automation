"""rlpj.vla.rollout with eased policy chunks.

The stock rollout runs each chunk as a piecewise-linear joint path that starts from rest with a velocity step; the
tracker then lags > 1 deg at t~0.2 s on fast chunks and the session aborts (eval 03: 3 of 12 rollouts). Here every
chunk is time-warped with a trapezoidal speed profile along the path (ramp up over the first RAMP s, down over the last
RAMP s); peak speeds stay those of the stock segment, so the speed caps still hold. Everything else (inference, model
check, gripper latch, session limits, claim "fail") is the stock rollout, which this program calls unchanged.
Same arguments as rlpj.vla.rollout."""
import sys

import numpy as np

import rlpj.vla.rollout as R
from rlpj.robot.plan import DT

RAMP = 0.15   # s (adds ~20% per 0.67 s chunk; time limit matters)


class EasedSegment:
    def __init__(self, base, ramp=RAMP):
        self.name, self.speed, self.kind = base.name, base.speed, base.kind
        self.path, self.scale = base.path, base.scale
        T0 = float(base.T)
        ta = min(ramp, T0) if T0 > 0 else 0.0
        self.T = T0 + ta
        n = max(int(np.ceil(self.T / DT)), 1)
        self.t = np.arange(n + 1) * DT
        self.t[-1] = self.T
        u = self._warp(self.t, T0, ta)
        self.q = np.stack([np.interp(u, base.t, base.q[:, j]) for j in range(6)], 1)
        self.qd = np.gradient(self.q, DT, axis=0) if len(self.t) > 1 else np.zeros_like(self.q)
        self.tool = base.tool

    @staticmethod
    def _warp(tau, T0, ta):
        if ta <= 0:
            return np.zeros_like(tau)
        T = T0 + ta
        u = np.where(tau < ta, tau ** 2 / (2 * ta), tau - ta / 2)
        return np.clip(np.where(tau > T - ta, T0 - (T - tau) ** 2 / (2 * ta), u), 0, T0)

    def ref(self, t):
        if t <= 0:
            return self.q[0], np.zeros(6)
        if t >= self.T:
            return self.q[-1], np.zeros(6)
        k = t / DT
        i = min(int(k), len(self.q) - 2)
        f = k - i
        return self.q[i] * (1 - f) + self.q[i + 1] * f, self.qd[i] * (1 - f) + self.qd[i + 1] * f


_stock = R.make_policy_segment


def make_eased_segment(kin, q0_deg, targets_deg, dt, speed=R.NORMAL, name="policy"):
    return EasedSegment(_stock(kin, q0_deg, targets_deg, dt, speed, name))


R.make_policy_segment = make_eased_segment

if __name__ == "__main__":
    sys.exit(R.main())
