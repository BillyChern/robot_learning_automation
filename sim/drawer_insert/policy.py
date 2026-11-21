"""Numpy-only insertion policy for the robot: Policy(path).act(obs) -> (dn m, dt m, dyaw deg[, dz m]). Obs: README.md.
v1 weights (policy.npz): 14 inputs, 3 actions; v2 weights (policy_v2.npz): 18 inputs, 4 actions (adds dz)."""
import numpy as np

H_STEPS = 48
STEP_LO = np.array([-0.005, -0.004, -2.0, -0.003])
STEP_HI = np.array([0.010, 0.004, 2.0, 0.003])
OBS_DIM, OBS_DIM2 = 14, 18
JAM_DN = 0.005                     # force stops within this n of each other count as the same jam (m)


def obs_vector(o):
    """Obs dict -> scaled network input: (14,) or, with the v2 keys z, gap_est, jam_rep and a 4-entry prev, (18,)."""
    f = np.asarray(o["f"], float)
    p = np.asarray(o["prev"], float)
    v = np.r_[o["n"] / 0.1, o["t"] / 0.02, o["yaw"] / 10.0, o["off_est"] / 0.02, o["yaw_est"] / 10.0, f / 30.0,
              o["f_peak"] / 60.0, float(o["stopped"]), p[:3] / np.array([0.01, 0.004, 2.0]), o["k"] / H_STEPS]
    if "z" in o:
        v = np.r_[v, o["z"] / 0.01, o["gap_est"] / 0.01, o["jam_rep"] / 3.0, p[3] / 0.003]
    return np.clip(v, -5.0, 5.0).astype(np.float32)


def step_of(a):
    """Network output in [-1, 1]^k (k = 3 or 4) -> (dn m, dt m, dyaw deg[, dz m])."""
    a = np.clip(np.asarray(a, float), -1, 1)
    k = len(a)
    return STEP_LO[:k] + 0.5 * (a + 1) * (STEP_HI[:k] - STEP_LO[:k])


class Policy:
    """Deterministic MLP (tanh) exported by train.py as .npz (W0, b0, W1, b1, ..., mean head last)."""

    def __init__(self, path):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.v2 = self.layers[0][0].shape[0] == OBS_DIM2

    def mean(self, x):
        for W, b in self.layers[:-1]:
            x = np.tanh(x @ W + b)
        W, b = self.layers[-1]
        return np.tanh(x @ W + b)

    def act(self, obs):
        """One decision: obs dict -> (dn, dt, dyaw[, dz]) in m, m, deg[, m]."""
        return tuple(float(v) for v in step_of(self.mean(obs_vector(obs))))


class Insertion:
    """Robot-side bookkeeping for one insertion: base-frame tool pose and force -> obs -> next tool target (base frame).
    normal/offset/t_centre: the believed slot (task yaml); yaw_straight: tool yaw (deg) at which a drawer held square
    in the jaws points along normal; tool_xy: tool point (between the fingertips, on the front wall's mid-plane).
    v2 weights also need tool_z (start tool height, m) and gap_est (estimated drawer bottom above the slot's front lip
    at the start, m); the height target is then self.z_target_abs after each next()."""

    T_BOUND, YAW_BOUND, N_BACK, N_PAST, Z_BOUND = 0.03, 12.0, 0.03, 0.025, 0.012

    def __init__(self, policy, normal, offset, t_centre, yaw_straight, off_est, yaw_est, tool_xy, tool_yaw, f0,
                 tool_z=None, gap_est=0.0):
        self.pol, self.N = policy, np.asarray(normal, float) / np.linalg.norm(normal)
        self.T = np.array([-self.N[1], self.N[0]])
        self.off, self.tc, self.ys, self.est = float(offset), float(t_centre), float(yaw_straight), (off_est, yaw_est)
        self.f0 = np.asarray(f0, float)
        self.cmd0 = self.cmd = self.to_slot(tool_xy, tool_yaw)
        self.v2 = getattr(policy, "v2", False)
        self.prev, self.f_peak, self.stopped, self.k = np.zeros(4 if self.v2 else 3), 0.0, 0.0, 0
        self.z0, self.z, self.gap_est, self.stops = tool_z, 0.0, float(gap_est), []

    def to_slot(self, xy, yaw):
        xy = np.asarray(xy, float)[:2]
        return np.array([xy @ self.N - self.off, xy @ self.T - self.tc, (yaw - self.ys + 180.0) % 360.0 - 180.0])

    def to_base(self, c):
        return (c[0] + self.off) * self.N + (c[1] + self.tc) * self.T, c[2] + self.ys

    def force(self, f_base):
        """Base-frame force minus the start value -> (f_n, f_t, f_z)."""
        f = np.asarray(f_base, float) - self.f0
        return np.array([f[:2] @ self.N, f[:2] @ self.T, f[2]])

    def next(self, f_base):
        """Decide the next step from the current base-frame force: returns (target xy, target yaw deg)."""
        o = dict(n=self.cmd[0], t=self.cmd[1], yaw=self.cmd[2], off_est=self.est[0], yaw_est=self.est[1],
                 f=self.force(f_base), f_peak=self.f_peak, stopped=self.stopped, prev=self.prev, k=self.k)
        if self.v2:
            o.update(z=self.z, gap_est=self.gap_est, jam_rep=self.jam_rep())
        self.obs = o                                  # log it with the step: the robot's view of the policy input
        self.step = np.array(self.pol.act(o))
        if self.v2:
            self.z_target = float(np.clip(self.z + self.step[3], -self.Z_BOUND, self.Z_BOUND))
            self.z_target_abs = self.z0 + self.z_target
        c = self.cmd + self.step[:3]
        c[0] = min(max(c[0], self.cmd0[0] - self.N_BACK), self.N_PAST)
        c[1] = np.clip(c[1], self.cmd0[1] - self.T_BOUND, self.cmd0[1] + self.T_BOUND)
        c[2] = np.clip(c[2], self.cmd0[2] - self.YAW_BOUND, self.cmd0[2] + self.YAW_BOUND)
        self.target = c
        return self.to_base(c)

    def jam_rep(self):
        """Force stops so far within JAM_DN of the commanded n (the 'same-depth jam' count)."""
        return float(sum(abs(n - self.cmd[0]) < JAM_DN for n in self.stops))

    def done(self, f_peak, stopped, tool_xy=None, tool_yaw=None, tool_z=None):
        """After the move: f_peak = max |f - f0| during it (N); stopped = force stop; then the measured tool pose."""
        self.prev, self.cmd = self.step, self.to_slot(tool_xy, tool_yaw) if stopped else self.target
        if self.v2:
            self.z = tool_z - self.z0 if (stopped and tool_z is not None) else self.z_target
        if stopped:
            self.stops.append(self.cmd[0])
        self.f_peak, self.stopped, self.k = float(f_peak), float(stopped), self.k + 1
        return self.k >= H_STEPS
