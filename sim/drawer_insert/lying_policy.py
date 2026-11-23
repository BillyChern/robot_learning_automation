"""Numpy-only policy for the lying cabinet (vertical insertion): Policy(path).act(obs) -> (dx m, dy m, dyaw deg, dz m);
Descent does the robot-side bookkeeping in the base frame. Frames, obs and signs: README.md "Lying cabinet"."""
import numpy as np

from .policy import Policy as Mlp

H_STEPS = 60
STEP_LO = np.array([-0.004, -0.004, -2.0, -0.008])     # dx, dy (m, opening frame), dyaw (deg), dz (m, - = down)
STEP_HI = np.array([0.004, 0.004, 2.0, 0.004])
OBS_DIM = 19
XY_BOUND, YAW_BOUND, Z_LOW, Z_HIGH = 0.015, 8.0, -0.015, 0.020   # command bounds (see Descent.next)


def obs_vector(o):
    """Obs dict -> scaled network input (19,)."""
    v = np.r_[o["x"] / 0.01, o["y"] / 0.01, o["yaw"] / 5.0, o["z"] / 0.05, np.asarray(o["off"]) / 0.01,
              o["yaw_est"] / 5.0, np.asarray(o["tilt"]) / 5.0, np.asarray(o["f"]) / 30.0, o["f_peak"] / 60.0,
              float(o["stopped"]), np.asarray(o["prev"]) / np.array([0.004, 0.004, 2.0, 0.008]), o["k"] / H_STEPS]
    return np.clip(v, -5.0, 5.0).astype(np.float32)


def step_of(a):
    """Network output in [-1, 1]^4 -> (dx m, dy m, dyaw deg, dz m)."""
    a = np.clip(np.asarray(a, float), -1, 1)
    return STEP_LO + 0.5 * (a + 1) * (STEP_HI - STEP_LO)


def action_of(d):
    """(dx, dy, dyaw, dz) -> network output in [-1, 1]^4 (inverse of step_of)."""
    return np.clip(2 * (np.asarray(d, float) - STEP_LO) / (STEP_HI - STEP_LO) - 1, -1, 1)


class Policy(Mlp):
    """Deterministic MLP exported by train.py --task lying (weights/policy_lying.npz)."""

    def act(self, obs):
        """One decision: obs dict -> (dx, dy, dyaw, dz) in m, m, deg, m."""
        return tuple(float(v) for v in step_of(self.mean(obs_vector(obs))))


def rot(a_deg):
    c, s = np.cos(np.radians(a_deg)), np.sin(np.radians(a_deg))
    return np.array([[c, -s], [s, c]])


def bottom_offset(est, length):
    """Estimated drawer bottom-face centre minus the tool point, horizontal, tool frame (m)."""
    tx, ty = np.radians(est["tilt"][0]), np.radians(est["tilt"][1])
    v = np.array([np.sin(tx) * np.cos(ty), np.sin(ty)]) * length       # Ry(-tx) Rx(ty) (0, 0, -L)
    return np.asarray(est["off"], float) + rot(est["yaw"]) @ v


class Descent:
    """Robot-side bookkeeping for one vertical insertion: base-frame tool pose and force -> obs -> next tool target.
    centre: believed opening centre (base xy); axis_deg: the opening's x axis (cabinet yaw_deg); rim_z: rim height
    (base z); yaw_straight: tool yaw (deg, base) at which the tool's x axis lies along the opening's x axis;
    est: dict(off=(x, y) m, yaw=deg, tilt=(x, y) deg, hold=m) from the wrist look-down, in that tool frame."""

    def __init__(self, policy, centre, axis_deg, rim_z, yaw_straight, est, tool_xyz, tool_yaw, f0):
        self.pol, self.c, self.axis, self.ys = policy, np.asarray(centre, float), float(axis_deg), float(yaw_straight)
        self.est, self.z_flush = est, float(rim_z) + float(est["hold"])
        self.R = rot(self.axis)
        self.f0 = np.asarray(f0, float)
        self.cmd0 = self.cmd = self.to_open(tool_xyz, tool_yaw)
        self.prev, self.f_peak, self.stopped, self.k = np.zeros(4), 0.0, 0.0, 0

    def to_open(self, xyz, yaw):
        """Base tool pose -> (x, y, yaw, z) in the believed opening frame (z = tool height above the flush pose)."""
        xyz = np.asarray(xyz, float)
        xy = self.R.T @ (xyz[:2] - self.c)
        return np.array([xy[0], xy[1], (yaw - self.ys + 180.0) % 360.0 - 180.0, xyz[2] - self.z_flush])

    def to_base(self, c):
        """(x, y, yaw, z) opening frame -> (tool xyz base, tool yaw deg base)."""
        return np.r_[self.c + self.R @ c[:2], c[3] + self.z_flush], c[2] + self.ys

    def force(self, f_base):
        """Base-frame force on the tool minus its value at the start of the descent -> opening frame (N)."""
        f = np.asarray(f_base, float) - self.f0
        return np.r_[self.R.T @ f[:2], f[2]]

    def next(self, f_base):
        """Decide the next step from the current base-frame force: returns (target xyz, target yaw deg), base frame."""
        e = self.est
        o = dict(x=self.cmd[0], y=self.cmd[1], yaw=self.cmd[2], z=self.cmd[3], off=e["off"], yaw_est=e["yaw"],
                 tilt=e["tilt"], f=self.force(f_base), f_peak=self.f_peak, stopped=self.stopped, prev=self.prev, k=self.k)
        self.obs = o                                  # log it with the step: the robot's view of the policy input
        self.step = np.array(self.pol.act(o))
        c = self.cmd + self.step
        c[:2] = np.clip(c[:2], self.cmd0[:2] - XY_BOUND, self.cmd0[:2] + XY_BOUND)
        c[2] = np.clip(c[2], self.cmd0[2] - YAW_BOUND, self.cmd0[2] + YAW_BOUND)
        c[3] = np.clip(c[3], Z_LOW, self.cmd0[3] + Z_HIGH)
        self.target = c
        return self.to_base(c)

    def done(self, f_peak, stopped, tool_xyz=None, tool_yaw=None):
        """After the move: f_peak = max |f - f0| during it (N); stopped = force stop; then the measured tool pose."""
        self.prev, self.cmd = self.step, self.to_open(tool_xyz, tool_yaw) if stopped else self.target
        self.f_peak, self.stopped, self.k = float(f_peak), float(stopped), self.k + 1
        return self.k >= H_STEPS


Insertion = Descent              # same name as the standing-cabinet helper in policy.py


def start_pose(centre, axis_deg, rim_z, yaw_straight, est, length, gap=0.015):
    """Scripted start (as in the sim): tool yaw that squares the estimated drawer with the opening, tool xy that puts the
    estimated bottom-face centre over the believed opening centre, tool z with the bottom gap above the rim.
    Returns (tool xyz base, tool yaw deg base)."""
    yaw = -float(est["yaw"])
    xy = -rot(yaw) @ bottom_offset(est, length)
    c = np.asarray(centre, float) + rot(axis_deg) @ xy
    return np.r_[c, float(rim_z) + gap + length + float(est["hold"])], yaw + float(yaw_straight)
