"""Drawer insertion env (MuJoCo, CPU): a held drawer pushed into the cabinet's bottom slot by small tool steps.

Frames: the policy works in the BELIEVED slot frame (from the task yaml); the sim world is the TRUE slot frame.
Obs/action definitions: README.md."""
import mujoco
import numpy as np

from . import params as P
from .scene import xml

H_STEPS = 48                       # decisions per episode
STEP_LO = np.array([-0.005, -0.004, -2.0])
STEP_HI = np.array([0.010, 0.004, 2.0])
T_BOUND, YAW_BOUND = 0.03, 12.0    # commanded t (m) and yaw (deg) bounds around the start pose
N_PAST = 0.02                      # tool may go this far past the believed flush pose (m)
FLUSH_TOL, F_OK = 0.010, 40.0      # success: front within 1 cm of the front plane, force below 40 N
SAMPLE = 25                        # substeps between force samples (40 Hz, like the arm's feedback)
OBS_KEYS = ("n", "t", "yaw", "off_est", "yaw_est", "f", "f_peak", "stopped", "prev", "k")


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def step_of(a):
    """Policy output in [-1, 1]^3 -> (dn m, dt m, dyaw deg)."""
    a = np.clip(np.asarray(a, float), -1, 1)
    return STEP_LO + 0.5 * (a + 1) * (STEP_HI - STEP_LO)


def action_of(d):
    """(dn, dt, dyaw) -> policy output in [-1, 1]^3 (inverse of step_of)."""
    return np.clip(2 * (np.asarray(d, float) - STEP_LO) / (STEP_HI - STEP_LO) - 1, -1, 1)


class DrawerEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.m = self.d = None

    # ------------------------------------------------------------ frames
    def to_true(self, nb, tb, yawb):
        d = np.radians(self.p["slot_dyaw"])
        xy = rot(-d) @ (np.array([nb, tb]) - [self.p["slot_dn"], self.p["slot_dt"]])
        return xy[0], xy[1], np.radians(yawb) - d

    def to_believed(self, x, y, th):
        d = np.radians(self.p["slot_dyaw"])
        nb, tb = rot(d) @ np.array([x, y]) + [self.p["slot_dn"], self.p["slot_dt"]]
        return nb, tb, float(np.degrees(th + d))

    # ------------------------------------------------------------ state readout
    def site_xy(self, name):
        return self.d.site(name).xpos[:2].copy()

    def front_err(self):
        """Front corners' n (true) -> worst distance from the front plane (m, + = protruding)."""
        n = [self.site_xy("front_l")[0], self.site_xy("front_r")[0]]
        return max(abs(n[0]), abs(n[1])), -0.5 * (n[0] + n[1])

    def depth(self):
        return float(np.clip(self.site_xy("back_c")[0], -0.05, self.p["L"]))

    def drawer_pose(self):
        """True drawer centre-line t at the front wall and yaw (deg) vs the slot normal."""
        xm = self.d.body("drawer").xmat.reshape(3, 3)
        c = 0.5 * (self.site_xy("front_l") + self.site_xy("front_r"))
        return float(c[1]), float(np.degrees(np.arctan2(xm[1, 0], xm[0, 0])))

    def contact_force(self):
        """Sum of environment forces on the drawer (world frame, N) = external force on the tool, quasi-static."""
        f, out = np.zeros(6), np.zeros(3)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if self.bid not in (b1, b2) or b1 == b2:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f)
            fw = c.frame.reshape(3, 3).T @ f[:3]
            out += fw if b2 == self.bid else -fw
        return out

    def measured(self, f_true):
        """Force estimate in the believed frame: true + residual bias + noise."""
        d = np.radians(self.p["slot_dyaw"])
        fb = np.r_[rot(d) @ f_true[:2], f_true[2]]
        return fb + self.f_bias + self.rng.normal(0, self.p["f_noise"], 3)

    # ------------------------------------------------------------ episode
    def reset(self, p=None, preroll=None, k_pre=0):
        """New episode with params p (sampled if None); optionally run a scripted controller for k_pre steps first
        (reset-to-state: partly inserted, misaligned or jammed starts)."""
        self.p = p = dict(p) if p is not None else P.sample(self.rng)
        self.m = mujoco.MjModel.from_xml_string(xml(p))
        self.d = mujoco.MjData(self.m)
        self.bid = self.m.body("drawer").id
        self.jx, self.jy, self.jyaw = (self.m.joint(n).qposadr[0] for n in ("hx", "hy", "hyaw"))
        self.f_bias = self.rng.uniform(*P.RANGES["f_bias"], 3)
        yaw0, t0 = -p["est_yaw"], -p["est_off"]
        x, y, th = self.to_true(0.0, t0, yaw0)
        self.set_hand(x, y, th)
        nb_back = self.site_xy("back_c")[0]
        d = np.radians(p["slot_dyaw"])
        n0 = (-p["back_gap"] - nb_back) / np.cos(d)
        x, y, th = self.to_true(n0, t0, yaw0)
        self.set_hand(x, y, th)
        self.cmd0 = np.array([n0, t0, yaw0])
        self.cmd = self.cmd0.copy()
        self.n_max = 0.5 * p["wall"] + N_PAST
        self.nsub = int(round(p["dt_dec"] / self.m.opt.timestep))
        for _ in range(30):
            mujoco.mj_step(self.m, self.d)
        self.k, self.prev, self.f_last, self.f_peak, self.stopped = 0, np.zeros(3), np.zeros(3), 0.0, 0.0
        self.e_t, self.e_yaw = self.drawer_pose()
        self.dep = self.depth()
        self.max_f = 0.0
        self.n_stops = 0
        obs = self.obs()
        if preroll is not None:
            for _ in range(k_pre):
                obs, _, _, _ = self.step(action_of(preroll(obs)))
            self.max_f, self.n_stops = 0.0, 0
        return obs

    def set_hand(self, x, y, th):
        self.d.qpos[[self.jx, self.jy, self.jyaw]] = x, y, th
        self.d.ctrl[:] = x, y, th
        mujoco.mj_forward(self.m, self.d)

    def obs(self):
        return dict(n=self.cmd[0], t=self.cmd[1], yaw=self.cmd[2], off_est=self.p["est_off"], yaw_est=self.p["est_yaw"],
                    f=self.f_last.copy(), f_peak=self.f_peak, stopped=self.stopped, prev=self.prev.copy(), k=self.k)

    def step(self, a):
        """One decision: a in [-1, 1]^3 -> (obs, reward, done, info)."""
        step = step_of(a)
        old = self.cmd.copy()
        new = old + step
        new[0] = np.clip(new[0], self.cmd0[0] - 0.03, self.n_max)
        new[1] = np.clip(new[1], self.cmd0[1] - T_BOUND, self.cmd0[1] + T_BOUND)
        new[2] = np.clip(new[2], self.cmd0[2] - YAW_BOUND, self.cmd0[2] + YAW_BOUND)
        n_move = max(1, int(0.8 * self.nsub))
        peak, peak_true, stopped, held = 0.0, 0.0, False, None
        f_true = np.zeros(3)
        for i in range(self.nsub):
            if held is None:
                c = old + (new - old) * min(1.0, (i + 1) / n_move)
                self.d.ctrl[:] = self.to_true(*c)
            mujoco.mj_step(self.m, self.d)
            if (i + 1) % SAMPLE == 0:
                f_true = self.contact_force()
                fm = self.measured(f_true)
                peak, peak_true = max(peak, np.linalg.norm(fm)), max(peak_true, np.linalg.norm(f_true))
                if held is None and np.linalg.norm(fm) > self.p["f_stop"]:
                    q = self.d.qpos
                    held = np.array(self.to_believed(q[self.jx], q[self.jy], q[self.jyaw]))
                    self.d.ctrl[:] = q[self.jx], q[self.jy], q[self.jyaw]
                    stopped = True
                self.f_last = fm
        self.cmd = new if held is None else held
        self.prev, self.f_peak, self.stopped, self.k = step, peak, float(stopped), self.k + 1
        self.n_stops += int(stopped)
        self.max_f = max(self.max_f, peak_true)
        err, protr = self.front_err()
        f_end = float(np.linalg.norm(f_true))
        succ = err <= FLUSH_TOL and f_end < F_OK
        dep = self.depth()
        r = 3.0 * (dep - self.dep) - 0.02 * max(0.0, peak_true - F_OK) / 10.0 + 0.1 * succ
        self.dep = dep
        done = self.k >= H_STEPS
        if done:
            r += 1.0 * succ
        info = dict(success=bool(succ), front_err=err, protrusion=protr, f_end=f_end)
        if done:
            info.update(max_f=self.max_f, n_stops=self.n_stops, e_t=self.e_t, e_yaw=self.e_yaw, depth=dep)
        return self.obs(), r, done, info
