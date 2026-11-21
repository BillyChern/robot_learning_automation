"""Drawer insertion env (MuJoCo, CPU): a held drawer pushed into the cabinet's bottom slot by small tool steps.

Frames: the policy works in the BELIEVED slot frame (from the task yaml); the sim world is the TRUE slot frame.
Obs/action definitions: README.md."""
import mujoco
import numpy as np

from . import params as P
from .baselines import preroll
from .policy import H_STEPS, JAM_DN, STEP_HI, STEP_LO, step_of
from .scene import xml

T_BOUND, YAW_BOUND, Z_BOUND = 0.03, 12.0, 0.012   # commanded t (m), yaw (deg), z (m, v2) bounds around the start
N_PAST = 0.02                      # tool may go this far past the believed flush pose (m)
FLUSH_TOL, F_OK = 0.010, 40.0      # success: front within 1 cm of the front plane, force below 40 N
F_PEN = 0.05                       # reward per 10 N of a step's peak force above F_OK (run r1: 0.02)
SAMPLE = 25                        # substeps between force samples (40 Hz, like the arm's feedback)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def action_of(d):
    """(dn, dt, dyaw[, dz]) -> policy output in [-1, 1]^k (inverse of step_of)."""
    d = np.asarray(d, float); k = len(d)
    return np.clip(2 * (d - STEP_LO[:k]) / (STEP_HI[:k] - STEP_LO[:k]) - 1, -1, 1)


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
    def reset(self, p=None, kind=None, k_pre=0):
        """New episode with params p (sampled if None); optionally run the scripted pre-roll `kind` for k_pre steps
        first (reset-to-state: partly inserted, misaligned or jammed starts; baselines.preroll)."""
        self.p = p = dict(p) if p is not None else P.sample(self.rng)
        self.m = mujoco.MjModel.from_xml_string(xml(p))
        self.d = mujoco.MjData(self.m)
        self.bid = self.m.body("drawer").id
        self.jx, self.jy, self.jyaw = (self.m.joint(n).qposadr[0] for n in ("hx", "hy", "hyaw"))
        self.v2 = bool(p.get("v2"))
        self.jz = self.m.joint("hz").qposadr[0] if self.v2 else None
        self.f_bias = self.rng.uniform(*P.RANGES["f_bias"], 3)
        yaw0, t0 = -p["est_yaw"], -p["est_off"]
        x, y, th = self.to_true(0.0, t0, yaw0)
        self.set_hand(x, y, th)
        nb_back = self.site_xy("back_c")[0]
        d = np.radians(p["slot_dyaw"])
        n0 = (-p["back_gap"] - nb_back) / np.cos(d)
        x, y, th = self.to_true(n0, t0, yaw0)
        self.set_hand(x, y, th)
        self.cmd0 = np.array([n0, t0, yaw0] + ([0.0] if self.v2 else []))
        self.cmd = self.cmd0.copy()
        self.n_max = 0.5 * p["wall"] + N_PAST
        self.nsub = int(round(p["dt_dec"] / self.m.opt.timestep))
        for _ in range(30):
            mujoco.mj_step(self.m, self.d)
        self.k, self.prev, self.f_last, self.f_peak, self.stopped = 0, np.zeros(len(self.cmd)), np.zeros(3), 0.0, 0.0
        self.stops = []
        self.e_t, self.e_yaw = self.drawer_pose()
        self.dep = self.depth()
        self.max_f = 0.0
        self.n_stops = 0
        obs = self.obs()
        if kind is not None:
            for _ in range(k_pre):
                a = preroll(kind, dict(stopped=[obs["stopped"]], f_peak=[obs["f_peak"]]), self.rng)[0][:len(self.cmd)]
                obs, _, _, _ = self.step(action_of(a))
            self.max_f, self.n_stops, self.stops = 0.0, 0, []
            obs = self.obs()
        return obs

    def set_hand(self, x, y, th):
        self.d.qpos[[self.jx, self.jy, self.jyaw]] = x, y, th
        self.d.ctrl[:3] = x, y, th
        mujoco.mj_forward(self.m, self.d)

    def ctrl(self, c):
        """Commanded pose (believed frame, + z offset in v2) -> actuator targets."""
        self.d.ctrl[:3] = self.to_true(*c[:3])
        if self.v2:
            self.d.ctrl[3] = c[3]

    def measured_cmd(self):
        q = self.d.qpos
        c = list(self.to_believed(q[self.jx], q[self.jy], q[self.jyaw]))
        return np.array(c + ([q[self.jz]] if self.v2 else []))

    def obs(self):
        o = dict(n=self.cmd[0], t=self.cmd[1], yaw=self.cmd[2], off_est=self.p["est_off"], yaw_est=self.p["est_yaw"],
                 f=self.f_last.copy(), f_peak=self.f_peak, stopped=self.stopped, prev=self.prev.copy(), k=self.k)
        if self.v2:
            o.update(z=self.cmd[3], gap_est=self.p["gap_est"],
                     jam_rep=float(sum(abs(n - self.cmd[0]) < JAM_DN for n in self.stops)))
        return o

    def step(self, a):
        """One decision: a in [-1, 1]^3 (v2: ^4; a 3-vector means dz = 0) -> (obs, reward, done, info)."""
        step = step_of(a)
        if len(step) < len(self.cmd):
            step = np.r_[step, 0.0]
        old = self.cmd.copy()
        new = old + step
        new[0] = np.clip(new[0], self.cmd0[0] - 0.03, self.n_max)
        new[1] = np.clip(new[1], self.cmd0[1] - T_BOUND, self.cmd0[1] + T_BOUND)
        new[2] = np.clip(new[2], self.cmd0[2] - YAW_BOUND, self.cmd0[2] + YAW_BOUND)
        if self.v2:
            new[3] = np.clip(new[3], -Z_BOUND, Z_BOUND)
        n_move = max(1, int(0.8 * self.nsub))
        peak, peak_true, stopped, held = 0.0, 0.0, False, None
        f_true = np.zeros(3)
        for i in range(self.nsub):
            if held is None:
                self.ctrl(old + (new - old) * min(1.0, (i + 1) / n_move))
            mujoco.mj_step(self.m, self.d)
            if (i + 1) % SAMPLE == 0:
                f_true = self.contact_force()
                fm = self.measured(f_true)
                peak, peak_true = max(peak, np.linalg.norm(fm)), max(peak_true, np.linalg.norm(f_true))
                if held is None and np.linalg.norm(fm) > self.p["f_stop"]:
                    held = self.measured_cmd()
                    self.ctrl(held)
                    stopped = True
                self.f_last = fm
        self.cmd = new if held is None else held
        self.prev, self.f_peak, self.stopped, self.k = step, peak, float(stopped), self.k + 1
        self.n_stops += int(stopped)
        if stopped:
            self.stops.append(self.cmd[0])
        self.max_f = max(self.max_f, peak_true)
        err, protr = self.front_err()
        f_end = float(np.linalg.norm(f_true))
        succ = err <= FLUSH_TOL and f_end < F_OK
        dep = self.depth()
        r = 3.0 * (dep - self.dep) - F_PEN * max(0.0, peak_true - F_OK) / 10.0 + 0.1 * succ
        self.dep = dep
        done = self.k >= H_STEPS
        if done:
            r += 1.0 * succ
        info = dict(success=bool(succ), front_err=err, protrusion=protr, f_end=f_end)
        if done:
            info.update(max_f=self.max_f, n_stops=self.n_stops, e_t=self.e_t, e_yaw=self.e_yaw, depth=dep,
                        h_mode=self.p.get("h_mode", 0.0))
        return self.obs(), r, done, info
