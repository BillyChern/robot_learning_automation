"""Lying-cabinet env (MuJoCo, CPU): the drawer, held by its pull, is lowered back wall first into the compartment.

The policy works in the BELIEVED opening frame (lying_params, the task yaml); the sim world is the TRUE opening frame.
Command c = (x, y, yaw, z): tool point in the believed frame, z = tool height above the believed flush pose."""
import mujoco
import numpy as np

from . import lying_params as P
from .lying_baselines import preroll
from .lying_policy import H_STEPS, XY_BOUND, YAW_BOUND, Z_HIGH, Z_LOW, action_of, bottom_offset, rot, step_of
from .lying_scene import pose, xml

FLUSH_TOL, F_OK = 0.010, 40.0      # success: every top corner within 1 cm of the rim, force below 40 N
F_PEN = 0.05                       # reward per 10 N of a step's true peak force above F_OK
SAMPLE = 25                        # substeps between force samples (40 Hz)
SETTLE = 30


class LyingEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ frames
    def to_true(self, c):
        """(x, y, yaw, z) believed -> hand joint targets (x, y, z, yaw rad) in the true frame."""
        d = self.p["open_dyaw"]
        xy = rot(-d) @ (np.asarray(c[:2]) - [self.p["open_dx"], self.p["open_dy"]])
        return np.array([xy[0], xy[1], c[3] + self.z_flush, np.radians(c[2] - d)])

    def to_believed(self, q):
        """Hand joint values (x, y, z, yaw rad) -> (x, y, yaw, z) believed."""
        xy = rot(self.p["open_dyaw"]) @ np.asarray(q[:2]) + [self.p["open_dx"], self.p["open_dy"]]
        return np.array([xy[0], xy[1], np.degrees(q[3]) + self.p["open_dyaw"], q[2] - self.z_flush])

    # ------------------------------------------------------------ state readout
    def site(self, name):
        return self.d.site(name).xpos

    def top_err(self):
        """Worst distance of the drawer's top corners from the true rim (m)."""
        return max(abs(self.site(f"top_{i}")[2] - self.p["rim_h"]) for i in range(4))

    def depth(self):
        """Bottom-face centre below the rim (m)."""
        return float(np.clip(self.p["rim_h"] - self.site("bot_c")[2], -0.03, self.p["L"]))

    def errors(self):
        """True bottom-centre xy (opening frame, m) and drawer yaw (deg) relative to the opening."""
        xm = self.d.body("drawer").xmat.reshape(3, 3)
        R = xm @ self.R_in
        return self.site("bot_c")[:2].copy(), float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

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
        """Force estimate in the believed frame: true + bias + height drift + noise."""
        fb = np.r_[rot(self.p["open_dyaw"]) @ f_true[:2], f_true[2]]
        dz = self.d.qpos[self.jq[2]] - self.z_start
        return fb + self.fb + self.fd * dz + self.rng.normal(0, self.p["f_noise"], 3)

    # ------------------------------------------------------------ episode
    def reset(self, p=None, kind=None, k_pre=0):
        """New episode with params p (sampled if None); optionally a scripted pre-roll of k_pre steps (reset-to-state)."""
        self.p = p = dict(p) if p is not None else P.sample(self.rng)
        self.m = mujoco.MjModel.from_xml_string(xml(p))
        self.d = mujoco.MjData(self.m)
        self.bid = self.m.body("drawer").id
        self.jq = [self.m.joint(n).qposadr[0] for n in ("hx", "hy", "hz", "hyaw")]
        self.R_in = pose(p)[1]                       # drawer frame in the grip frame (for the yaw readout)
        self.fb = np.array([p["fb_x"], p["fb_y"], p["fb_z"]])
        self.fd = np.array([p["fd_x"], p["fd_y"], p["fd_z"]])
        self.z_flush = P.RIM_H + p["est_hold"]
        est = dict(off=(p["est_off_x"], p["est_off_y"]), yaw=p["est_yaw"], tilt=(p["est_tilt_x"], p["est_tilt_y"]))
        yaw0 = -p["est_yaw"]
        xy0 = -rot(yaw0) @ bottom_offset(est, P.RIM_H - P.FLOOR_H)
        c = np.array([xy0[0], xy0[1], yaw0, 0.2])
        self.set_hand(self.to_true(c))
        low = min(self.site(f"bot_{i}")[2] for i in range(4))
        c[3] += p["rim_h"] + p["gap0"] - low
        self.set_hand(self.to_true(c))
        self.z_start = self.to_true(c)[2]
        self.cmd0, self.cmd = c.copy(), c.copy()
        self.nsub = int(round(p["dt_dec"] / self.m.opt.timestep))
        for _ in range(SETTLE):
            mujoco.mj_step(self.m, self.d)
        self.k, self.prev, self.f_last, self.f_peak, self.stopped = 0, np.zeros(4), np.zeros(3), 0.0, 0.0
        self.e_xy, self.e_yaw = self.errors()
        self.dep = self.depth()
        self.max_f, self.n_stops = 0.0, 0
        obs = self.obs()
        if kind is not None:
            for _ in range(k_pre):
                a = preroll(kind, dict(stopped=[obs["stopped"]], f_peak=[obs["f_peak"]]), self.rng)[0]
                obs, _, _, _ = self.step(action_of(a))
            self.max_f, self.n_stops = 0.0, 0
            obs = self.obs()
        return obs

    def set_hand(self, q):
        self.d.qpos[self.jq] = q
        self.d.ctrl[:4] = q
        mujoco.mj_forward(self.m, self.d)

    def obs(self):
        p = self.p
        return dict(x=self.cmd[0], y=self.cmd[1], yaw=self.cmd[2], z=self.cmd[3], off=(p["est_off_x"], p["est_off_y"]),
                    yaw_est=p["est_yaw"], tilt=(p["est_tilt_x"], p["est_tilt_y"]), f=self.f_last.copy(),
                    f_peak=self.f_peak, stopped=self.stopped, prev=self.prev.copy(), k=self.k)

    def step(self, a):
        """One decision: a in [-1, 1]^4 -> (obs, reward, done, info)."""
        step = step_of(a)
        old = self.cmd.copy()
        new = old + step
        new[:2] = np.clip(new[:2], self.cmd0[:2] - XY_BOUND, self.cmd0[:2] + XY_BOUND)
        new[2] = np.clip(new[2], self.cmd0[2] - YAW_BOUND, self.cmd0[2] + YAW_BOUND)
        new[3] = np.clip(new[3], Z_LOW, self.cmd0[3] + Z_HIGH)
        n_move = max(1, int(0.8 * self.nsub))
        peak, peak_true, held = 0.0, 0.0, None
        f_true = np.zeros(3)
        for i in range(self.nsub):
            if held is None:
                self.d.ctrl[:4] = self.to_true(old + (new - old) * min(1.0, (i + 1) / n_move))
            mujoco.mj_step(self.m, self.d)
            if (i + 1) % SAMPLE == 0:
                f_true = self.contact_force()
                fm = self.measured(f_true)
                peak, peak_true = max(peak, np.linalg.norm(fm)), max(peak_true, np.linalg.norm(f_true))
                if held is None and np.linalg.norm(fm) > self.p["f_stop"]:
                    held = self.to_believed(self.d.qpos[self.jq])
                    self.d.ctrl[:4] = self.to_true(held)
                self.f_last = fm
        self.cmd = new if held is None else held
        self.prev, self.f_peak, self.stopped, self.k = step, peak, float(held is not None), self.k + 1
        self.n_stops += int(held is not None)
        self.max_f = max(self.max_f, peak_true)
        err = self.top_err()
        f_end = float(np.linalg.norm(f_true))
        succ = err <= FLUSH_TOL and f_end < F_OK
        dep = self.depth()
        r = 3.0 * (dep - self.dep) - F_PEN * max(0.0, peak_true - F_OK) / 10.0 + 0.1 * succ
        self.dep = dep
        done = self.k >= H_STEPS
        if done:
            r += 1.0 * succ
        info = dict(success=bool(succ), top_err=err, f_end=f_end, depth=dep)
        if done:
            info.update(max_f=self.max_f, n_stops=self.n_stops, e_x=self.e_xy[0], e_y=self.e_xy[1], e_yaw=self.e_yaw)
        return self.obs(), r, done, info
