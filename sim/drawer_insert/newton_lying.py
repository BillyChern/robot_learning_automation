"""GPU twin of lying_env.LyingEnv: N worlds in Newton (SolverMuJoCo / mujoco_warp), synchronized batched episodes.

One model built from lying_scene.xml(nominal) and replicated; every world's geometry (geom poses and sizes, the drawer's
in-hand pose baked into its shapes), inertia, joint and actuator values are parsed from its own lying_scene.xml(p)."""
import re

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact

from . import lying_params as P
from .lying_baselines import preroll
from .lying_env import F_OK, F_PEN, FLUSH_TOL, SAMPLE, SETTLE, W_DEPTH, W_XY
from .lying_policy import H_STEPS, XY_BOUND, YAW_BOUND, Z_HIGH, Z_LOW, action_of, step_of
from .lying_scene import xml
from .newton_env import CHUNK, DT, NUM, rot2, tick, vec

NSUB = 200
JOINTS = ("hx", "hy", "hz", "hyaw", "slip_z", "slip_a", "slip_b", "slip_yaw")


@wp.kernel
def drive(old: wp.array2d(dtype=float), new: wp.array2d(dtype=float), n_move: wp.array(dtype=int),
          held: wp.array(dtype=int), hold: wp.array2d(dtype=float), sub: wp.array(dtype=int), nd: int,
          target: wp.array(dtype=float)):
    """Hand targets (x, y, z, yaw) of this physics step: linear move old -> new over n_move steps, or the held pose."""
    w = wp.tid()
    f = wp.min(1.0, float(sub[0] + 1) / float(n_move[w]))
    for j in range(4):
        v = old[w, j] + (new[w, j] - old[w, j]) * f
        if held[w] != 0:
            v = hold[w, j]
        target[w * nd + j] = v


def parse(p):
    """Per-world values from lying_scene.xml(p): geoms (pos, size, quat wxyz), sites, drawer inertial, joints, kp, mu."""
    s = xml(p)
    g = {n: (vec(a), vec(b), vec(q)) for n, a, b, q in
         re.findall(rf'<geom name="(\w+)" type="\w+" pos="{NUM}" size="{NUM}" quat="{NUM}"', s)}
    sites = {n: vec(v) for n, v in re.findall(rf'<site name="(\w+)" pos="{NUM}"', s)}
    ipos, iq, mass, dinert = (vec(v) for v in re.search(rf'<inertial pos="{NUM}" quat="{NUM}" mass="{NUM}" diaginertia="{NUM}"', s).groups())
    joints = {n: dict(re.findall(r'(\w+)="([^"]+)"', a)) for n, a in re.findall(r'<joint name="(\w+)"([^/]*)/>', s)}
    kp = dict(re.findall(r'<position name="\w+" joint="(\w+)" kp="([^"]+)"', s))
    mu = float(re.search(r'<default class="drw"><geom friction="([0-9.]+)', s).group(1))
    return dict(g=g, sites=sites, ipos=ipos, iq=iq, mass=float(mass[0]), dinert=dinert, joints=joints, kp=kp, mu=mu)


def rotq(q):
    """Rotation matrices (N, 3, 3) from quaternions (N, 4) w x y z."""
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


class NewtonLying:
    def __init__(self, num_worlds, seed=0, device="cuda:0"):
        wp.init()
        self.N, self.rng, self.device = num_worlds, np.random.default_rng(seed), device
        env = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(env)
        env.rigid_gap = 0.0                                  # MuJoCo default: no widened contact detection
        env.add_mjcf(xml(P.nominal()))
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.replicate(env, num_worlds)
        with wp.ScopedDevice(device):
            self.model = m = b.finalize()
            self.ns, self.nb, self.nd = m.shape_count // self.N, m.body_count // self.N, m.joint_dof_count // self.N
            self.names = [lab.rsplit("/", 1)[-1] for lab in m.shape_label[: self.ns]]
            self.b_drw = [lab.rsplit("/", 1)[-1] for lab in m.body_label[: self.nb]].index("drawer")
            self.solver = newton.solvers.SolverMuJoCo(m, njmax=300, nconmax=64)
            self.sensor = SensorContact(m, sensing_bodies="*/hand/drawer", request_contact_attributes=False)
            cp = newton.CollisionPipeline(m, rigid_contact_max=self.solver.get_max_contact_count(), soft_contact_max=0)
            self.contacts = cp.contacts()
            kinds = set(self.sensor.solver_observable_kinds) | {self.solver.ObservableKind.OVERFLOW}
            self.obsk = self.solver.observables(contacts=self.contacts, kinds=kinds)
            self.overflow = 0
            self.s0, self.s1, self.ctrl = m.state(), m.state(), m.control()
            z4 = lambda: wp.zeros((self.N, 4), dtype=float)
            self.d_old, self.d_new, self.d_hold = z4(), z4(), z4()
            self.d_move, self.d_held = wp.ones(self.N, dtype=int), wp.ones(self.N, dtype=int)
            self.d_sub = wp.zeros(1, dtype=int)
        self.graph = None

    # ------------------------------------------------------------ parameters
    def apply(self, p):
        """Write per-world geometry, friction, inertia and joint parameters into the model; notify the solver."""
        m, N, ns = self.model, self.N, self.ns
        X = [parse(pi) for pi in p]
        st = m.shape_transform.numpy().reshape(N, ns, 7)
        sc = m.shape_scale.numpy().reshape(N, ns, 3)
        mu = m.shape_material_mu.numpy().reshape(N, ns)
        for i, n in enumerate(self.names):
            if n in X[0]["g"]:
                G = [x["g"][n] for x in X]
                st[:, i, :3] = [g[0] for g in G]
                st[:, i, 3:] = [np.r_[g[2][1:], g[2][0]] for g in G]          # w x y z -> x y z w
                sc[:, i, : len(G[0][1])] = [g[1] for g in G]
                if n.startswith("d_"):
                    mu[:, i] = [x["mu"] for x in X]
            elif n in X[0]["sites"]:
                st[:, i, :3] = [x["sites"][n] for x in X]
        m.shape_transform.assign(st.reshape(-1, 7)); m.shape_scale.assign(sc.reshape(-1, 3))
        m.shape_material_mu.assign(mu.ravel())
        bm, bc, bi = m.body_mass.numpy().reshape(N, -1), m.body_com.numpy().reshape(N, -1, 3), m.body_inertia.numpy().reshape(N, -1, 3, 3)
        R = rotq(np.array([x["iq"] for x in X]))
        bc[:, self.b_drw] = [x["ipos"] for x in X]
        bm[:, self.b_drw] = [x["mass"] for x in X]
        bi[:, self.b_drw] = R @ (np.array([x["dinert"] for x in X])[:, :, None] * np.eye(3)) @ R.transpose(0, 2, 1)
        m.body_mass.assign(bm.ravel()); m.body_com.assign(bc.reshape(-1, 3)); m.body_inertia.assign(bi.reshape(-1, 3, 3))
        m.body_inv_mass.assign(np.where(bm > 0, 1 / np.maximum(bm, 1e-12), 0).ravel())
        m.body_inv_inertia.assign(np.linalg.inv(bi.reshape(-1, 3, 3) + 1e-12 * np.eye(3)))
        jv = lambda key: np.array([[float(x["joints"][j].get(key, 0.0)) for j in JOINTS] for x in X])
        m.joint_damping.assign(jv("damping").ravel()); m.joint_stiffness.assign(jv("stiffness").ravel())
        m.joint_friction.assign(jv("frictionloss").ravel())
        m.joint_target_ke.assign(np.array([[float(x["kp"].get(j, 0.0)) for j in JOINTS] for x in X]).ravel())
        F = newton.ModelFlags
        self.solver.notify_model_changed(F.SHAPE_PROPERTIES | F.BODY_INERTIAL_PROPERTIES | F.JOINT_DOF_PROPERTIES)
        self.site = {n: np.array([x["sites"][n] for x in X]) for n in X[0]["sites"]}
        self.R_in = R                                                       # drawer frame in the grip frame

    # ------------------------------------------------------------ frames and readout
    def to_true(self, c):
        """(N, 4) believed (x, y, yaw deg, z) -> hand joint targets (x, y, z, yaw rad), true frame."""
        xy = rot2(-self.d, c[:, :2] - self.sd)
        return np.c_[xy, c[:, 3] + self.z_flush, np.radians(c[:, 2]) - self.d]

    def to_believed(self, q):
        return np.c_[rot2(self.d, q[:, :2]) + self.sd, np.degrees(q[:, 3] + self.d), q[:, 2] - self.z_flush]

    def sites(self, names):
        """World positions (N, 3) of the named drawer sites, and the drawer body rotation (N, 3, 3)."""
        bq = self.s0.body_q.numpy().reshape(self.N, self.nb, 7)[:, self.b_drw]
        R = rotq(np.c_[bq[:, 6], bq[:, 3:6]])
        return [bq[:, :3] + np.einsum("nij,nj->ni", R, self.site[n]) for n in names], R

    def readout(self):
        """top_err (N,), depth (N,), bottom-centre xy (N, 2), drawer yaw vs the opening (deg)."""
        S, R = self.sites([f"top_{i}" for i in range(4)] + ["bot_c"])
        err = np.max([np.abs(s[:, 2] - self.rim) for s in S[:4]], 0)
        dep = np.clip(self.rim - S[4][:, 2], -0.03, self.L)
        Rd = R @ self.R_in
        return err, dep, S[4][:, :2], np.degrees(np.arctan2(Rd[:, 1, 0], Rd[:, 0, 0]))

    def force(self):
        self.sensor.update(self.s0, self.contacts, observables=self.obsk)
        return self.sensor.total_force.numpy().reshape(self.N, -1, 3).sum(1)

    def hand(self):
        return self.s0.joint_q.numpy().reshape(self.N, self.nd)[:, :4]

    def measured(self, f, z):
        fb = np.c_[rot2(self.d, f[:, :2]), f[:, 2]]
        return fb + self.fb + self.fd * (z - self.z_start)[:, None] + self.rng.normal(0, 1, (self.N, 3)) * self.f_noise[:, None]

    def obs(self):
        """lying_policy.obs_vector of every world, (N, 19) float32."""
        v = np.c_[self.cmd[:, 0] / 0.01, self.cmd[:, 1] / 0.01, self.cmd[:, 2] / 5.0, self.cmd[:, 3] / 0.05,
                  self.est[:, :2] / 0.01, self.est[:, 2] / 5.0, self.est[:, 3:5] / 5.0, self.f_last / 30.0,
                  self.f_peak / 60.0, self.stopped, self.prev / [0.004, 0.004, 2.0, 0.008], self.k / H_STEPS]
        return np.clip(v, -5.0, 5.0).astype(np.float32)

    # ------------------------------------------------------------ simulation
    def chunk(self):
        for _ in range(CHUNK):
            wp.launch(drive, self.N, inputs=[self.d_old, self.d_new, self.d_move, self.d_held, self.d_hold, self.d_sub,
                                             self.nd], outputs=[self.ctrl.joint_target_q], device=self.device)
            wp.launch(tick, 1, inputs=[self.d_sub], device=self.device)
            self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, DT, observables=self.obsk)
            self.s0.assign(self.s1)

    def run(self, n):
        if self.graph is None:
            self.chunk()                                   # compile, then capture
            with wp.ScopedCapture(device=self.device) as cap:
                self.chunk()
            self.graph = cap.graph
        for _ in range(n // CHUNK):
            wp.capture_launch(self.graph)

    def reset(self, starts):
        """starts: list of dict(p, kind, k_pre) per world -> obs (N, 19)."""
        p = [s["p"] for s in starts]
        A = {k: np.array([pi[k] for pi in p], float) for k in p[0]}
        self.apply(p)
        self.d, self.sd = np.radians(A["open_dyaw"]), np.c_[A["open_dx"], A["open_dy"]]
        self.rim, self.L, self.f_noise, self.f_stop = A["rim_h"], A["L"], A["f_noise"], A["f_stop"]
        self.fb = np.c_[A["fb_x"], A["fb_y"], A["fb_z"]]
        self.fd = np.c_[A["fd_x"], A["fd_y"], A["fd_z"]]
        self.est = np.c_[A["est_off_x"], A["est_off_y"], A["est_yaw"], A["est_tilt_x"], A["est_tilt_y"]]
        self.z_flush = P.RIM_H + A["est_hold"]
        yaw0 = -A["est_yaw"]
        tx, ty = np.radians(A["est_tilt_x"]), np.radians(A["est_tilt_y"])
        v = np.c_[np.sin(tx) * np.cos(ty), np.sin(ty)] * (P.L_EST)    # lying_policy.bottom_offset
        b = self.est[:, :2] + rot2(np.radians(A["est_yaw"]), v)
        xy0 = -rot2(np.radians(yaw0), b)
        low = np.min([self.site[f"bot_{i}"][:, 2] for i in range(4)], 0)
        z0 = A["rim_h"] + A["gap0"] - low
        c0 = np.c_[xy0, yaw0, z0 - self.z_flush]
        self.cmd0, self.cmd = c0.copy(), c0.copy()
        self.n_move = np.maximum(1, (0.8 * np.round(A["dt_dec"] / DT)).astype(int))
        q = self.to_true(c0)
        self.z_start = q[:, 2]
        jq = np.zeros((self.N, self.nd)); jq[:, :4] = q
        self.s0.joint_q.assign(jq.ravel()); self.s0.joint_qd.zero_()
        self.ctrl.joint_target_q.assign(jq.ravel())
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self.d_hold.assign(q); self.d_old.assign(q); self.d_new.assign(q); self.d_held.fill_(1); self.d_move.assign(self.n_move)
        self.run(SETTLE)
        z = lambda *s: np.zeros((self.N, *s))
        self.k, self.prev, self.f_last, self.f_peak, self.stopped = np.zeros(self.N, int), z(4), z(3), z(), z()
        _, self.dep, self.e_xy, self.e_yaw = self.readout()
        self.dxy = np.linalg.norm(self.e_xy, axis=1)
        self.max_f, self.n_stops = z(), np.zeros(self.N, int)
        kinds = np.array([s.get("kind") or "" for s in starts])
        k_pre = np.where(kinds != "", [int(s.get("k_pre", 0)) for s in starts], 0)
        for j in range(int(k_pre.max(initial=0))):
            live = k_pre > j
            a = np.zeros((self.N, 4))
            for kind in ("descend", "drift"):
                sel = live & (kinds == kind)
                if sel.any():
                    a[sel] = preroll(kind, dict(stopped=self.stopped[sel], f_peak=self.f_peak[sel]), self.rng)
            self.advance(action_of(a), live)
        self.max_f[:], self.n_stops[:] = 0.0, 0
        return self.obs()

    def advance(self, a, live):
        """One decision for the live worlds (others hold their pose): -> reward (N,), info dict."""
        step = step_of(a)
        new = self.cmd + step
        new[:, :2] = np.clip(new[:, :2], self.cmd0[:, :2] - XY_BOUND, self.cmd0[:, :2] + XY_BOUND)
        new[:, 2] = np.clip(new[:, 2], self.cmd0[:, 2] - YAW_BOUND, self.cmd0[:, 2] + YAW_BOUND)
        new[:, 3] = np.clip(new[:, 3], Z_LOW, self.cmd0[:, 3] + Z_HIGH)
        new = np.where(live[:, None], new, self.cmd)
        old_t = self.to_true(self.cmd)
        self.d_old.assign(old_t); self.d_new.assign(self.to_true(new)); self.d_hold.assign(old_t)
        held = ~live
        self.d_held.assign(held.astype(np.int32)); self.d_sub.zero_()
        stop_b = np.zeros((self.N, 4)); stopped = np.zeros(self.N, bool)
        peak, peak_true, f_true = np.zeros(self.N), np.zeros(self.N), np.zeros((self.N, 3))
        for _ in range(NSUB // SAMPLE):
            self.run(SAMPLE)
            f_true = self.force()
            self.overflow |= int(np.bitwise_or.reduce(self.obsk.overflow.numpy()))
            q = self.hand()
            fm = self.measured(f_true, q[:, 2])
            nf = np.linalg.norm(fm, axis=1)
            peak, peak_true = np.maximum(peak, nf), np.maximum(peak_true, np.linalg.norm(f_true, axis=1))
            hit = ~held & (nf > self.f_stop)
            if hit.any():
                hold = np.where(hit[:, None], q, self.d_hold.numpy())
                stop_b[hit] = self.to_believed(q)[hit]
                held |= hit; stopped |= hit
                self.d_hold.assign(hold); self.d_held.assign(held.astype(np.int32))
            self.f_last[live] = fm[live]
        L = live
        self.cmd = np.where(stopped[:, None], stop_b, new)
        self.prev[L], self.f_peak[L], self.stopped[L] = step[L], peak[L], stopped[L]
        self.k += L
        self.n_stops += stopped
        self.max_f[L] = np.maximum(self.max_f[L], peak_true[L])
        err, dep, bxy, _ = self.readout()
        dxy = np.linalg.norm(bxy, axis=1)
        f_end = np.linalg.norm(f_true, axis=1)
        succ = (err <= FLUSH_TOL) & (f_end < F_OK)
        r = W_DEPTH * (dep - self.dep) + W_XY * (self.dxy - dxy) - F_PEN * np.maximum(0.0, peak_true - F_OK) / 10.0 + 0.1 * succ
        done = self.k >= H_STEPS
        r = np.where(L, r + 1.0 * (succ & done), 0.0)
        self.dep, self.dxy = np.where(L, dep, self.dep), np.where(L, dxy, self.dxy)
        info = dict(success=succ, top_err=err, f_end=f_end, max_f=self.max_f.copy(), n_stops=self.n_stops.copy(),
                    e_x=self.e_xy[:, 0], e_y=self.e_xy[:, 1], e_yaw=self.e_yaw, depth=dep, active=L)
        return r, info

    def step(self, actions):
        """actions (N, 4) in [-1, 1] -> obs (N, 19), rew (N,), done (N,), info; done worlds idle (info['active'] False)."""
        live = self.k < H_STEPS
        r, info = self.advance(np.asarray(actions, float), live)
        return self.obs(), r.astype(np.float32), self.k >= H_STEPS, info
