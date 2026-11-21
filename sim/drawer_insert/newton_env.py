"""GPU twin of env.DrawerEnv: N worlds in Newton (SolverMuJoCo / mujoco_warp), synchronized batched episodes.

All worlds share one model built from scene.xml(nominal); per-world parameters come from parsing scene.xml(p) of each
world and are written into the model arrays. The drawer body keeps the template frame; each world's in-hand offset,
yaw and height are folded into its shape transforms and inertia (same rigid geometry, same dynamics)."""
import re

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact

from . import params as P
from .baselines import preroll
from .env import F_OK, F_PEN, FLUSH_TOL, N_PAST, SAMPLE, T_BOUND, YAW_BOUND, action_of
from .policy import H_STEPS, step_of
from .scene import xml

DT = 0.001          # physics step (s), as in scene.xml
NSUB = 200          # physics steps per decision (0.2 s); the move time follows dt_dec per world
CHUNK = 5           # physics steps per captured CUDA graph
SETTLE = 30         # settle steps after placing the hand
NUM = r"([-0-9.e ]+)"


@wp.kernel
def drive(old: wp.array2d(dtype=float), new: wp.array2d(dtype=float), n_move: wp.array(dtype=int),
          held: wp.array(dtype=int), hold: wp.array2d(dtype=float), sub: wp.array(dtype=int), nd: int,
          target: wp.array(dtype=float)):
    """Hand targets of this physics step: linear move old -> new over n_move steps, or the held pose."""
    w = wp.tid()
    f = wp.min(1.0, float(sub[0] + 1) / float(n_move[w]))
    for j in range(3):
        v = old[w, j] + (new[w, j] - old[w, j]) * f
        if held[w] != 0:
            v = hold[w, j]
        target[w * nd + j] = v


@wp.kernel
def tick(sub: wp.array(dtype=int)):
    sub[0] = sub[0] + 1


def vec(s):
    return np.array(s.split(), float)


def parse(p):
    """Per-world values from scene.xml(p): geoms, drawer frame, inertial, joints, actuators, sites, friction."""
    s = xml(p)
    g = {n: (vec(a), vec(b)) for n, _, a, b in re.findall(rf'<geom name="(\w+)" type="(\w+)" pos="{NUM}" size="{NUM}"', s)}
    dpos, dquat = (vec(v) for v in re.search(rf'<body name="drawer" pos="{NUM}" quat="{NUM}"', s).groups())
    hand_z = vec(re.search(rf'<body name="hand" pos="{NUM}"', s).group(1))[2]
    ipos, mass, dinert = (vec(v) for v in re.findall(rf'<inertial pos="{NUM}" mass="{NUM}" diaginertia="{NUM}"', s)[-1])
    joints = {n: dict(re.findall(r'(\w+)="([^"]+)"', a)) for n, a in re.findall(r'<joint name="(\w+)"([^/]*)/>', s)}
    kp = dict(re.findall(r'<position name="\w+" joint="(\w+)" kp="([^"]+)"', s))
    sites = {n: vec(v) for n, v in re.findall(rf'<site name="(\w+)" pos="{NUM}"', s)}
    mu = float(re.search(r'<default class="drw"><geom friction="([0-9.]+)', s).group(1))
    return dict(g=g, dpos=dpos, psi=2 * np.arctan2(dquat[3], dquat[0]), hand_z=hand_z, ipos=ipos, mass=float(mass[0]),
                dinert=dinert, joints=joints, kp=kp, sites=sites, mu=mu)


def rot2(a, v):
    """Rotate xy rows of v (N, ..., 2) by angles a (N,)."""
    c, s = np.cos(a), np.sin(a)
    while c.ndim < v.ndim - 1:
        c, s = c[..., None], s[..., None]
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


class NewtonEnv:
    def __init__(self, num_worlds, seed=0, device="cuda:0"):
        wp.init()
        self.N, self.rng, self.device = num_worlds, np.random.default_rng(seed), device
        self.t0 = parse(P.nominal())
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
            bodies = [lab.rsplit("/", 1)[-1] for lab in m.body_label[: self.nb]]
            self.b_grip, self.b_drw = bodies.index("grip"), bodies.index("drawer")
            self.solver = newton.solvers.SolverMuJoCo(m, njmax=200, nconmax=48)
            self.sensor = SensorContact(m, sensing_bodies="*/grip/drawer", request_contact_attributes=False)
            cp = newton.CollisionPipeline(m, rigid_contact_max=self.solver.get_max_contact_count(), soft_contact_max=0)
            self.contacts = cp.contacts()
            kinds = set(self.sensor.solver_observable_kinds) | {self.solver.ObservableKind.OVERFLOW}
            self.obsk = self.solver.observables(contacts=self.contacts, kinds=kinds)
            self.overflow = 0                                # OR of the solver's overflow bits seen at force samples
            self.s0, self.s1, self.ctrl = m.state(), m.state(), m.control()
            z3 = lambda: wp.zeros((self.N, 3), dtype=float)
            self.d_old, self.d_new, self.d_hold = z3(), z3(), z3()
            self.d_move, self.d_held = wp.ones(self.N, dtype=int), wp.ones(self.N, dtype=int)
            self.d_sub = wp.zeros(1, dtype=int)
        self.graph = None

    # ------------------------------------------------------------ parameters
    def apply(self, p):
        """Write per-world geometry, friction, inertia and joint parameters into the model; notify the solver."""
        m, N, ns, t0 = self.model, self.N, self.ns, self.t0
        X = [parse(pi) for pi in p]
        psi = np.array([x["psi"] for x in X])
        dz = np.array([x["hand_z"] - t0["hand_z"] for x in X])
        off = np.array([x["dpos"] for x in X]) - t0["dpos"] + np.c_[np.zeros((N, 2)), dz]   # in the template drawer frame
        st = m.shape_transform.numpy().reshape(N, ns, 7)
        sc = m.shape_scale.numpy().reshape(N, ns, 3)
        mu = m.shape_material_mu.numpy().reshape(N, ns)
        qz = np.c_[np.zeros((N, 2)), np.sin(psi / 2), np.cos(psi / 2)]                     # xyzw about z
        for i, n in enumerate(self.names):
            if n not in t0["g"]:
                continue
            pos = np.array([x["g"][n][0] for x in X])
            sc[:, i, : len(t0["g"][n][1])] = [x["g"][n][1] for x in X]
            if n.startswith("d_"):
                pos = off + np.c_[rot2(psi, pos[:, :2]), pos[:, 2]]
                st[:, i, 3:] = qz
                mu[:, i] = [x["mu"] for x in X]
            st[:, i, :3] = pos
        m.shape_transform.assign(st.reshape(-1, 7)); m.shape_scale.assign(sc.reshape(-1, 3))
        m.shape_material_mu.assign(mu.ravel())
        bm, bc, bi = m.body_mass.numpy().reshape(N, -1), m.body_com.numpy().reshape(N, -1, 3), m.body_inertia.numpy().reshape(N, -1, 3, 3)
        ip = np.array([x["ipos"] for x in X])
        bc[:, self.b_drw] = off + np.c_[rot2(psi, ip[:, :2]), ip[:, 2]]
        bm[:, self.b_drw] = [x["mass"] for x in X]
        c, s = np.cos(psi), np.sin(psi)
        R = np.zeros((N, 3, 3)); R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
        bi[:, self.b_drw] = R @ (np.array([x["dinert"] for x in X])[:, :, None] * np.eye(3)) @ R.transpose(0, 2, 1)
        m.body_mass.assign(bm.ravel()); m.body_com.assign(bc.reshape(-1, 3)); m.body_inertia.assign(bi.reshape(-1, 3, 3))
        m.body_inv_mass.assign(np.where(bm > 0, 1 / np.maximum(bm, 1e-12), 0).ravel())
        m.body_inv_inertia.assign(np.linalg.inv(bi.reshape(-1, 3, 3) + 1e-12 * np.eye(3)))
        jn = ("hx", "hy", "hyaw", "slip_t", "slip_yaw")
        jv = lambda key: np.array([[float(x["joints"][j].get(key, 0.0)) for j in jn] for x in X])
        m.joint_damping.assign(jv("damping").ravel()); m.joint_stiffness.assign(jv("stiffness").ravel())
        m.joint_friction.assign(jv("frictionloss").ravel())
        ke = np.array([[float(x["kp"].get(j, 0.0)) for j in jn] for x in X])
        m.joint_target_ke.assign(ke.ravel())
        F = newton.ModelFlags
        self.solver.notify_model_changed(F.SHAPE_PROPERTIES | F.BODY_INERTIAL_PROPERTIES | F.JOINT_DOF_PROPERTIES)
        self.o_xy, self.psi = off[:, :2] + t0["dpos"][:2], psi           # true drawer frame in the grip frame
        self.site = {n: np.array([x["sites"][n][:2] for x in X]) for n in ("front_l", "front_r", "back_c")}

    # ------------------------------------------------------------ frames and readout
    def to_true(self, c):
        xy = rot2(-self.d, c[:, :2] - self.sd)
        return np.c_[xy, np.radians(c[:, 2]) - self.d]

    def to_believed(self, q):
        return np.c_[rot2(self.d, q[:, :2]) + self.sd, np.degrees(q[:, 2] + self.d)]

    def drawer(self):
        """True drawer frame (xy (N, 2), yaw (N,)) from the grip body pose."""
        bq = self.s0.body_q.numpy().reshape(self.N, self.nb, 7)[:, self.b_grip]
        th = 2 * np.arctan2(bq[:, 5], bq[:, 6])
        return bq[:, :2] + rot2(th, self.o_xy), th + self.psi

    def sites(self):
        xy, yaw = self.drawer()
        return {n: xy + rot2(yaw, v) for n, v in self.site.items()}, yaw

    def readout(self):
        """front_err, protrusion, depth, drawer front-centre t and yaw (deg)."""
        S, yaw = self.sites()
        nl, nr = S["front_l"][:, 0], S["front_r"][:, 0]
        err = np.maximum(np.abs(nl), np.abs(nr))
        dep = np.clip(S["back_c"][:, 0], -0.05, self.L)
        return err, -0.5 * (nl + nr), dep, 0.5 * (S["front_l"][:, 1] + S["front_r"][:, 1]), np.degrees(yaw)

    def force(self):
        self.sensor.update(self.s0, self.contacts, observables=self.obsk)
        return self.sensor.total_force.numpy().reshape(self.N, -1, 3).sum(1)

    def measured(self, f):
        return np.c_[rot2(self.d, f[:, :2]), f[:, 2]] + self.f_bias + self.rng.normal(0, 1, (self.N, 3)) * self.f_noise[:, None]

    def obs(self):
        """policy.obs_vector of every world, (N, 14) float32."""
        v = np.c_[self.cmd[:, 0] / 0.1, self.cmd[:, 1] / 0.02, self.cmd[:, 2] / 10.0, self.est_off / 0.02, self.est_yaw / 10.0,
                  self.f_last / 30.0, self.f_peak / 60.0, self.stopped, self.prev / [0.01, 0.004, 2.0], self.k / H_STEPS]
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
        """starts: list of dict(p, kind, k_pre) per world -> obs (N, 14)."""
        p = [s["p"] for s in starts]
        A = {k: np.array([pi[k] for pi in p], float) for k in p[0]}
        self.apply(p)
        self.d, self.sd, self.L = np.radians(A["slot_dyaw"]), np.c_[A["slot_dn"], A["slot_dt"]], A["L"]
        self.est_off, self.est_yaw, self.f_noise, self.f_stop = A["est_off"], A["est_yaw"], A["f_noise"], A["f_stop"]
        self.f_bias = self.rng.uniform(*P.RANGES["f_bias"], (self.N, 3))
        c0 = np.c_[np.zeros(self.N), -A["est_off"], -A["est_yaw"]]
        q = self.to_true(c0)                                 # hand at believed n = 0: where is the drawer's back?
        back = q[:, :2] + rot2(q[:, 2], self.o_xy + rot2(self.psi, self.site["back_c"]))
        c0[:, 0] = (-A["back_gap"] - back[:, 0]) / np.cos(self.d)
        self.cmd0, self.cmd = c0.copy(), c0.copy()
        self.n_max = 0.5 * A["wall"] + N_PAST
        self.n_move = np.maximum(1, (0.8 * np.round(A["dt_dec"] / DT)).astype(int))
        q = self.to_true(c0)
        jq = np.zeros((self.N, self.nd)); jq[:, :3] = q
        self.s0.joint_q.assign(jq.ravel()); self.s0.joint_qd.zero_()
        self.ctrl.joint_target_q.assign(jq.ravel())
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self.d_hold.assign(q); self.d_old.assign(q); self.d_new.assign(q); self.d_held.fill_(1); self.d_move.assign(self.n_move)
        self.run(SETTLE)
        z = lambda *s: np.zeros((self.N, *s))
        self.k, self.prev, self.f_last, self.f_peak, self.stopped = np.zeros(self.N, int), z(3), z(3), z(), z()
        _, _, self.dep, self.e_t, self.e_yaw = self.readout()
        self.max_f, self.n_stops = z(), np.zeros(self.N, int)
        kinds = np.array([s.get("kind") or "" for s in starts])
        k_pre = np.where(kinds != "", [int(s.get("k_pre", 0)) for s in starts], 0)
        for j in range(int(k_pre.max(initial=0))):
            live = k_pre > j
            a = np.zeros((self.N, 3))
            for kind in ("push", "drift"):
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
        new[:, 0] = np.clip(new[:, 0], self.cmd0[:, 0] - 0.03, self.n_max)
        new[:, 1] = np.clip(new[:, 1], self.cmd0[:, 1] - T_BOUND, self.cmd0[:, 1] + T_BOUND)
        new[:, 2] = np.clip(new[:, 2], self.cmd0[:, 2] - YAW_BOUND, self.cmd0[:, 2] + YAW_BOUND)
        new = np.where(live[:, None], new, self.cmd)
        old_t = self.to_true(self.cmd)
        self.d_old.assign(old_t); self.d_new.assign(self.to_true(new)); self.d_hold.assign(old_t)
        held = ~live
        self.d_held.assign(held.astype(np.int32)); self.d_sub.zero_()
        stop_b = np.zeros((self.N, 3)); stopped = np.zeros(self.N, bool)
        peak, peak_true, f_true = np.zeros(self.N), np.zeros(self.N), np.zeros((self.N, 3))
        for _ in range(NSUB // SAMPLE):
            self.run(SAMPLE)
            f_true = self.force()
            self.overflow |= int(np.bitwise_or.reduce(self.obsk.overflow.numpy()))
            fm = self.measured(f_true)
            nf = np.linalg.norm(fm, axis=1)
            peak, peak_true = np.maximum(peak, nf), np.maximum(peak_true, np.linalg.norm(f_true, axis=1))
            hit = ~held & (nf > self.f_stop)
            if hit.any():
                q = self.s0.joint_q.numpy().reshape(self.N, self.nd)[:, :3]
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
        err, protr, dep, _, _ = self.readout()
        f_end = np.linalg.norm(f_true, axis=1)
        succ = (err <= FLUSH_TOL) & (f_end < F_OK)
        r = 3.0 * (dep - self.dep) - F_PEN * np.maximum(0.0, peak_true - F_OK) / 10.0 + 0.1 * succ
        done = self.k >= H_STEPS
        r = np.where(L, r + 1.0 * (succ & done), 0.0)
        self.dep = np.where(L, dep, self.dep)
        info = dict(success=succ, front_err=err, protrusion=protr, f_end=f_end, max_f=self.max_f.copy(),
                    n_stops=self.n_stops.copy(), e_t=self.e_t, e_yaw=self.e_yaw, depth=dep, active=L)
        return r, info

    def step(self, actions):
        """actions (N, 3) in [-1, 1] -> obs (N, 14), rew (N,), done (N,), info; done worlds idle (info['active'] False)."""
        live = self.k < H_STEPS
        r, info = self.advance(np.asarray(actions, float), live)
        return self.obs(), r.astype(np.float32), self.k >= H_STEPS, info
