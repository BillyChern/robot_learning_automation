"""Joint-trajectory planning: joint-space quintic moves and straight Cartesian lines (IK on the fitted model,
corrected at the endpoints with the robot's FK oracle when available)."""
import numpy as np
from .kin import Kin, rotvec, unwrap_near, wrap180
from . import calib

DT = 0.01
NORMAL = dict(name="normal", jv=20.0, lin=0.10)   # deg/s, m/s
SLOW = dict(name="slow", jv=5.0, lin=0.02)
MARGIN = 0.92                                      # plan at 92% of the limits
ACC_JOINT = 30.0                                   # deg/s^2 peak (gentle)
ACC_LIN = 0.10                                     # m/s^2 peak


def quintic(n_or_s):
    s = np.asarray(n_or_s, float)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def slerp_R(R0, R1, s):
    w = rotvec(R1 @ R0.T)
    a = np.linalg.norm(w)
    if a < 1e-12:
        return R0.copy()
    k = w / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = a * s
    return (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ R0


class Segment:
    def __init__(self, name, path_q, speed, kin, kind):
        """path_q: (M,6) geometric path (deg), timed with a quintic profile under the speed class limits."""
        self.name, self.speed, self.kind = name, speed, kind
        self.path = np.asarray(path_q, float)
        P = np.array([kin.fk(q)[:3, 3] for q in self.path])
        dq = np.abs(np.diff(self.path, axis=0)).sum(0) if len(self.path) > 1 else np.zeros(6)
        # time so that peak joint speed / lin speed / accel stay within limits (quintic peak vel = 1.875 * L/T)
        Lq = np.max(np.abs(self.path[-1] - self.path[0])) if kind == "joint" else 0.0
        seg_dq = np.abs(np.diff(self.path, axis=0))           # per path step
        seg_dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
        # path length parameterisation (uniform in s): rates = d/ds
        M = len(self.path) - 1
        if M < 1:
            self.T = 0.0
        else:
            dq_ds = seg_dq.max(1) * M; dp_ds = seg_dp * M
            vq = speed["jv"] * MARGIN; vl = speed["lin"] * MARGIN
            # quintic: max ds/dt = 1.875/T
            T1 = 1.875 * max(dq_ds.max() / vq, dp_ds.max() / vl)
            T2 = np.sqrt(5.7735 * max(dq_ds.max() / ACC_JOINT, dp_ds.max() / ACC_LIN))
            self.T = float(max(T1, T2, 0.5))
        self.sample(kin)

    def sample(self, kin):
        n = max(int(np.ceil(self.T / DT)), 1)
        t = np.arange(n + 1) * DT; t[-1] = self.T
        s = quintic(np.clip(t / max(self.T, 1e-9), 0, 1))
        u = s * (len(self.path) - 1)
        i0 = np.clip(np.floor(u).astype(int), 0, len(self.path) - 2) if len(self.path) > 1 else np.zeros_like(u, int)
        f = (u - i0)[:, None]
        if len(self.path) > 1:
            self.q = self.path[i0] * (1 - f) + self.path[i0 + 1] * f
        else:
            self.q = np.repeat(self.path[:1], len(t), 0)
        self.t = t
        self.qd = np.gradient(self.q, DT, axis=0) if len(t) > 1 else np.zeros_like(self.q)
        self.tool = np.array([kin.fk(q)[:3, 3] for q in self.q[:: max(1, len(self.q) // 60)]])

    def ref(self, t):
        """Reference joints and velocities at time t (clamped)."""
        if t <= 0:
            return self.q[0], np.zeros(6)
        if t >= self.T:
            return self.q[-1], np.zeros(6)
        k = t / DT; i = int(k); f = k - i
        i = min(i, len(self.q) - 2)
        return self.q[i] * (1 - f) + self.q[i + 1] * f, self.qd[i] * (1 - f) + self.qd[i + 1] * f


class TimedSegment(Segment):
    """Joint targets at a fixed period (policy chunks): linear interpolation, no quintic re-timing."""

    def __init__(self, name, path_q, dt, speed, kin):
        self.name, self.speed, self.kind = name, speed, "path"
        self.path = np.asarray(path_q, float)
        self.T = float(dt * max(len(self.path) - 1, 0))
        self.sample(kin)

    def sample(self, kin):
        n = max(int(np.ceil(self.T / DT)), 1)
        t = np.arange(n + 1) * DT; t[-1] = self.T
        s = np.clip(t / max(self.T, 1e-9), 0, 1)
        u = s * (len(self.path) - 1)
        i0 = np.clip(np.floor(u).astype(int), 0, max(len(self.path) - 2, 0))
        f = (u - i0)[:, None]
        self.q = self.path[i0] * (1 - f) + self.path[np.minimum(i0 + 1, len(self.path) - 1)] * f
        self.t = t
        self.qd = np.gradient(self.q, DT, axis=0) if len(t) > 1 else np.zeros_like(self.q)
        self.tool = np.array([kin.fk(q)[:3, 3] for q in self.q[:: max(1, len(self.q) // 60)]])


class Planner:
    def __init__(self, kin=None, oracle=None, log=print):
        self.kin = kin or Kin.load()
        self.oracle = oracle          # callable q_deg -> tool_pose (x,y,z,tx,ty,tz) or None
        self.log = log

    def _oracle_T(self, q):
        """The robot's own FK as a 4x4, or None (no oracle, or the controller refused this joint set: Kortex rejects
        some with ERROR_INTERNAL); the fitted model, which every safety check uses anyway, then stands in."""
        if self.oracle is None:
            return None
        try:
            return calib.pose_to_T(self.oracle(q))
        except Exception as e:
            self.log(f"FK oracle refused q {np.round(np.asarray(q, float), 1).tolist()} ({str(e)[:80]}); using the fitted model")
            return None

    def ik_exact(self, T_goal, q_seed, tol=0.0003):
        """IK on the model; then shift the model target by the oracle error until the oracle hits T_goal."""
        T_adj = T_goal.copy(); q = np.array(q_seed, float)
        info = {}
        for it in range(4):
            q, ep, er = self.kin.ik(T_adj, q)
            if ep > 1e-3 or er > 0.1:
                raise RuntimeError(f"IK failed (model) pos err {ep * 1000:.1f} mm rot {er:.2f} deg")
            To = self._oracle_T(q)
            if To is None:
                info = dict(oracle=False); break
            dp = T_goal[:3, 3] - To[:3, 3]
            dR = T_goal[:3, :3] @ To[:3, :3].T
            info = dict(oracle=True, pos_err_mm=float(np.linalg.norm(dp) * 1000), rot_err_deg=float(np.degrees(np.linalg.norm(rotvec(dR)))), it=it)
            if np.linalg.norm(dp) < tol and info["rot_err_deg"] < 0.05:
                break
            T_adj[:3, 3] += dp; T_adj[:3, :3] = dR @ T_adj[:3, :3]
        q = unwrap_near(q, q_seed)
        return q, T_adj, info

    def joint_move(self, name, q0, q1, speed=NORMAL):
        q1 = unwrap_near(q1, q0)
        n = int(max(np.max(np.abs(q1 - q0)) / 0.5, 2))
        path = np.linspace(q0, q1, n + 1)
        return Segment(name, path, speed, self.kin, "joint")

    def cart_line(self, name, q0, T_goal, speed=SLOW, step=0.002, exact_goal=True):
        """Straight tool line from FK(q0) to T_goal. The goal joint set is solved exactly (oracle); intermediate
        samples blend the model-target correction linearly between start (oracle error at q0) and goal."""
        q0 = np.array(q0, float)
        T0 = self.kin.fk(q0)
        if exact_goal:
            q1, T1adj, info = self.ik_exact(T_goal, q0)
        else:
            q1, ep, er = self.kin.ik(T_goal, q0); T1adj = T_goal; info = {}
        # the robot's actual start pose (oracle) -> model target offset at s=0 is zero by definition (we start at q0)
        Ts = self._oracle_T(q0)
        if Ts is None:
            Ts = T0
        L = np.linalg.norm(T_goal[:3, 3] - Ts[:3, 3])
        n = int(max(np.ceil(L / step), 2))
        path = [q0]; q = q0.copy()
        corr1 = T1adj[:3, 3] - T_goal[:3, 3]          # model target offset at the goal
        corr0 = T0[:3, 3] - Ts[:3, 3]                  # model target offset at the start
        for k in range(1, n + 1):
            s = k / n
            p = Ts[:3, 3] * (1 - s) + T_goal[:3, 3] * s + corr0 * (1 - s) + corr1 * s
            R = slerp_R(T0[:3, :3], T1adj[:3, :3], s)
            Tk = np.eye(4); Tk[:3, :3] = R; Tk[:3, 3] = p
            q, ep, er = self.kin.ik(Tk, q, iters=50)
            if ep > 1e-3:
                raise RuntimeError(f"IK failed on line {name} at s={s:.2f}: {ep * 1000:.1f} mm")
            path.append(q.copy())
        path[-1] = unwrap_near(q1, path[-2])
        path = np.array(path)
        jump = np.abs(np.diff(path, axis=0)).max()
        if jump > 3.0:
            raise RuntimeError(f"line {name}: joint jump {jump:.1f} deg between samples")
        seg = Segment(name, path, speed, self.kin, "line")
        seg.goal_info = info
        return seg


def tool_down_T(xyz, yaw_deg):
    """Tool pointing down (tool z = -base z); jaw-closing axis (tool x) at yaw in the base xy plane."""
    y = np.radians(yaw_deg)
    x_ax = np.array([np.cos(y), np.sin(y), 0.0]); z_ax = np.array([0, 0, -1.0]); y_ax = np.cross(z_ax, x_ax)
    T = np.eye(4); T[:3, :3] = np.stack([x_ax, y_ax, z_ax], 1); T[:3, 3] = xyz
    return T
