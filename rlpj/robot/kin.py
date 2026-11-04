"""Kinematic model of the Kinova Gen3 6-DoF + tool (from the Kinova URDF, ros_kortex gen3 6dof), with an
optional fitted correction (per-joint origin perturbations) identified against the robot's own FK oracle."""
import json, os
import numpy as np

PI = np.pi
# (xyz, rpy) of joint i origin in its parent link (URDF)
URDF = [
    ((0, 0, 0.15643), (-PI, 0, 0)),
    ((0, 0.005375, -0.12838), (PI / 2, 0, 0)),
    ((0, -0.41, 0), (PI, 0, 0)),
    ((0, 0.20843, -0.006375), (PI / 2, 0, 0)),
    ((0, -0.00017505, -0.10593), (-PI / 2, 0, 0)),
    ((0, 0.10593, -0.00017505), (PI / 2, 0, 0)),
]
EE = ((0, 0, -0.061525), (PI, 0, PI))     # bracelet -> end_effector_link (flange)
TOOL_Z = 0.12                              # Kortex tool transform
LIMITS_DEG = np.array([[-1e9, 1e9], [-128.9, 128.9], [-147.8, 147.8], [-1e9, 1e9], [-120.3, 120.3], [-1e9, 1e9]])
VMAX_DEG = np.array([79, 79, 79, 69, 69, 69])
def fit_file():
    """Per-arm fit to the FK oracle, written by `python -m rlpj.robot.fit_kin` (read-only, ~1 min)."""
    from rlpj import config
    return str(config.RUNS / "calib" / config.robot()["name"] / "kin_fit.json")


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def Tf(xyz, rpy):
    T = np.eye(4); T[:3, :3] = rpy_R(*rpy); T[:3, 3] = xyz; return T


def Rz(q):
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T


class Kin:
    def __init__(self, delta=None):
        """delta: (7,6) perturbations [dx,dy,dz,droll,dpitch,dyaw] for the 6 joint origins + EE."""
        self.delta = np.zeros((7, 6)) if delta is None else np.asarray(delta, float).reshape(7, 6)
        self._build()

    def _build(self):
        self.origins = []
        for i, (xyz, rpy) in enumerate(URDF + [EE]):
            d = self.delta[i]
            self.origins.append(Tf(np.add(xyz, d[:3]), np.add(rpy, d[3:])))
        self.T_flange_tool = Tf((0, 0, TOOL_Z), (0, 0, 0))

    @classmethod
    def load(cls, path=None):
        path = path or fit_file()
        if os.path.exists(path):
            return cls(json.load(open(path))["delta"])
        return cls()

    def frames(self, q_deg):
        """List of 4x4: joint frames 1..6 (after rotation), flange, tool."""
        q = np.radians(q_deg)
        T = np.eye(4); out = []
        for i in range(6):
            T = T @ self.origins[i] @ Rz(q[i]); out.append(T)
        Tfl = T @ self.origins[6]; out.append(Tfl); out.append(Tfl @ self.T_flange_tool)
        return out

    def fk(self, q_deg):
        return self.frames(q_deg)[-1]

    def jac(self, q_deg, eps=1e-4):
        """6x6 numeric Jacobian of [pos (m), rotvec (rad)] wrt q (rad)."""
        T0 = self.fk(q_deg); J = np.zeros((6, 6))
        for i in range(6):
            dq = np.array(q_deg, float); dq[i] += np.degrees(eps)
            T1 = self.fk(dq)
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            J[3:, i] = rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps
        return J

    def ik(self, T_target, q0_deg, iters=200, tol_p=1e-5, tol_r=1e-4, w_rot=1.0):
        """Damped least squares IK from q0. Returns (q_deg, pos_err, rot_err_deg)."""
        q = np.array(q0_deg, float)
        for _ in range(iters):
            T = self.fk(q)
            ep = T_target[:3, 3] - T[:3, 3]
            er = rotvec(T_target[:3, :3] @ T[:3, :3].T)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                break
            J = self.jac(q)
            e = np.r_[ep, w_rot * er]
            Jw = J.copy(); Jw[3:] *= w_rot
            lam = 1e-4
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.2: dq *= 0.2 / n
            q += np.degrees(dq)
        T = self.fk(q)
        return q, float(np.linalg.norm(T_target[:3, 3] - T[:3, 3])), float(np.degrees(np.linalg.norm(rotvec(T_target[:3, :3] @ T[:3, :3].T))))


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    if a > PI - 1e-6:
        w, v = np.linalg.eigh(R + np.eye(3)) if False else (None, None)
        # near pi: use the axis from the symmetric part
        A = (R + np.eye(3)) / 2; k = np.argmax(np.diag(A)); ax = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return ax * a
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def unwrap_near(q, ref):
    """Shift continuous joints (1,4,6) by multiples of 360 to be closest to ref; limited joints wrapped to +-180."""
    q = wrap180(q); ref = np.asarray(ref, float)
    return ref + wrap180(q - ref)
