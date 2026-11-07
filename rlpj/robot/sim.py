"""Simulation replay of planned joint trajectories: MuJoCo model of the Gen3 6-DoF + 2F-85 (from the fitted
kinematics) for videos, and numeric checks (joint limits, speeds, table clearance)."""
import functools

import numpy as np
from .kin import Kin, LIMITS_DEG, VMAX_DEG
from . import calib


@functools.lru_cache(None)
def cell_keep_in():
    """The work cell's keep-in box for the gripper (config robot.keep_in, e.g. {y: [lo, hi]}), or {}."""
    try:
        from rlpj import config
        return dict(config.load().get("robot", {}).get("keep_in") or {})
    except Exception:
        return {}

# gripper sample points in the TOOL frame (x = jaw closing axis, z = pointing direction)
def gripper_points(opening=0.085, closed_frac=0.0):
    """Fingertip/body points; fingertips 27 mm below the tool frame when open, ~47 mm when closed."""
    tip_z = 0.027 + 0.020 * closed_frac
    half = opening / 2
    pts = []
    for sx in (-1, 1):
        for y in (-0.011, 0.011):
            pts += [(sx * (half + 0.008), y, tip_z), (sx * (half + 0.008), y, tip_z - 0.04)]
    # gripper body (between flange at z=-0.12 and the knuckles at ~z=-0.04), radius ~45 mm
    for z in (-0.12, -0.08, -0.04):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            pts.append((0.045 * np.cos(a), 0.045 * np.sin(a), z))
    return np.array(pts)


def check(seg_q, kin, speed, dt=0.01, gripper_frac=0.0, table_z=None, tip_margin=0.008, allow_low=False,
          cart_tol=1.15, max_radius=0.80, keep_in=None):
    """Return (ok, report dict) for a sampled joint trajectory. keep_in ({x|y|z: [lo, hi]}, default config robot.keep_in):
    every gripper point must stay inside (the curtain beside the table)."""
    keep = cell_keep_in() if keep_in is None else keep_in
    out_of_cell = 0
    table_z = calib.table_z() if table_z is None else table_z
    rep = {}
    q = np.asarray(seg_q)
    lo, hi = LIMITS_DEG[:, 0] + 3, LIMITS_DEG[:, 1] - 3
    from .kin import wrap180
    qw = q.copy(); qw[:, [1, 2, 4]] = wrap180(qw[:, [1, 2, 4]])
    lim_bad = np.where((qw < lo) | (qw > hi))
    rep["joint_limit_violations"] = int(len(lim_bad[0]))
    v = np.abs(np.diff(q, axis=0)) / dt if len(q) > 1 else np.zeros((1, 6))
    rep["max_joint_speed"] = float(v.max())
    rep["joint_speed_ok"] = bool(v.max() <= speed["jv"] * 1.05 and np.all(v.max(0) <= VMAX_DEG))
    G = gripper_points(0.085 * (1 - gripper_frac), gripper_frac)
    min_tip, min_body, min_link, max_lin, prev, max_r = 9, 9, 9, 0.0, None, 0
    for k in range(0, len(q), 2):
        F = kin.frames(q[k])
        T = F[-1]
        P = G @ T[:3, :3].T + T[:3, 3]
        min_tip = min(min_tip, P[:8, 2].min()); min_body = min(min_body, P[8:, 2].min())
        for ax, (lo, hi) in keep.items():
            c = P[:, "xyz".index(ax)]
            out_of_cell += int(c.min() < lo or c.max() > hi)
        for Tj in F[1:6]:
            min_link = min(min_link, Tj[2, 3])
        if prev is not None:
            max_lin = max(max_lin, np.linalg.norm(T[:3, 3] - prev) / (2 * dt))
        prev = T[:3, 3]
        max_r = max(max_r, np.hypot(*T[:2, 3]))
        # keep the gripper away from the base column
        if np.hypot(*T[:2, 3]) < 0.18 and T[2, 3] < 0.25:
            rep["near_base"] = True
    rep["keep_in_violations"] = out_of_cell
    rep.update(min_tip_above_table=float(min_tip - table_z), min_body_above_table=float(min_body - table_z),
               min_link_z=float(min_link), max_lin_speed=float(max_lin), max_radius=float(max_r))
    ok = (rep["joint_limit_violations"] == 0 and rep["joint_speed_ok"] and max_lin <= speed["lin"] * cart_tol
          and min_tip - table_z > tip_margin and min_body - table_z > 0.03 and min_link - table_z > 0.05
          and not rep.get("near_base") and max_r < max_radius and out_of_cell == 0)
    rep["ok"] = bool(ok)
    return ok, rep


# ---------------- MuJoCo replay ----------------

def _quat(R):
    import mujoco
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten()); return q


def build_xml(kin, cups=()):
    def f(a): return " ".join(f"{v:.6f}" for v in a)
    body = ""
    close = ""
    radii = [0.06, 0.055, 0.05, 0.045, 0.04, 0.04]
    for i in range(6):
        T = kin.origins[i]
        nxt = kin.origins[i + 1][:3, 3] if i < 6 else np.zeros(3)
        body += f'<body name="l{i+1}" pos="{f(T[:3, 3])}" quat="{f(_quat(T[:3, :3]))}"><joint name="j{i+1}" axis="0 0 1" range="-1000 1000"/>'
        body += f'<geom type="capsule" fromto="0 0 0 {f(nxt)}" size="{radii[i]}" rgba="0.6 0.6 0.65 1"/>'
        close += "</body>"
    Tf = kin.origins[6]
    body += f'<body name="flange" pos="{f(Tf[:3, 3])}" quat="{f(_quat(Tf[:3, :3]))}">'
    body += '<geom type="cylinder" fromto="0 0 0 0 0 0.08" size="0.042" rgba="0.15 0.15 0.15 1"/>'
    body += '<geom type="box" pos="0 0 0.09" size="0.045 0.015 0.012" rgba="0.2 0.2 0.2 1"/>'
    body += '<body name="fl" pos="0.0425 0 0.12"><joint name="fl" type="slide" axis="-1 0 0" range="0 0.0425"/><geom type="box" pos="0.006 0 0.0" size="0.006 0.011 0.03" rgba="0.1 0.1 0.1 1"/></body>'
    body += '<body name="fr" pos="-0.0425 0 0.12"><joint name="fr" type="slide" axis="1 0 0" range="0 0.0425"/><geom type="box" pos="-0.006 0 0.0" size="0.006 0.011 0.03" rgba="0.1 0.1 0.1 1"/></body>'
    body += '<site name="tool" pos="0 0 0.12" size="0.005" rgba="1 0 0 1"/></body>'
    cupx = ""
    for k, c in enumerate(cups):
        cupx += f'<body name="cup{k}" mocap="true" pos="{c[0]:.4f} {c[1]:.4f} {calib.table_z() + c[2] / 2:.4f}"><geom type="cylinder" size="0.032 {c[2] / 2:.4f}" rgba="{c[3]}" contype="0" conaffinity="0"/></body>'
    return f"""<mujoco><option gravity="0 0 0"/><visual><global offwidth="640" offheight="480"/></visual>
<worldbody><light pos="0.5 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/><light pos="1.5 -1 1.5" dir="-1 1 -1" diffuse="0.4 0.4 0.4"/>
<geom type="box" pos="0.55 0 {calib.table_z() - 0.01}" size="0.6 0.6 0.01" rgba="0.85 0.85 0.85 1"/>
<geom type="cylinder" pos="0 0 0.05" size="0.06 0.05" rgba="0.2 0.3 0.6 1"/>
{cupx}
<body name="base">{body}{close}</body></worldbody></mujoco>"""


class Replay:
    def __init__(self, kin, cups=(), size=(480, 360)):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(build_xml(kin, cups))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, size[1], size[0])
        self.cams = []
        for az, el in ((150, -25), (240, -15)):
            c = mujoco.MjvCamera(); c.lookat[:] = [0.45, -0.05, 0.1]; c.distance = 1.5; c.azimuth = az; c.elevation = el
            self.cams.append(c)
        self.ncup = len(cups)

    def frame(self, q_deg, grip_frac=0.0, cup_pos=None):
        self.d.qpos[:6] = np.radians(q_deg)
        self.d.qpos[6] = self.d.qpos[7] = 0.0425 * grip_frac * 0.7
        if cup_pos is not None:
            for k, p in cup_pos.items():
                self.d.mocap_pos[k] = p
        self.mj.mj_forward(self.m, self.d)
        imgs = []
        for c in self.cams:
            self.r.update_scene(self.d, camera=c); imgs.append(self.r.render()[:, :, ::-1])
        return np.hstack(imgs)

    def video(self, path, q_list, grip_list=None, cup_list=None, fps=25, stride=4, label=""):
        import cv2
        frames = []
        for k in range(0, len(q_list), stride):
            im = self.frame(q_list[k], 0.0 if grip_list is None else grip_list[k], None if cup_list is None else cup_list[k]).copy()
            cv2.putText(im, f"{label} t={k * 0.01:.2f}s", (8, 20), 0, 0.5, (0, 0, 200), 1)
            frames.append(im)
        if frames:
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for im in frames:
                vw.write(im)
            vw.release()
        return frames

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass
