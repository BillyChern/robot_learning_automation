"""MJCF of the lying cabinet's target compartment and the drawer held by its pull (tool pointing down).

World frame = the TRUE opening frame (lying_params): origin at the opening centre on the table, x along the cabinet's
long axis (+x: divider to the middle compartment), z up. The drawer stands on its back wall: its local frame has the
origin at the centre of its top (front) face, z up, x across its thickness (open side +x), y along its width."""
import numpy as np

from .scene import quat, rotm

HAND_MASS = 1.0                   # moving hand mass (kg)
BT = 0.006                        # drawer bottom panel thickness (m)
QX, QY = (0.70711, 0, 0.70711, 0), (0.70711, -0.70711, 0, 0)   # capsule axis along x / along y


def mul(a, b):
    """Quaternion product (w x y z)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.r_[a[0] * b[0] - a[1:] @ b[1:], a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def geom(name, kind, c, s, q=(1, 0, 0, 0), cls="cab"):
    """One <geom> line from type, centre, size (half sizes / radius, half length) and quaternion."""
    f = lambda v: " ".join(f"{x:.6f}" for x in v)
    return f'<geom name="{name}" type="{kind}" pos="{f(c)}" size="{f(s)}" quat="{f(q)}" class="{cls}"/>'


def cabinet(p):
    """Target compartment: floor, four walls with rounded inner top edges; the middle compartment's floor and far
    divider beyond +x (the end wall at -x and the side walls at +-y drop to the table outside)."""
    ox, oy, w, rim, fl, r = 0.5 * p["open_x"], 0.5 * p["open_y"], p["wall"], p["rim_h"], p["floor_h"], p["r_rim"]
    g = [geom("floor", "box", (0, 0, fl / 2), (ox, oy, fl / 2)),
         geom("floor2", "box", (2 * ox + w, 0, fl / 2), (ox, oy, fl / 2)),
         geom("wall_x2", "box", (3 * ox + 1.5 * w, 0, rim / 2), (w / 2, oy + w, rim / 2))]
    for s, ax in ((-1, "x"), (1, "x"), (-1, "y"), (1, "y")):
        n = f"{'m' if s < 0 else 'p'}{ax}"
        if ax == "x":         # walls normal to x span the corners
            g += [geom("wall_" + n, "box", (s * (ox + (w + r) / 2), 0, rim / 2), ((w - r) / 2, oy + w, rim / 2)),
                  geom("lip_" + n, "box", (s * (ox + r / 2), 0, (rim - r) / 2), (r / 2, oy, (rim - r) / 2)),
                  geom("edge_" + n, "capsule", (s * (ox + r), 0, rim - r), (r, oy), QY)]
        else:
            g += [geom("wall_" + n, "box", (0, s * (oy + (w + r) / 2), rim / 2), (ox, (w - r) / 2, rim / 2)),
                  geom("lip_" + n, "box", (0, s * (oy + r / 2), (rim - r) / 2), (ox, r / 2, (rim - r) / 2)),
                  geom("edge_" + n, "capsule", (0, s * (oy + r), rim - r), (r, ox), QX)]
    return "\n    ".join(g)


def pose(p):
    """Drawer frame in the grip frame: rotation (yaw, tilt) and the top-face centre (off_x, off_y, -hold)."""
    q = mul(quat(np.radians(p["hand_yaw"]), 0, 0), mul(quat(0, -np.radians(p["tilt_x"]), 0), quat(0, 0, np.radians(p["tilt_y"]))))
    return q, rotm(q), np.array([p["off_x"], p["off_y"], -p["hold"]])


def dims(p):
    """Drawer thickness (x), width (y), length (z) (m)."""
    return p["open_x"] - p["clr_x"], p["open_y"] - p["clr_y"], p["L"]


def drawer(p):
    """Hollow drawer standing on its back wall, transformed into the grip frame; rounded bottom edges."""
    H, W, L = dims(p)
    w, r = p["dwall"], p["r_edge"]
    q, R, t = pose(p)
    loc = [("d_top", "box", (0, 0, -w / 2), (H / 2, W / 2, w / 2), (1, 0, 0, 0)),
           ("d_bot", "box", (0, 0, -L + w / 2), (H / 2 - r, W / 2 - r, w / 2), (1, 0, 0, 0)),
           ("d_ym", "box", (0, -W / 2 + w / 2, -(L - r) / 2), (H / 2, w / 2, (L - r) / 2), (1, 0, 0, 0)),
           ("d_yp", "box", (0, W / 2 - w / 2, -(L - r) / 2), (H / 2, w / 2, (L - r) / 2), (1, 0, 0, 0)),
           ("d_xm", "box", (-H / 2 + BT / 2, 0, -(L - r) / 2), (BT / 2, W / 2 - w, (L - r) / 2), (1, 0, 0, 0))]
    for s in (-1, 1):
        loc += [(f"d_ex{'m' if s < 0 else 'p'}", "capsule", (s * (H / 2 - r), 0, -L + r), (r, W / 2 - r), QY),
                (f"d_ey{'m' if s < 0 else 'p'}", "capsule", (0, s * (W / 2 - r), -L + r), (r, H / 2 - r), QX)]
    g = [geom(n, k, t + R @ np.array(c), s, mul(q, qq), "drw") for n, k, c, s, qq in loc]
    corners = [(sx * H / 2, sy * W / 2) for sx in (-1, 1) for sy in (-1, 1)]
    sites = [(f"top_{i}", (x, y, 0)) for i, (x, y) in enumerate(corners)] + \
            [(f"bot_{i}", (x, y, -L)) for i, (x, y) in enumerate(corners)] + [("bot_c", (0, 0, -L)), ("top_c", (0, 0, 0))]
    g += [f'<site name="{n}" pos="{" ".join(f"{v:.6f}" for v in t + R @ np.array(c))}"/>' for n, c in sites]
    I = p["mass"] / 12 * np.array([W**2 + L**2, H**2 + L**2, H**2 + W**2])
    com = t + R @ np.array([0, 0, -L / 2])
    g.insert(0, f'<inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
                f'mass="{p["mass"]:.4f}" diaginertia="{I[0]:.7f} {I[1]:.7f} {I[2]:.7f}"/>')
    g.append(geom("pull", "box", (0, 0, -p["hold"] / 2), (0.002, 0.008, p["hold"] / 2), cls="vis"))
    return "\n        ".join(g)


def slip_joints(p):
    """Grasp compliance at the tool point: slide along z, tilt about the tool x and y axes, yaw; spring + slip."""
    m, L = p["mass"], p["L"]
    i_piv = m * (L**2 / 3 + p["hold"] ** 2)
    soft = (p["k_soft"], p["f_soft"]), (p["k_stiff"], p["f_stiff"])
    ka, kb = soft if p["jaw"] < 0.5 else soft[::-1]                  # jaw closing axis along tool x (0) or y (1)
    rows = [("slip_z", "slide", "0 0 1", 300.0, p["f_zslip"], 2 * np.sqrt(3000.0 * m), "-0.008 0.004"),
            ("slip_a", "hinge", "1 0 0", ka[0], ka[1], 0.5 * np.sqrt(ka[0] * i_piv), "-0.25 0.25"),
            ("slip_b", "hinge", "0 1 0", kb[0], kb[1], 0.5 * np.sqrt(kb[0] * i_piv), "-0.25 0.25"),
            ("slip_yaw", "hinge", "0 0 1", p["k_yaw"], p["f_yaw"], 0.5 * np.sqrt(p["k_yaw"] * i_piv), "-0.25 0.25")]
    return "\n        ".join(f'<joint name="{n}" type="{t}" axis="{a}" stiffness="{k:.3f}" frictionloss="{f:.3f}" '
                             f'damping="{d:.5f}" range="{rg}" limited="true" solreflimit="0.004 1"/>' for n, t, a, k, f, d, rg in rows)


def xml(p):
    """Full MJCF string. The hand origin is the tool point (between the fingertips, on the pull's tab)."""
    mt = HAND_MASS + p["mass"]
    d_lin = 2.0 * np.sqrt(p["kp_arm"] * mt)
    d_yaw = 2.0 * np.sqrt(p["kp_arm_yaw"] * (0.01 + p["mass"] * 0.01))
    return f"""<mujoco model="drawer_lying">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <default>
    <geom solref="0.003 1" solimp="0.95 0.99 0.001" condim="3"/>
    <default class="cab"><geom friction="0.05 0.005 0.0001" rgba="0.55 0.33 0.18 1"/></default>
    <default class="drw"><geom friction="{p['mu']:.3f} 0.005 0.0001" rgba="0.80 0.50 0.28 1" density="0"/></default>
    <default class="vis"><geom contype="0" conaffinity="0" density="0" rgba="0.6 0.6 0.6 1"/></default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.8 0.8 0.8" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="-0.3 -0.3 0.8" dir="0.4 0.4 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" pos="0 0 0" material="grid" friction="0.5 0.005 0.0001"/>
    {cabinet(p)}
    <body name="hand" pos="0 0 0">
      <joint name="hx" type="slide" axis="1 0 0" damping="{d_lin:.2f}"/>
      <joint name="hy" type="slide" axis="0 1 0" damping="{d_lin:.2f}"/>
      <joint name="hz" type="slide" axis="0 0 1" damping="{d_lin:.2f}"/>
      <joint name="hyaw" type="hinge" axis="0 0 1" damping="{d_yaw:.4f}"/>
      <inertial pos="0 0 0.05" mass="{HAND_MASS}" diaginertia="0.004 0.004 0.002"/>
      <geom class="vis" type="box" pos="0 -0.011 0.012" size="0.011 0.004 0.02" rgba="0.2 0.2 0.2 1"/>
      <geom class="vis" type="box" pos="0 0.011 0.012" size="0.011 0.004 0.02" rgba="0.2 0.2 0.2 1"/>
      <geom class="vis" type="cylinder" pos="0 0 0.08" size="0.035 0.045" rgba="0.9 0.9 0.9 0.35"/>
      <body name="drawer">
        {slip_joints(p)}
        {drawer(p)}
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="hx" kp="{p['kp_arm']:.1f}" ctrlrange="-1 1"/>
    <position name="ay" joint="hy" kp="{p['kp_arm']:.1f}" ctrlrange="-1 1"/>
    <position name="az" joint="hz" kp="{p['kp_arm']:.1f}" ctrlrange="-1 1"/>
    <position name="ayaw" joint="hyaw" kp="{p['kp_arm_yaw']:.2f}" ctrlrange="-3.2 3.2"/>
  </actuator>
</mujoco>"""
