"""MJCF of table, cabinet bottom slot and the held hollow drawer for one parameter set.

World frame = the TRUE slot frame: x = n (into the cabinet), y = t (along the front), z up from the table;
origin on the front plane, on the slot centre line, at table height."""
import numpy as np

BT = 0.006          # drawer bottom thickness (m)
SIDE = 0.02         # cabinet side wall / rail thickness (m)
HAND_MASS = 1.0     # moving hand mass (kg)


def box(name, c, h, **kw):
    """<geom type=box> from centre and half sizes."""
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return (f'<geom name="{name}" type="box" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
            f'size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" {extra}/>')


def cabinet(p):
    """Static slot geoms: floor rail, side walls with rounded front inner edges, ceiling rail, back stop."""
    s = 0.5 * (p["W"] + p["clearance"])
    hs = p["H"] + p["vgap"]
    f, D, r = p["floor_h"], p["L"] + p["stop_extra"], p["r_post"]
    top = f + hs
    n_c, n_h, zc, zh = 0.5 * (D + SIDE), 0.5 * (D + SIDE), 0.5 * (top + SIDE), 0.5 * (top + SIDE)
    g = [box("floor", (n_c, 0, 0.5 * f), (n_h, s + SIDE, 0.5 * f)),
         box("ceiling", (n_c, 0, top + 0.5 * SIDE), (n_h, s, 0.5 * SIDE)),
         box("stop", (D + 0.5 * SIDE, 0, f + 0.5 * hs), (0.5 * SIDE, s, 0.5 * hs))]
    for side, sg in (("r", 1), ("l", -1)):
        g += [box(f"side_{side}", (n_c + 0.5 * r, sg * (s + 0.5 * SIDE), zc), (n_h - 0.5 * r, 0.5 * SIDE, zh)),
              box(f"lip_{side}", (0.5 * r, sg * (s + 0.5 * (SIDE + r)), zc), (0.5 * r, 0.5 * (SIDE - r), zh)),
              f'<geom name="edge_{side}" type="capsule" pos="{r:.5f} {sg * (s + r):.5f} {zc:.5f}" size="{r:.5f} {zh - r:.5f}"/>']
    return "\n    ".join(x.replace("/>", ' class="cab"/>') for x in g)


def drawer_geoms(p):
    """Hollow drawer in its own frame: x from the front wall's mid-plane toward the back, y centred, z up from its
    bottom; the back vertical edges are rounded (radius r_edge)."""
    L, W, H, w, r = p["L"], p["W"], p["H"], p["wall"], p["r_edge"]
    xc = 0.5 * (L - w - r)
    g = [box("d_front", (0, 0, 0.5 * H), (0.5 * w, 0.5 * W, 0.5 * H)),
         box("d_back", (L - w, 0, 0.5 * H), (0.5 * w, 0.5 * W - r, 0.5 * H)),
         box("d_left", (xc, -0.5 * (W - w), 0.5 * H), (0.5 * (L - r), 0.5 * w, 0.5 * H)),
         box("d_right", (xc, 0.5 * (W - w), 0.5 * H), (0.5 * (L - r), 0.5 * w, 0.5 * H)),
         box("d_bottom", (0.5 * (L - w), 0, 0.5 * BT), (0.5 * L - r, 0.5 * W - r, 0.5 * BT))]
    for side, sg in (("l", -1), ("r", 1)):
        g.append(f'<geom name="d_edge_{side}" type="capsule" pos="{L - 0.5 * w - r:.5f} {sg * (0.5 * W - r):.5f} '
                 f'{0.5 * H:.5f}" size="{r:.5f} {0.5 * H - r:.5f}"/>')
    return "\n          ".join(x.replace("/>", ' class="drw"/>') for x in g)


def xml(p):
    """Full MJCF string. The hand origin is the tool point: top of the front wall, mid-thickness."""
    L, W, H, m = p["L"], p["W"], p["H"], p["mass"]
    z_tool = p["floor_h"] + p["lift"] + H
    psi = np.radians(p["hand_yaw"])
    c, s_ = np.cos(psi), np.sin(psi)
    dpos = (-s_ * p["hand_off"], c * p["hand_off"], -H)          # R(psi) @ (0, off, -H)
    dquat = (np.cos(psi / 2), 0, 0, np.sin(psi / 2))
    xc = 0.5 * (L - p["wall"])
    ixx, iyy, izz = m * (W**2 + H**2) / 12, m * (L**2 + H**2) / 12, m * (L**2 + W**2) / 12
    mt = HAND_MASS + m
    d_lin = 2.0 * np.sqrt(p["kp_arm"] * mt)
    i_yaw = 0.01 + m * (L**2 + W**2) / 3
    d_yaw = 2.0 * np.sqrt(p["kp_arm_yaw"] * i_yaw)
    d_slide = np.sqrt(p["k_slide"] * m)
    d_hy = 0.5 * np.sqrt(p["k_yaw"] * m * L**2 / 3)
    return f"""<mujoco model="drawer_insert">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <default>
    <geom solref="0.003 1" solimp="0.95 0.99 0.001" condim="3"/>
    <default class="cab"><geom friction="0.05 0.005 0.0001" rgba="0.55 0.33 0.18 1"/></default>
    <default class="drw"><geom friction="{p['mu']:.3f} 0.005 0.0001" rgba="0.75 0.45 0.25 1" density="0"/></default>
    <default class="vis"><geom contype="0" conaffinity="0" density="0" rgba="0.2 0.2 0.2 1"/></default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.8 0.8 0.8" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="-0.3 -0.3 0.8" dir="0.4 0.4 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" pos="0 0 0" material="grid" friction="0.5 0.005 0.0001"/>
    {cabinet(p)}
    <body name="hand" pos="0 0 {z_tool:.5f}">
      <joint name="hx" type="slide" axis="1 0 0" damping="{d_lin:.2f}"/>
      <joint name="hy" type="slide" axis="0 1 0" damping="{d_lin:.2f}"/>
      <joint name="hyaw" type="hinge" axis="0 0 1" damping="{d_yaw:.4f}"/>
      <inertial pos="0 0 0.05" mass="{HAND_MASS}" diaginertia="0.004 0.004 0.002"/>
      <geom class="vis" type="box" pos="-0.011 0 0.01" size="0.004 0.012 0.02"/>
      <geom class="vis" type="box" pos="0.011 0 0.01" size="0.004 0.012 0.02"/>
      <geom class="vis" type="cylinder" pos="0 0 0.07" size="0.035 0.04" rgba="0.9 0.9 0.9 0.25"/>
      <body name="grip">
        <joint name="slip_t" type="slide" axis="0 1 0" stiffness="{p['k_slide']:.1f}" frictionloss="{p['f_slide']:.2f}"
               damping="{d_slide:.3f}" range="-0.02 0.02" limited="true"/>
        <joint name="slip_yaw" type="hinge" axis="0 0 1" stiffness="{p['k_yaw']:.3f}" frictionloss="{p['f_yaw']:.3f}"
               damping="{d_hy:.5f}" range="-0.26 0.26" limited="true"/>
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
        <body name="drawer" pos="{dpos[0]:.5f} {dpos[1]:.5f} {dpos[2]:.5f}" quat="{dquat[0]:.6f} 0 0 {dquat[3]:.6f}">
          <inertial pos="{xc:.5f} 0 {0.35 * H:.5f}" mass="{m:.4f}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>
          {drawer_geoms(p)}
          <site name="front_l" pos="{-0.5 * p['wall']:.5f} {-0.5 * W:.5f} 0"/>
          <site name="front_r" pos="{-0.5 * p['wall']:.5f} {0.5 * W:.5f} 0"/>
          <site name="back_c" pos="{L - 0.5 * p['wall']:.5f} 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="hx" kp="{p['kp_arm']:.1f}" ctrlrange="-1 1"/>
    <position name="ay" joint="hy" kp="{p['kp_arm']:.1f}" ctrlrange="-1 1"/>
    <position name="ayaw" joint="hyaw" kp="{p['kp_arm_yaw']:.2f}" ctrlrange="-3.2 3.2"/>
  </actuator>
</mujoco>"""
