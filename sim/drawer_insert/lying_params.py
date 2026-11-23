"""Lying-cabinet task: measured geometry (config/tasks/drawer_lying.yaml) and its randomization (SI units, degrees).

Opening frame: origin at the believed target-compartment centre on the table, x along the cabinet's long axis
(yaw_deg, away from the robot base, toward the middle compartment), y = z cross x, z up from the table."""
import numpy as np

OPEN_X, OPEN_Y, RIM_H, L_EST = 0.0757, 0.1150, 0.095, 0.091    # believed (measured) values; L_EST: drawer length

RANGES = dict(
    open_x=(0.0745, 0.0770),     # true opening along x (m): measured 0.0757 (outer length, 12 mm walls)
    open_y=(0.1135, 0.1165),     # true opening along y (m): measured 0.115
    rim_h=(0.093, 0.097),        # rim above the table (m): 0.095; an inserted drawer's top measured 0.097
    L=(0.088, 0.094),            # drawer front-to-back length = its height standing (m): 0.0905-0.093 in the side cloud
                                 # of the standing drawer (live run, reset attempt_14); the yaml's floor_h 0.013 is not used
    wall=(0.010, 0.014),         # cabinet wall / divider thickness (m): 0.012
    clr_x=(0.002, 0.006),        # opening minus drawer thickness along x (m)
    clr_y=(0.002, 0.006),        # opening minus drawer width along y (m)
    flush=(-0.002, 0.003),       # drawer top above the rim when it rests on the floor (m): floor = rim + flush - length
    dwall=(0.007, 0.010),        # drawer wall thickness (m)
    r_edge=(0.0005, 0.003),      # rounding of the drawer's bottom (back wall) edges (m)
    r_rim=(0.0005, 0.002),       # rounding of the rim's inner edges (m)
    mass=(0.12, 0.30),           # drawer mass (kg)
    mu=(0.2, 0.6),               # drawer-cabinet friction
    open_dx=(-0.010, 0.010),     # true minus believed opening centre along x (m)
    open_dy=(-0.010, 0.010),     # ... along y (m)
    open_dyaw=(-3.0, 3.0),       # true minus believed opening yaw (deg)
    off_x=(-0.025, 0.025),       # drawer top-face centre minus tool point, tool x (m): the knob sits ~17 mm off centre
                                 # along the drawer's thickness (wrist look-down), on either side (drawer turned 180 deg)
    off_y=(-0.015, 0.015),       # ... tool y (m)
    hand_yaw=(-6.0, 6.0),        # drawer yaw minus tool yaw (deg)
    tilt=(0.0, 8.0),             # in-hand tilt of the drawer's axis from vertical (deg), direction uniform
    hold=(0.006, 0.018),         # tool point above the drawer's top face (m): fingertips on the pull's tab
    est_sd_off=(0.002, 0.005),   # sd of the wrist estimate of off_x / off_y (m)
    est_sd_yaw=(1.0, 3.0),       # ... of hand_yaw (deg)
    est_sd_tilt=(1.0, 3.0),      # ... of each tilt component (deg)
    est_sd_hold=(0.001, 0.003),  # ... of hold (m)
    gap0=(0.005, 0.025),         # drawer's lowest point above the rim at the start (m)
    k_soft=(0.5, 5.0),           # in-hand tilt stiffness about the jaw closing axis (N m/rad)
    f_soft=(0.08, 0.5),          # ... slip torque (N m): pinch on a ~16 mm knob, 0.1-0.3 by estimate
    k_stiff=(5.0, 20.0),         # about the other horizontal axis (N m/rad)
    f_stiff=(0.4, 1.5),          # (N m)
    k_yaw=(2.0, 8.0),            # in-hand yaw stiffness (N m/rad)
    f_yaw=(0.3, 1.0),            # in-hand yaw slip torque (N m)
    f_zslip=(20.0, 50.0),        # the drawer slides along the fingers at this force (N)
    kp_arm=(2e4, 8e4),           # arm translational stiffness (N/m)
    kp_arm_yaw=(50.0, 200.0),    # arm yaw stiffness (N m/rad)
    f_noise=(1.0, 3.0),          # force estimate white noise sd per axis (N)
    f_stop=(40.0, 65.0),         # executor force stop on |f - f0| (N)
    dt_dec=(0.18, 0.22),         # decision period (s)
)
F_BIAS, F_DRIFT = 3.0, 30.0      # per axis: residual bias after zeroing at the descent start (N), drift with height (N/m)
VARIANTS = dict(
    carry={},                                                          # the full ranges above (training, main eval)
    small=dict(open_dx=(-0.003, 0.003), open_dy=(-0.003, 0.003), open_dyaw=(-1.0, 1.0), tilt=(0.0, 3.0)),
    big_open=dict(open_dx=(-0.015, 0.015), open_dy=(-0.015, 0.015), open_dyaw=(-5.0, 5.0)),
    noisy_force=dict(f_noise=(3.0, 5.0), f_bias=(6.0, 6.0)),
    high_friction=dict(mu=(0.6, 0.9)),
    tight=dict(clr_x=(0.001, 0.002), clr_y=(0.001, 0.002)),
    loose_grip=dict(f_soft=(0.03, 0.08), f_stiff=(0.15, 0.4), f_zslip=(8.0, 20.0)),
    soft_arm=dict(kp_arm=(5e3, 2e4)),
    slow_steps=dict(dt_dec=(0.5, 1.0)),
    low_stop=dict(f_stop=(25.0, 40.0)),
    poor_est=dict(est_sd_off=(0.005, 0.010), est_sd_yaw=(3.0, 5.0), est_sd_tilt=(3.0, 6.0)),
    no_tilt_est={},                                                    # the robot passes tilt = (0, 0)
    robust=dict(f_noise=(1.0, 4.5), f_bias=(5.0, 5.0)),                # training share with a worse force estimate
)


def sample(rng, variant="carry"):
    """One episode's parameters (dict of floats): true geometry, in-hand pose, estimates, dynamics."""
    p = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    p.update({k: float(rng.uniform(*v)) for k, v in VARIANTS[variant].items()})
    a = rng.uniform(0, 2 * np.pi)
    p["tilt_x"], p["tilt_y"] = p["tilt"] * np.cos(a), p["tilt"] * np.sin(a)
    p["jaw"] = float(rng.integers(2))                                  # jaw closing axis: 0 = tool x, 1 = tool y
    e = lambda k, sd, lim: float(p[k] + np.clip(rng.normal(0, sd), -lim, lim))
    p["est_off_x"], p["est_off_y"] = e("off_x", p["est_sd_off"], 0.012), e("off_y", p["est_sd_off"], 0.012)
    p["est_yaw"] = e("hand_yaw", p["est_sd_yaw"], 8.0)
    p["est_tilt_x"], p["est_tilt_y"] = e("tilt_x", p["est_sd_tilt"], 8.0), e("tilt_y", p["est_sd_tilt"], 8.0)
    p["est_hold"] = e("hold", p["est_sd_hold"], 0.008)
    if variant == "no_tilt_est":
        p["est_tilt_x"] = p["est_tilt_y"] = 0.0
    b = p.pop("f_bias", F_BIAS)
    for i, ax in enumerate("xyz"):
        p["fb_" + ax], p["fd_" + ax] = float(rng.uniform(-b, b)), float(rng.uniform(-F_DRIFT, F_DRIFT))
    p["floor_h"] = max(0.003, p["rim_h"] + p["flush"] - p["L"])      # the back panel is at least 3 mm thick
    p["flush"] = p["floor_h"] + p["L"] - p["rim_h"]
    return p


def nominal():
    """Mid-range parameters with perfect estimates and no in-hand tilt (debugging, tests)."""
    p = {k: 0.5 * (a + b) for k, (a, b) in RANGES.items()}
    p.update(open_dx=0.0, open_dy=0.0, open_dyaw=0.0, off_x=0.0, off_y=0.0, hand_yaw=0.0, tilt=0.0, tilt_x=0.0,
             tilt_y=0.0, jaw=0.0, est_off_x=0.0, est_off_y=0.0, est_yaw=0.0, est_tilt_x=0.0,
             est_tilt_y=0.0)
    p["est_hold"] = p["hold"]
    p.update({k + ax: 0.0 for k in ("fb_", "fd_") for ax in "xyz"})
    p["floor_h"] = p["rim_h"] + p["flush"] - p["L"]
    return p
