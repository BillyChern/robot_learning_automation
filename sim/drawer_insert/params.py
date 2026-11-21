"""Measured drawer/cabinet parameters and their domain randomization (SI units, angles in degrees)."""
import numpy as np

# ranges (lo, hi), sampled uniformly per episode; sources in README.md "Measured parameters"
RANGES = dict(
    L=(0.115, 0.130),            # drawer length front-to-back (m)
    W=(0.098, 0.108),            # drawer width (m)
    H=(0.060, 0.070),            # drawer height (m)
    wall=(0.008, 0.011),         # wall thickness (m)
    mass=(0.2, 0.5),             # drawer mass (kg)
    clearance=(0.003, 0.010),    # slot width - drawer width (m)
    vgap=(0.006, 0.015),         # slot height - drawer height (m)
    floor_h=(0.012, 0.020),      # slot floor above the table (m)
    lift=(0.001, 0.004),         # drawer bottom above the slot floor at the start (m)
    mu=(0.2, 0.7),               # drawer-cabinet friction
    slot_dn=(-0.010, 0.010),     # true minus believed front plane along n (m)
    slot_dt=(-0.010, 0.010),     # true minus believed slot centre along t (m)
    slot_dyaw=(-1.5, 1.5),       # true minus believed slot normal (deg)
    hand_off=(-0.030, 0.030),    # drawer centre minus tool along the jaw wall (m)
    hand_yaw=(-10.0, 10.0),      # drawer yaw in the hand (deg)
    est_sd_off=(0.002, 0.015),   # sd of the in-hand offset estimate error (m)
    est_sd_yaw=(1.0, 5.0),       # sd of the in-hand yaw estimate error (deg)
    back_gap=(0.020, 0.040),     # drawer back in front of the believed front plane at the start (m)
    k_slide=(1000.0, 5000.0),    # in-hand sideways stiffness (N/m)
    f_slide=(8.0, 25.0),         # in-hand sideways slip force (N)
    k_yaw=(1.0, 5.0),            # in-hand yaw stiffness (N m/rad)
    f_yaw=(0.3, 1.0),            # in-hand yaw slip torque (N m)
    kp_arm=(2e4, 8e4),           # arm translational stiffness (N/m); real: 66 N at < 0.5 mm joint-model deviation
    kp_arm_yaw=(50.0, 200.0),    # arm yaw stiffness (N m/rad)
    f_noise=(0.5, 2.5),          # force estimate white noise sd per axis (N)
    f_bias=(-3.0, 3.0),          # residual force bias per axis after zeroing (N)
    f_stop=(60.0, 70.0),         # force stop of the executor (task force_limit_n 65)
    dt_dec=(0.18, 0.22),         # decision period (s)
)
P_DEEP_STOP = 0.25               # share of cabinets whose back stop lies beyond flush (the real one may: see README)
STOP_EXTRA = ((0.0, 0.003), (0.003, 0.020))   # back stop beyond flush (m): usual, deep


def sample(rng):
    """One episode's parameters: dict of floats."""
    p = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    p["stop_extra"] = float(rng.uniform(*STOP_EXTRA[int(rng.random() < P_DEEP_STOP)]))
    p["est_off"] = p["hand_off"] + float(np.clip(rng.normal(0, p["est_sd_off"]), -0.03, 0.03))
    p["est_yaw"] = p["hand_yaw"] + float(np.clip(rng.normal(0, p["est_sd_yaw"]), -10, 10))
    return p


def nominal():
    """Mid-range parameters with perfect estimates (debugging, images)."""
    p = {k: 0.5 * (a + b) for k, (a, b) in RANGES.items()}
    p.update(slot_dn=0.0, slot_dt=0.0, slot_dyaw=0.0, hand_off=0.0, hand_yaw=0.0, est_off=0.0, est_yaw=0.0,
             f_bias=0.0, stop_extra=0.0015)
    return p
