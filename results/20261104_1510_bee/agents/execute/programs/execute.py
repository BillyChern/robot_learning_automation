"""bee_in_bowl execute: pick the toy bee from the table and drop it into the bowl (kinova_a, RobotSession).

Perception is geometry only (side depth blobs from s.tabletop(): height, footprint, rim circle), the bowl centre is
refined with the wrist camera (rim circle), the bee is grasped across its short axis with one close at a fixed tool
height above the fitted table, carried high and released above the bowl centre. Verification re-perceives the table.
Grasp/place structure ADAPTED from inputs/reuse/battery_placement_kinova_b (battery_grasp_place, place_verification)
and the kinova_a fingertip lesson from inputs/reuse/seed_2_kinova_a (closed tips ~38 mm below the tool).
"""
import json
import os

import cv2
import numpy as np

from rlpj.robot.session import RobotSession, Abort
from rlpj.robot.plan import tool_down_T
from rlpj.perception import wrist as wr

# ---- geometry priors (m), measured on kinova_a 2026-11-04 (config bowl h 0.069 ext 0.127-0.154; bee h 0.043 ext 0.108x0.076)
BOWL_H = (0.055, 0.095); BOWL_EXT = (0.10, 0.40)   # side footprint grows when the bee or the phantom blob merges (0.196 att. 10, 0.241 att. 21, 0.251 att. 25); centre comes from the rim fit
BOWL_H_AFTER = (0.055, 0.125)   # bowl height after the place: bee may stick up above the rim (0.099 m in att. 163)
BEE_H = (0.034, 0.060); BEE_EXT = (0.03, 0.17)   # att. 73: bee lying along the camera ray read 0.154 long (depth smear)
MIN_BEE_PTS = 500   # real bee 930-1350 side points; a phantom blob 0.07-0.15 m from the bowl (bowl depth artifact) read 60-334 points, h 0.019-0.032 (att. 3,4,8,11)
BEE_LEN_SMEAR = 0.11    # longer than the real bee (~0.09): depth smear in the blob
SPLIT_BEE_PTS = 300  # bee split off a merged bowl blob (partly hidden behind/merged with the bowl)
TASK_X = (0.28, 0.60); TASK_Y = (-0.27, 0.15)        # object range with margin for side-camera error

# ---- grasp / place parameters (tool frame heights above the fitted table)
GRASP_Z = 0.052        # closed fingertips ~38 mm below tool on kinova_a -> tips ~14 mm above table, pads on the bee's middle
PRE_Z = 0.13           # hover above the bee before the slow descent
CARRY_Z = 0.23         # bee bottom ~0.18 m above the table, far above the 7 cm rim
RELEASE_ABOVE_RIM = 0.030   # bee bottom this far above the rim at release
BEE_BELOW_TOOL = GRASP_Z    # bee bottom was on the table when grasped
LOOK_Z = 0.32          # wrist look height above the bowl
CLOSED_ON_NOTHING = 96.0    # reading above this = nothing between the jaws (99.1 empty)
LIFT_TIP_MARGIN = -0.002    # model puts closed tips 47 mm below the tool (real ~38 mm): allow the lift from GRASP_Z


def save_small(path, img, w=640):
    if img is None:
        return
    h = int(img.shape[0] * w / img.shape[1])
    cv2.imwrite(path, cv2.resize(img, (w, h)))


# ---- release check (operator note 2026-11-05: the foam bee sometimes stays wedged between the fingers after opening)
# Colour is used here only to see whether the bee hangs on the fingers, as the operator specified; objects are
# still identified by geometry.
YELLOW_LO = (0, 120, 110); YELLOW_HI = (40, 255, 255)   # operator 2026-11-06: orange bee, hue 0-40, S > 120, V > 110
STUCK_FRAC = 0.10        # bottom 15 % of the wrist image: stuck 0.21-0.22, bee below <= 0.052, empty <= 0.027
SQUEEZE_HELD = 97.5      # closing on nothing reads ~99.1, on the bee 78-91
RELEASE_TRIES = 3
CHECK_H = 0.18           # squeeze/open height limit above the table (library refuses opening a held bee above 0.20)
DARK_V = 100            # disc median brightness (HSV V) below which the wrist colour test is void: normal 165-211;
                         # att. 204 (room dark, V 23-36) read 0.164 on the EMPTY bowl before the place -> could confirm nothing
DARK_MIN_V = 12         # below this even the dark-mode red test is void (abort before grasping)
DARK_RED_FRAC = 0.04    # dark mode: share of pixels with R > G+30, R > B+30, R > 50 inside the rim. Att. 204/205 (dark):
                         # empty bowl 0.000/0.000, bee in bowl 0.127 (release_check view 0.201)
IN_BOWL_FRAC = 0.08      # bee share inside the rim seen from LOOK_Z: empty bowl 0.000, bee in bowl 0.20-0.45 (att. 3-63)
SIDE_VIEW_BACK = 0.13    # side-camera view pose: this far from the bowl centre toward the base, 8 cm above the check height


def yellow_bottom_frac(img):
    if img is None:
        return None
    h = img.shape[0]
    hsv = cv2.cvtColor(img[int(0.85 * h):], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(YELLOW_LO, np.uint8), np.array(YELLOW_HI, np.uint8))
    return float(np.count_nonzero(m)) / m.size


def bowl_disc_frac(img, T_base_tool, bowl_xy, rim_z, bowl_r):
    """Bee-colour share inside the bowl rim (0.85 of the rim radius) in the wrist image, bottom 15 % (fingertips)
    excluded. Offline on 61 runs: empty bowl 0.000, bee in the bowl 0.204-0.446."""
    uv = wr.project_base_to_wrist(np.array([[bowl_xy[0], bowl_xy[1], rim_z],
                                            [bowl_xy[0] + bowl_r, bowl_xy[1], rim_z]]), T_base_tool)
    if not np.all(np.isfinite(uv)):
        return None
    r = 0.85 * float(np.linalg.norm(uv[1] - uv[0]))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx - uv[0, 0]) ** 2 + (yy - uv[0, 1]) ** 2 < r * r) & (yy < 0.85 * h)
    if m.sum() < 1000:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # yellow, orange and red (the bee carries orange/red markings since ~att. 64; yellow alone fell to 0.104 in att. 69).
    # The cream bowl and grey table stay below S 100: offline empty bowl 0.000, bee in bowl 0.222-0.541.
    y = (cv2.inRange(hsv, np.array((0, 100, 90), np.uint8), np.array((40, 255, 255), np.uint8)) > 0) | \
        (cv2.inRange(hsv, np.array((170, 100, 90), np.uint8), np.array((180, 255, 255), np.uint8)) > 0)
    bowl_disc_frac.v_median = float(np.median(hsv[..., 2][m]))
    b_, g_, r_ = [img[..., i].astype(np.int16) for i in range(3)]
    red = (r_ - g_ > 30) & (r_ - b_ > 30) & (r_ > 50)
    bowl_disc_frac.red_frac = float(np.count_nonzero(red & m)) / float(m.sum())
    return float(np.count_nonzero(y & m)) / float(m.sum())


def side_blobs_without_gripper(s, tool_xy, tz):
    """Side-camera blobs with the arm's points removed (within 12 cm of the tool axis and higher than 9 cm above the
    table: the bowl rim is at most 9.5 cm, a bee on the table at most 6 cm). Falls back to s.tabletop()."""
    try:
        from rlpj.perception import tabletop as tt
        P = tt.cloud(s.side, n=5)
        keep = ~((np.hypot(P[:, 0] - tool_xy[0], P[:, 1] - tool_xy[1]) < 0.12) & (P[:, 2] > tz + 0.09))   # att. 67: the open fingers (+-5 cm along the jaws) survived 0.08
        blobs = tt.detect(P[keep], s.workspace)
        s.log(f"side blobs without gripper: {len(blobs)} " + str([(np.round(b['xy'], 3).tolist(), round(float(b['height']), 3)) for b in blobs])[:300])
        return blobs
    except Exception as e:
        s.log(f"side blobs without gripper failed ({e!r}); using tabletop()")
        return s.tabletop()


def release_check(s, A, k):
    """Wrist bottom-strip colour share (gripper open) + one squeeze; returns (stuck, info). Ends with the jaws open."""
    import time
    time.sleep(0.6)                                     # fresh wrist frame after the motion settled
    img, _ = s.wrist.latest()
    save_small(os.path.join(A, f"release_check_{k}.jpg"), img)
    frac = yellow_bottom_frac(img)
    sq = s.gripper(1.0, f"squeeze test {k}")
    s.gripper(0.0, "open after squeeze")
    held = (not s.dry) and sq < SQUEEZE_HELD
    seen = (not s.dry) and frac is not None and frac > STUCK_FRAC
    info = dict(try_=k, strip_frac=None if frac is None else round(frac, 3), squeeze=round(float(sq), 1),
                strip_high=bool(seen), held=bool(held))
    s.decide(f"release check {k}: bottom-strip bee share {info['strip_frac']} (info only; bee in the bowl reads up to 0.275), "
             f"squeeze reading {info['squeeze']} (<{SQUEEZE_HELD} = holding)" + (" [dry: not judged]" if s.dry else ""))
    return bool(held), info


def blob_info(b):
    return dict(xy=[round(float(v), 3) for v in b["xy"][:2]], height=round(float(b["height"]), 3),
                extent=[round(float(v), 3) for v in b["extent"]], n=int(b["n"]),
                outer_r=b.get("outer_r"))


def in_range(v, r):
    return r[0] <= v <= r[1]


def classify(blobs):
    """-> (bowl, bee, others). Bowl: tall (>5.5 cm) and wide (>11 cm); bee: lower and smaller; inside the task area."""
    cand = [b for b in blobs if in_range(b["xy"][0], TASK_X) and in_range(b["xy"][1], TASK_Y)]
    bowls = [b for b in cand if in_range(b["height"], BOWL_H) and in_range(max(b["extent"]), BOWL_EXT) and b["n"] >= 2500]
    bowl = max(bowls, key=lambda b: b["n"]) if bowls else None
    rest = [b for b in cand if b is not bowl]
    bees = [b for b in rest if in_range(b["height"], BEE_H) and in_range(max(b["extent"]), BEE_EXT) and b["n"] >= MIN_BEE_PTS]
    bee = max(bees, key=lambda b: b["n"]) if bees else None
    others = [b for b in rest if b is not bee]
    return bowl, bee, others


def bee_pose(b, tz):
    """Centre and long-axis angle of the bee from its upper points (PCA in xy; bbox centre in the PCA frame)."""
    P = np.asarray(b["points"], float)
    h = P[:, 2] - tz
    U = P[h > 0.35 * float(b["height"])][:, :2]
    if len(U) < 30:
        U = P[:, :2]
    m = U.mean(0)
    w, V = np.linalg.eigh(np.cov((U - m).T))
    ax = V[:, 1]                                      # long axis
    perp = np.array([-ax[1], ax[0]])
    a, c = (U - m) @ ax, (U - m) @ perp
    a0, a1 = np.percentile(a, [2, 98]); c0, c1 = np.percentile(c, [2, 98])
    cen = m + ax * 0.5 * (a0 + a1) + perp * 0.5 * (c0 + c1)
    return cen, float(np.degrees(np.arctan2(ax[1], ax[0]))), float(a1 - a0), float(c1 - c0)


def bowl_circle(b, tz):
    """Rim circle from the bowl's upper points (top 25 % of its height): least-squares circle, radius bounded to
    0.060-0.082 m (bowl ~15 cm across). The ring-test centre moved 2.6 cm between two looks in attempt 1."""
    from scipy.optimize import least_squares
    P = np.asarray(b["points"], float)
    h = P[:, 2] - tz
    U = P[h > 0.75 * float(b["height"])][:, :2]
    if len(U) < 50:
        return np.array(b["xy"][:2], float), None, 0
    c0 = np.array(b["xy"][:2], float)
    def res(p):
        return np.hypot(U[:, 0] - p[0], U[:, 1] - p[1]) - p[2]
    sol = least_squares(res, [c0[0], c0[1], 0.072], bounds=([c0[0] - 0.06, c0[1] - 0.06, 0.060], [c0[0] + 0.06, c0[1] + 0.06, 0.082]),
                        loss="soft_l1", f_scale=0.004)
    return sol.x[:2], float(sol.x[2]), len(U)


def split_bee_from_bowl(bowl, tz, cxy, R):
    """Attempt 21: the bee lay against the bowl and side depth merged them into one blob (extent 0.24 m).
    Remove the bowl (points within R + 18 mm of the rim centre, or above bee height), keep the largest
    grid-connected rest as the bee if it is bee-sized. Returns a blob dict or None."""
    from scipy import ndimage
    P = np.asarray(bowl["points"], float)
    h = P[:, 2] - tz
    d = np.hypot(P[:, 0] - cxy[0], P[:, 1] - cxy[1])
    Q = P[(d > R + 0.018) & (h < BEE_H[1])]
    if len(Q) < SPLIT_BEE_PTS:
        return None
    lo = Q[:, :2].min(0) - 0.01
    ij = np.floor((Q[:, :2] - lo) / 0.005).astype(int)
    occ = np.zeros(ij.max(0) + 2, np.int32)
    np.add.at(occ, (ij[:, 0], ij[:, 1]), 1)
    lab, n = ndimage.label(ndimage.binary_closing(occ > 1, iterations=1))
    if n == 0:
        return None
    pl = lab[ij[:, 0], ij[:, 1]]
    k = np.bincount(pl[pl > 0]).argmax()
    B = Q[pl == k]
    if len(B) < SPLIT_BEE_PTS:
        return None
    z_top = float(np.percentile(B[:, 2], 98))
    ext = np.percentile(B[:, :2], 98, 0) - np.percentile(B[:, :2], 2, 0)
    b = dict(xy=B[:, :2].mean(0).tolist(), z_top=z_top, height=z_top - tz, extent=ext.tolist(), n=int(len(B)), points=B,
             split=True)
    if not (in_range(b["height"], BEE_H) and in_range(max(ext), BEE_EXT)):
        return None
    return b


def grasp_yaw(bee, tz, bee_xy, long_deg, bowl_xy, bowl_r, cur_yaw):
    """Jaw yaw across the bee whose open fingers (50 mm from the centre, 22 mm wide) stay outside the bowl rim
    (radius >= 75 mm + 18 mm) and whose bee width along the jaws is < 80 mm; perpendicular to the long axis first.
    Returns (yaw, width, finger clearance) or None."""
    P = np.asarray(bee["points"], float)
    U = P[P[:, 2] - tz > 0.35 * float(bee["height"])][:, :2]
    if len(U) < 30:
        U = P[:, :2]
    r_keep = max(bowl_r, 0.075) + 0.018   # + split-off bee centre reads ~1 cm away from the bowl (offline test)
    for off in (0, 15, -15, 30, -30, 45, -45):
        y = long_deg + 90.0 + off
        u = np.array([np.cos(np.radians(y)), np.sin(np.radians(y))]); v = np.array([-u[1], u[0]])
        a = (U - bee_xy) @ u
        width = float(np.percentile(a, 98) - np.percentile(a, 2))
        fing = [bee_xy + sx * 0.050 * u + sy * 0.011 * v for sx in (-1, 1) for sy in (-1, 1)]
        clear = min(float(np.linalg.norm(f - bowl_xy)) for f in fing) - r_keep
        if width < 0.080 and clear > 0:
            return pick_yaw(y - 90.0, cur_yaw), width, clear
    return None


def bee_top_centre(b, tz, long_deg):
    """Centre of the bee from its top points only (h > 0.7 of its height), bbox centre in the long-axis frame."""
    P = np.asarray(b["points"], float)
    U = P[P[:, 2] - tz > 0.7 * float(b["height"])][:, :2]
    if len(U) < 30:
        return None
    ax = np.array([np.cos(np.radians(long_deg)), np.sin(np.radians(long_deg))]); perp = np.array([-ax[1], ax[0]])
    m = U.mean(0)
    a, c = (U - m) @ ax, (U - m) @ perp
    return m + ax * 0.5 * sum(np.percentile(a, [5, 95])) + perp * 0.5 * sum(np.percentile(c, [5, 95]))


def pick_yaw(long_deg, cur_yaw):
    """Jaw axis (tool x) across the bee's long axis; of the two equivalent yaws take the one nearest the current."""
    best = None
    for y in (long_deg + 90.0, long_deg - 90.0, long_deg + 270.0, long_deg - 270.0):
        d = (y - cur_yaw + 180) % 360 - 180
        if best is None or abs(d) < abs(best[1]):
            best = (cur_yaw + d, d)
    return best[0]


def tool_yaw(s):
    T = s.kin.fk(s.feedback()["q"])
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


# ---- keep-in routing (operator 2026-11-07: robot.keep_in y -0.45..+0.30; joint moves home <-> look_bowl swung the
# tool out to y 0.29-0.30 at 0.43 m). Home <-> workspace goes through a fixed READY pose over the table centre
# (tool down, jaw yaw 90 deg = the j4 ~ 0 / j6 ~ 90 family next to home); everything else is straight tool lines.
READY_XY = (0.50, 0.0)
READY_YAW = 90.0
Q_READY_SEED = [0.2, 27.7, 287.4, 0.0, -79.7, 90.2]   # offline: joint move home -> READY max gripper y 0.052 m, 13.7 s


def to_ready(s, tz):
    """From home: one joint move to READY (fixed, checked path inside the cell)."""
    T = tool_down_T([READY_XY[0], READY_XY[1], tz + LOOK_Z], READY_YAW)
    try:
        q, _, _ = s.planner.ik_exact(T, np.array(Q_READY_SEED, float))
    except RuntimeError as e:
        raise Abort(f"to_ready: {e}")
    s.move_joint("to_ready", q, "normal")


def travel(s, name, T, speed="normal", tip_margin=0.008):
    """Straight tool line; if the line cannot be planned (IK jump on a large rotation over a short distance), a joint
    move to the same goal. Model-check refusals (keep-in, clearance) are not retried."""
    try:
        line(s, name, T, speed, tip_margin=tip_margin)
    except Abort as e:
        msg = str(e)
        if "joint jump" not in msg and "IK failed" not in msg:
            raise
        s.decide(f"{name}: straight line not plannable ({msg[:70]}); joint move to the same goal")
        goto_joint(s, name + "_joint", T, speed)


def go_home_via_ready(s, tz):
    """Straight line to READY, then the fixed joint move READY -> home (reverse of to_ready)."""
    T = tool_down_T([READY_XY[0], READY_XY[1], tz + LOOK_Z], READY_YAW)
    travel(s, "to_ready_back", T, "normal")
    try:
        q_r, _, _ = s.planner.ik_exact(T, np.array(Q_READY_SEED, float))
    except RuntimeError as e:
        raise Abort(f"ready: {e}")
    from rlpj.robot.kin import unwrap_near
    q_now = np.asarray(s.feedback()["q"], float)
    if np.abs(unwrap_near(q_now, q_r) - q_r).max() > 3.0:
        # a DAgger start can leave the arm in the other wrist family (j4 ~ 180): switch over the table centre, where
        # only the wrist turns, instead of on the way home (that swing left the keep-in box in offline tests)
        s.decide("ready: wrist in the other joint family; switch it here over the table centre before going home")
        s.move_joint("ready_wrist_switch", q_r, "normal")
    s.move_joint("home", s.q_home, "normal")


def goto_joint(s, name, T, speed="normal"):
    """IK (oracle-corrected) from the current joints, then a checked joint move."""
    q0 = s.feedback()["q"]
    try:
        q1, _, info = s.planner.ik_exact(T, q0)
    except RuntimeError as e:
        raise Abort(f"{name}: {e}")
    s.move_joint(name, q1, speed)


def line(s, name, T, speed="slow", tip_margin=0.008):
    try:
        s.move_line(name, T, speed, tip_margin=tip_margin)
    except RuntimeError as e:
        raise Abort(f"{name}: {e}")


def main():
    with RobotSession(program="execute") as s:
        A = s.A
        s.summary(claimed_outcome="fail", first_failed_check=None, objects={}, verify={})
        fail = {"check": None}

        def require(cond, msg):
            if not cond and fail["check"] is None:
                fail["check"] = msg
                s.summary(first_failed_check=msg)
            s.require(cond, msg)

        try:
            run(s, A, require, fail)
        except Abort as e:
            if fail["check"] is None:
                s.summary(first_failed_check=str(e)[:200])
            raise


def bowl_moved(bowl, bowl2, bowl2_xy, alt_bowl_xy, bowl_xy, tz):
    """Smallest distance between any before-estimate of the bowl centre (rim fit, ring test, the centre used) and any
    after-estimate (rim fit, ring test). Att. 254: a bad dark rim fit (588 top points, 6.7 cm off) made a bowl that had not
    moved (same place in the wrist images before/after) read 'moved 0.08 m' -> false fail."""
    before = [np.asarray(c, float) for c in list(alt_bowl_xy) + [bowl_xy, bowl_circle(bowl, tz)[0]] if c is not None]
    after = [np.asarray(c, float) for c in (bowl2_xy, bowl2["xy"][:2]) if c is not None]
    return float(min(np.linalg.norm(a - b) for a in after for b in before))


def refine_bowl(s, A, bowl_xy, rim_z, bowl_r, tag):
    """Wrist rim circle from the current pose (above the bowl); returns the (possibly) refined centre."""
    import time
    from rlpj.robot import calib
    time.sleep(0.3)
    img, _ = s.wrist.latest()
    save_small(os.path.join(A, f"wrist_bowl_{tag}.jpg"), img)
    ref = None
    refine_bowl.v_median = None
    if not s.dry and img is not None:
        T_now = calib.pose_to_T(s.feedback()["pose"])
        ref, info = wr.circle_refine(img, T_now, bowl_xy, rim_z, bowl_r)
        s.summary(wrist_bowl=info)
        bowl_disc_frac.v_median = bowl_disc_frac.red_frac = None
        bowl_disc_frac(img, T_now, bowl_xy, rim_z, bowl_r)
        refine_bowl.v_median = bowl_disc_frac.v_median
        if refine_bowl.v_median is not None:
            s.log(f"wrist brightness inside the rim ({tag}): median V {refine_bowl.v_median:.0f}")
    if ref is not None and refine_bowl.v_median is not None and refine_bowl.v_median < DARK_V:
        c_, rp_ = info.get("circle"), info.get("r_px")
        ratio = None if not (c_ and rp_) else float(c_[2]) / float(rp_)
        if ratio is None or not (0.95 <= ratio <= 1.15):
            s.decide(f"dark room: wrist rim circle {np.round(ref, 3).tolist()} rejected (radius ratio {ratio if ratio is None else round(ratio, 3)} "
                     f"outside 0.95-1.15; att. 209: ratio 1.195 put the bowl 5 cm off)")
            ref = None
    if ref is not None and np.linalg.norm(ref - bowl_xy) < 0.035:
        s.decide(f"wrist rim circle moves the bowl centre {bowl_xy.round(3).tolist()} -> {np.round(ref, 3).tolist()}")
        return np.asarray(ref, float)
    s.decide(f"wrist rim refinement not used ({'dry run' if s.dry else ref}); keep the side estimate")
    return bowl_xy


# ---- release zone (operator 2026-11-10: the library refuses to open on a held bee, or with the bee colour at the bottom
# of the wrist image, outside the task release_zone or above 0.20 m; a correction opened at 0.27 m with the bee stuck on
# the closed fingers and it fell off the table). Every open of a possibly loaded gripper happens inside the zone, low.
ZONE_DEFAULT = dict(x=[0.36, 0.60], y=[-0.22, 0.08], max_height=0.20)
OPEN_EPS = 3.0           # gripper reading at or below this = already open (0.4 when open)
DROP_H = PRE_Z           # tool height above the table for putting a stuck bee down (<= max_height)


def zone(s):
    z = getattr(s, "release_zone", None)
    return z if isinstance(z, dict) and "x" in z and "y" in z else ZONE_DEFAULT


def clamp_zone(s, xy, margin=0.005):
    z = zone(s)
    return np.array([np.clip(xy[0], z["x"][0] + margin, z["x"][1] - margin),
                     np.clip(xy[1], z["y"][0] + margin, z["y"][1] - margin)], float)


def try_open(s, what):
    """Open; False (nothing moved, gripper unchanged) if the library refuses the release here."""
    try:
        s.gripper(0.0, what)
        return True
    except Abort as e:
        if "release refused" not in str(e):
            raise
        s.decide(f"{what}: {str(e)[:220]}")
        return False


def stuck_frac(s):
    import time
    time.sleep(0.6)
    return yellow_bottom_frac(s.wrist.latest()[0])


DROP_X = (0.40, 0.54); DROP_Y = (-0.17, 0.03)   # inside the bee layout (well seen, >= 4 cm from the table edges)
DROP_C = (0.47, -0.07)


def drop_spot(s, blobs, tz):
    """Free table point: the one nearest DROP_C with >= 2 cm clearance (bowl: 7.5 cm rim + 5 cm half bee; other blobs:
    half their extent + 5 cm), first in the middle of the bee layout, then anywhere in the release zone (2 cm margin);
    else the one with the most clearance (refused by the caller if < 0)."""
    z = zone(s)
    best = (None, -9.0)
    for X, Y in ((DROP_X, DROP_Y), ((z["x"][0] + 0.02, min(z["x"][1], 0.58) - 0.02), (z["y"][0] + 0.02, z["y"][1] - 0.02))):
        cands = []
        for x in np.arange(X[0], X[1] + 1e-9, 0.01):
            for y in np.arange(Y[0], Y[1] + 1e-9, 0.01):
                xy = clamp_zone(s, (x, y), 0.02)
                c = 9.0
                for b in blobs:
                    d = float(np.hypot(xy[0] - b["xy"][0], xy[1] - b["xy"][1]))
                    r = 0.075 + 0.05 if in_range(b["height"], BOWL_H) else 0.5 * max(b["extent"]) + 0.05
                    c = min(c, d - r)
                cands.append((xy, c))
        ok = [k for k in cands if k[1] >= 0.02]
        if ok:
            return min(ok, key=lambda k: np.hypot(k[0][0] - DROP_C[0], k[0][1] - DROP_C[1]))
        m = max(cands, key=lambda k: k[1])
        best = m if m[1] > best[1] else best
    return best


def put_down_stuck(s, blobs, tz, T_now):
    """The bee is on the (not opened) fingers: rise to the carry height, go level to a free spot in the release zone,
    descend slowly to DROP_H and open there; up to two close/open shakes if it still shows on the fingers. Keeps the
    current tool orientation. Returns (spot, still_stuck)."""
    spot, clear = drop_spot(s, blobs, tz)
    if spot is None or clear < 0.0:
        raise Abort(f"bee on the fingers and no free spot in the release zone (best clearance {clear:.3f} m): not opening")
    s.decide(f"bee on the fingers: put it down at {spot.round(3).tolist()} (clearance {clear:.3f} m), tool {DROP_H} m above the table")
    T = np.array(T_now, float)
    if T[2, 3] < tz + CARRY_Z - 0.005:
        T[2, 3] = tz + CARRY_Z
        line(s, "stuck_rise", T, "normal", tip_margin=-0.01)   # straight up: open space
    T[:2, 3] = spot
    line(s, "stuck_to_spot", T, "normal")
    T[2, 3] = tz + DROP_H + 0.05
    line(s, "stuck_down", T, "normal")
    T[2, 3] = tz + DROP_H
    line(s, "stuck_lower", T, "slow")                           # last 5 cm slow
    if not try_open(s, "put the stuck bee down"):
        raise Abort("release refused even inside the release zone: bee left on the fingers")
    f = stuck_frac(s)
    for k in range(2):
        if f is None or f <= STUCK_FRAC:
            break
        s.decide(f"bee still on the fingers (bottom-strip share {f:.3f}): close/open shake {k + 1}")
        s.gripper(1.0, "stuck: close"); s.gripper(0.0, "stuck: open")
        f = stuck_frac(s)
    T[2, 3] = tz + CARRY_Z
    line(s, "stuck_up", T, "slow")
    stuck = f is not None and f > STUCK_FRAC
    s.summary(stuck_bee_put_down=dict(xy=spot.round(3).tolist(), strip_after=None if f is None else round(f, 3), still_stuck=stuck))
    return spot, stuck


def dagger_recover(s):
    """RLPJ_DAGGER=1 (operator 2026-11-06): the arm starts at the policy's last pose, maybe low beside or over an
    object. Open the gripper if it is not open (the library refuses that with the bee on the fingers outside the release
    zone / high: then it stays closed and the bee is put down after the side scan), then rise straight up (same
    orientation) to the hover height. Returns True if the bee is still on the fingers."""
    from rlpj.robot import calib
    s.phase("dagger_recover")
    fb = s.feedback()
    g0 = float(fb["gripper"])
    s.summary(dagger=dict(start_q=[round(float(v), 2) for v in fb["q"]], start_tool=[round(float(v), 4) for v in fb["pose"][:3]],
                          start_gripper=round(g0, 1)))
    stuck = False
    if g0 > OPEN_EPS:
        s.decide(f"dagger: gripper reading {g0:.1f}: open it")
        stuck = not try_open(s, "dagger: open")
    T0 = calib.pose_to_T(fb["pose"])
    z_hover = s.table_z + PRE_Z
    if T0[2, 3] < z_hover - 0.005:
        T1 = T0.copy(); T1[2, 3] = z_hover
        s.decide(f"dagger: rise straight up from {T0[2, 3] - s.table_z:.3f} m to the hover height {PRE_Z} m above the table")
        line(s, "dagger_lift", T1, "slow", tip_margin=-0.01)
    else:
        s.decide(f"dagger: tool already {T0[2, 3] - s.table_z:.3f} m above the table (hover {PRE_Z} m): no lift")
    return stuck


WRIST_FLIP = (90.0, 270.0)   # operator 2026-11-08: joint 4 must never enter 90..270 deg (converter drops the episode;
                             # policies are trained on joint angles and the normal wrist solution, joint 4 near 0)


def j4_flipped(q):
    j = np.mod(np.asarray(q, float)[..., 3], 360.0)
    return (j > WRIST_FLIP[0]) & (j < WRIST_FLIP[1])


def install_wrist_guard(s):
    """Refuse (Abort, before execution) any segment whose joint 4 enters 90..270 deg, if the episode started in the
    normal wrist configuration. The guard switches itself off when it fires so the session's return home is not blocked."""
    q0 = np.asarray(s.feedback()["q"], float)
    if bool(j4_flipped(q0)):
        s.decide(f"start joint 4 = {q0[3]:.1f} deg is already in the flipped wrist range: wrist guard not armed")
        return
    orig = s._execute
    state = {"on": True}

    def guarded(seg, *a, **k):
        if state["on"]:
            Q = np.asarray(seg.q, float)
            if bool(np.any(j4_flipped(Q))):
                state["on"] = False
                j = np.mod(Q[:, 3], 360.0)
                raise Abort(f"segment {seg.name} would turn joint 4 into the flipped wrist range "
                            f"({float(j[j4_flipped(Q)][0]):.1f} deg; allowed outside {WRIST_FLIP[0]:.0f}..{WRIST_FLIP[1]:.0f})")
        return orig(seg, *a, **k)
    s._execute = guarded
    s.summary(wrist_guard=dict(start_j4=round(float(q0[3]), 2)))


def run(s, A, require, fail):
    if True:
        dagger = os.environ.get("RLPJ_DAGGER") == "1"
        install_wrist_guard(s)
        stuck = False
        if dagger:
            stuck = dagger_recover(s)
        else:   # operator 2026-11-10: open here at home - at the look pose (0.32 m) the bee/bowl in the bottom of the wrist
            # image makes the library refuse the open (start readings 31.6 are common)
            g0 = float(np.nan_to_num(s.feedback()["gripper"]))
            if g0 > OPEN_EPS:
                s.decide(f"gripper reading {g0:.1f} at the start: open it at home")
                stuck = not try_open(s, "open at home")
        # ------------------------------------------------------------ perceive
        s.phase("perceive")
        color, _, _ = s.side.latest()
        save_small(os.path.join(A, "side_before.jpg"), color)
        if dagger:   # the arm may hang over the table: drop its points before looking for blobs
            blobs = side_blobs_without_gripper(s, np.asarray(s.feedback()["pose"][:2], float), s.table_z)
        else:
            blobs = s.tabletop()
        tz_fit = None
        for b in blobs:
            tz_fit = float(b["z_top"] - b["height"]); break
        require(tz_fit is not None, "no objects on the table")
        if stuck:
            from rlpj.robot import calib
            if not dagger:
                to_ready(s, tz_fit)
            spot, still = put_down_stuck(s, blobs, tz_fit, calib.pose_to_T(s.feedback()["pose"]))
            require(not still, f"bee still on the fingers after opening at {spot.round(3).tolist()} inside the release zone")
            if not dagger:
                go_home_via_ready(s, tz_fit)
                require(False, f"bee was on the fingers at the start: put down at {spot.round(3).tolist()} (no demo this episode)")
            blobs = side_blobs_without_gripper(s, spot, s.table_z)
            tz_fit = None
            for b in blobs:
                tz_fit = float(b["z_top"] - b["height"]); break
            require(tz_fit is not None, "no objects on the table after putting the stuck bee down")
        bowl, bee, others = classify(blobs)
        try:                                            # att. 270/272: keep the side blobs for offline analysis of refusals
            np.savez_compressed(os.path.join(A, "side_blobs.npz"),
                                **{f"blob{i}": np.asarray(b["points"], np.float32) for i, b in enumerate(blobs) if b.get("points") is not None},
                                table_z=np.float32(tz_fit))
        except Exception as e:
            s.log(f"side blob save failed: {e!r}")
        s.summary(objects=dict(bowl=blob_info(bowl) if bowl else None, bee=blob_info(bee) if bee else None,
                               others=[blob_info(b) for b in others][:4], table_z=round(tz_fit, 4)))
        require(bowl is not None, f"no bowl-sized blob (blobs: {[blob_info(b) for b in blobs]})")
        tz = tz_fit
        if bee is None:
            cxy0, cr0, _ = bowl_circle(bowl, tz)
            bee = split_bee_from_bowl(bowl, tz, cxy0, float(np.clip(cr0 or 0.072, 0.06, 0.082)))
            if bee is not None:
                s.decide(f"bee merged with the bowl blob (extent {bowl['extent']}): split off {bee['n']} points at "
                         f"{np.round(bee['xy'], 3).tolist()} h {bee['height']:.3f}")
                s.summary(bee_split_from_bowl=True)
                s.summary(objects=dict(bowl=blob_info(bowl), bee=blob_info(bee), others=[blob_info(b) for b in others][:4],
                                       table_z=round(tz_fit, 4)))
        require(bee is not None, f"no bee-sized blob (blobs: {[blob_info(b) for b in blobs]})")
        require(abs(tz - s.table_z) < 0.012, f"fitted table z {tz:.4f} far from calibration {s.table_z:.4f}")
        bowl_xy = np.array(bowl["xy"][:2], float)
        alt_bowl_xy = [bowl_xy.copy()]                  # other bowl centre estimates for the release confirmation (att. 209)
        rim_z = float(bowl["z_top"])
        cxy, cr, nU = bowl_circle(bowl, tz)
        s.decide(f"bowl rim circle fit: centre {np.round(cxy, 3).tolist()} r {cr} from {nU} top points (ring test {bowl_xy.round(3).tolist()})")
        if cr is not None:
            ring_xy = bowl_xy.copy()
            bowl_xy = np.asarray(cxy, float)
            alt_bowl_xy.append(bowl_xy.copy())
            dfr = float(np.linalg.norm(bowl_xy - ring_xy))
            if dfr > 0.03 and bee.get("split"):
                # att. 234: the ring test still holds the split-off bee points (pulled 4.8 cm toward the bee; the fit was 0.4 cm
                # from the bowl found after the place) - keep the rim fit
                s.decide(f"rim fit and ring test {dfr * 100:.1f} cm apart, but the bee was split from the bowl blob: keep the rim fit")
            elif dfr > 0.03:
                # 39 runs with |fit - ring| > 3 cm, judged by the wrist rim: midpoint error 3-14 mm, fit 6-33 mm, ring 10-36 mm
                # (att. 209/221 in the dark: fit 2.7/4.0 cm off, midpoint 0.7/0.9 cm off by back-projection)
                bowl_xy = (bowl_xy + ring_xy) / 2.0
                s.decide(f"rim fit and ring test {dfr * 100:.1f} cm apart: using their midpoint {np.round(bowl_xy, 3).tolist()}")
        bowl_r = float(np.clip(cr or 0.072, 0.06, 0.082))
        bee_xy, long_deg, bee_len, bee_wid = bee_pose(bee, tz)
        if bee_len > BEE_LEN_SMEAR or bee.get("split") or max(bee["extent"]) > 0.12:
            # att. 79: a split/smeared blob read 0.145 m long and the grasp landed ~3 cm beside the bee (closed on
            # nothing). The depth smear trails behind the bee at lower heights: centre the grasp on the top points.
            # att. 168: the smear also made the blob nearly square (len 0.095, wid 0.104 -> refused): axis and width
            # from the top third of the bee only.
            P = np.asarray(bee["points"], float)
            upper = P[P[:, 2] - tz > 0.65 * float(bee["height"])]   # offline: 0.5 still held the smear (axis off 40-60 deg), 0.65-0.7 within 6 deg
            if len(upper) >= 60:
                bee = dict(bee, points=upper)
                _, long_deg, bee_len, bee_wid = bee_pose(bee, tz)
                s.decide(f"smeared bee: axis/width from its top third: long axis {long_deg:.0f} deg, len {bee_len:.3f}, wid {bee_wid:.3f}")
            top_xy = bee_top_centre(bee, tz, long_deg)
            if top_xy is not None:
                s.decide(f"bee blob smeared (len {bee_len:.3f}, extent {np.round(bee['extent'], 3).tolist()}, split {bool(bee.get('split'))}): grasp centre from its top points "
                         f"{np.round(bee_xy, 3).tolist()} -> {np.round(top_xy, 3).tolist()}")
                bee_xy = top_xy
        bee_h = float(bee["height"])
        s.decide(f"bowl at {bowl_xy.round(3).tolist()} rim z {rim_z:.3f} r {bowl_r:.3f}; bee at {bee_xy.round(3).tolist()} "
                 f"h {bee_h:.3f} len {bee_len:.3f} wid {bee_wid:.3f} long axis {long_deg:.0f} deg; table z {tz:.4f}")
        require(bee_wid < 0.080, f"bee too wide across the jaws ({bee_wid:.3f} m)")
        s.view.set_scene(objects=[dict(xy=bowl_xy.tolist(), z_top=rim_z, r=bowl_r, label="bowl"),
                                  dict(xy=bee_xy.tolist(), z_top=float(bee["z_top"]), r=bee_len / 2, label="bee")])

        # ------------------------------------------------------------ plan (refine the bowl centre with the wrist camera)
        s.phase("plan")
        yaw = pick_yaw(long_deg, tool_yaw(s))
        s.decide(f"grasp across the bee's short axis: jaw yaw {yaw:.0f} deg, tool {GRASP_Z * 1000:.0f} mm above table "
                 f"(closed tips ~14 mm up, bee {bee_h * 1000:.0f} mm tall), full open 85 mm vs bee width {bee_wid * 1000:.0f} mm")
        if not dagger:
            to_ready(s, tz)
            travel(s, "look_bowl", tool_down_T([bowl_xy[0], bowl_xy[1], tz + LOOK_Z], yaw), "normal")
            bowl_xy = refine_bowl(s, A, bowl_xy, rim_z, bowl_r, "before")
            vb = refine_bowl.v_median
            require(vb is None or vb >= DARK_MIN_V,
                    f"wrist image too dark even for the red test (median V {vb if vb is None else round(vb)} < {DARK_MIN_V})")
            if vb is not None and vb < DARK_V:
                r0 = bowl_disc_frac.red_frac
                s.decide(f"room dark (wrist median V {vb:.0f} < {DARK_V}; normal 165-211): the in-bowl check will use red dominance "
                         f"(> {DARK_RED_FRAC}) instead of the colour share; empty bowl reads {r0} now")
                s.summary(dark_mode=dict(v_median=round(vb, 1), empty_red_frac=r0))
                require(r0 is not None and r0 < DARK_RED_FRAC / 2,
                        f"dark mode: the empty bowl already reads red share {r0} (>= {DARK_RED_FRAC / 2}); cannot confirm a release")
        gy = grasp_yaw(bee, tz, bee_xy, long_deg, bowl_xy, bowl_r, yaw)
        require(gy is not None, f"no grasp yaw keeps the open fingers outside the bowl rim (bee {np.linalg.norm(bee_xy - bowl_xy):.3f} m from the bowl centre)")
        yaw, gwid, gclear = gy
        s.decide(f"grasp yaw {yaw:.0f} deg: bee width along jaws {gwid * 1000:.0f} mm, open-finger clearance to the rim {gclear * 1000:.0f} mm")
        s.summary(grasp=dict(yaw=round(yaw, 1), width=round(gwid, 3), finger_clearance=round(gclear, 3)))
        rel_z = rim_z + RELEASE_ABOVE_RIM + BEE_BELOW_TOOL
        s.decide(f"release over bowl centre {bowl_xy.round(3).tolist()} at tool z {rel_z:.3f} (bee bottom {RELEASE_ABOVE_RIM * 1000:.0f} mm over the rim)")
        zn = zone(s)   # the library refuses to open a held bee outside the release zone / above 0.20 m: check before the grasp
        require(np.linalg.norm(clamp_zone(s, bowl_xy, 0.0) - bowl_xy) < 0.01,
                f"bowl centre {bowl_xy.round(3).tolist()} outside the release zone x {zn['x']} y {zn['y']}")
        require(rel_z - tz <= zn.get("max_height", 0.20) - 0.01, f"release height {rel_z - tz:.3f} m above the release zone limit")

        # ------------------------------------------------------------ approach
        s.phase("approach")
        s.gripper(0.0, "open fully")
        T_above = tool_down_T([bee_xy[0], bee_xy[1], tz + PRE_Z], yaw)
        if dagger:   # straight, level move at hover height from where the policy left the arm (no detour)
            T_above[2, 3] = max(tz + PRE_Z, float(s.feedback()["pose"][2]))
            try:
                line(s, "above_bee", T_above, "normal", tip_margin=-0.01)
            except Abort as ex:   # planning/model refusal happens before any motion: fall back to a joint move
                s.decide(f"dagger: straight move refused ({str(ex)[:80]}); joint move instead")
                goto_joint(s, "above_bee_joint", T_above, "normal")
            line(s, "to_pre_grasp", tool_down_T([bee_xy[0], bee_xy[1], tz + PRE_Z], yaw), "slow")
        else:
            travel(s, "above_bee", T_above, "normal")
        line(s, "descend_bee", tool_down_T([bee_xy[0], bee_xy[1], tz + GRASP_Z], yaw), "slow")

        # ------------------------------------------------------------ grasp
        s.phase("grasp")
        g = s.gripper(1.0, "close on the bee")
        s.summary(grip_reading=round(float(g), 1))
        if not s.dry:
            require(g < CLOSED_ON_NOTHING, f"gripper closed on nothing (reading {g:.1f})")
        s.decide(f"gripper reading {g:.1f} after closing")

        # ------------------------------------------------------------ lift
        s.phase("lift")
        line(s, "lift", tool_down_T([bee_xy[0], bee_xy[1], tz + PRE_Z], yaw), "slow", tip_margin=LIFT_TIP_MARGIN)
        line(s, "lift_high", tool_down_T([bee_xy[0], bee_xy[1], tz + CARRY_Z], yaw), "normal")
        g2 = float(s.feedback()["gripper"])
        s.summary(grip_after_lift=round(g2, 1))
        if not s.dry:
            require(g2 < CLOSED_ON_NOTHING, f"bee lost during lift (reading {g2:.1f})")

        # ------------------------------------------------------------ place
        s.phase("place")
        line(s, "carry", tool_down_T([bowl_xy[0], bowl_xy[1], tz + CARRY_Z], yaw), "normal")
        if dagger:   # bowl centre from the wrist camera at the carry pose (no separate look before the grasp)
            new_xy = refine_bowl(s, A, bowl_xy, rim_z, bowl_r, "carry")
            if np.linalg.norm(new_xy - bowl_xy) > 0.003:
                bowl_xy = new_xy
                line(s, "carry_adjust", tool_down_T([bowl_xy[0], bowl_xy[1], tz + CARRY_Z], yaw), "slow")
        rel_xy = clamp_zone(s, bowl_xy)   # (<= 1.5 cm shift, inside the rim) so the release cannot be refused
        line(s, "lower_into_bowl", tool_down_T([rel_xy[0], rel_xy[1], rel_z], yaw), "slow")
        g3 = float(s.feedback()["gripper"])
        if not s.dry:
            require(g3 < CLOSED_ON_NOTHING, f"bee lost during carry (reading {g3:.1f})")
        gr = s.gripper(0.0, "release over the bowl")
        s.summary(release_reading=round(float(gr), 1))
        # Operator notes 2026-11-05: the bee can stay between or on the outside of a fingertip after opening. The
        # library refuses to open on a held bee above 0.20 m / outside the release zone, so every squeeze/open below
        # happens over the bowl at most CHECK_H above the table. Never go home with the bee unconfirmed.
        chk_z = min(rel_z + 0.04, tz + CHECK_H)
        line(s, "retreat", tool_down_T([rel_xy[0], rel_xy[1], chk_z], yaw), "slow")
        s.phase("release_check")
        checks = []
        side = np.array([-np.sin(np.radians(yaw)), np.cos(np.radians(yaw))])   # tool y: across the jaw axis
        toward = -bowl_xy / max(np.linalg.norm(bowl_xy), 1e-6)                # toward the robot base axis
        cur = rel_xy.copy()
        confirmed, img, dark_msg = False, None, None
        for k in range(RELEASE_TRIES + 1):
            if k > 0:
                s.decide(f"bee not confirmed in the bowl: try {k} - lower over the bowl, close/open, shift 2 cm sideways, rise")
                line(s, f"relower_{k}", tool_down_T([rel_xy[0], rel_xy[1], rel_z], yaw), "slow")
                s.gripper(1.0, "shake: close"); s.gripper(0.0, "shake: open")
                cur = clamp_zone(s, rel_xy + (0.02 if k % 2 else -0.02) * side)
                line(s, f"shift_{k}", tool_down_T([cur[0], cur[1], rel_z], yaw), "slow")
                line(s, f"rise_{k}", tool_down_T([cur[0], cur[1], chk_z], yaw), "slow")
            # (a) squeeze + wrist bottom strip (bee between the jaws)
            held, info = release_check(s, A, k)
            info["held_or_strip"] = bool(held)
            # (b) side camera from a pose 8 cm higher and 8 cm toward the base: bee left on the table?
            vp = cur + SIDE_VIEW_BACK * toward   # att. 64: at 8 cm the gripper merged with the bowl blob (z_top 0.279)
            line(s, f"side_view_{k}", tool_down_T([vp[0], vp[1], chk_z + 0.08], yaw), "normal")   # moving up and away: open space
            bl = side_blobs_without_gripper(s, vp, tz)   # att. 64/65: the gripper merged with / hid the bowl blob
            tb = [b for b in bl if in_range(b["height"], BEE_H) and in_range(max(b["extent"]), BEE_EXT)
                  and b["n"] >= MIN_BEE_PTS and np.linalg.norm(np.subtract(b["xy"][:2], bowl_xy)) > bowl_r + 0.02]
            bw, _, _ = classify(bl)
            info.update(side_table_bee=[blob_info(b) for b in tb][:2],
                        side_bowl_height=None if bw is None else round(float(bw["height"]), 3))
            # (c) wrist camera straight above the bowl: the bee inside the rim
            travel(s, f"look_bowl_after_{k}", tool_down_T([bowl_xy[0], bowl_xy[1], tz + LOOK_Z], yaw), "normal")
            import time
            time.sleep(0.6)
            img, _ = s.wrist.latest()
            save_small(os.path.join(A, f"wrist_bowl_after_{k}.jpg"), img)
            df, vmed, rf = None, None, None
            bowl_disc_frac.v_median = bowl_disc_frac.red_frac = None
            if img is not None:
                from rlpj.robot import calib
                df = bowl_disc_frac(img, calib.pose_to_T(s.feedback()["pose"]), bowl_xy, rim_z, bowl_r)
                vmed, rf = bowl_disc_frac.v_median, bowl_disc_frac.red_frac
            alt = None                                     # best reading around the other bowl estimates (side ring test,
            if img is not None and not s.dry:             # side rim fit, side bowl blob now) - used only without a table bee
                cands = list(alt_bowl_xy) + ([np.array(bw["xy"][:2], float)] if bw is not None else [])
                for c_ in cands:
                    if np.linalg.norm(c_ - bowl_xy) < 0.01 or np.linalg.norm(c_ - bowl_xy) > 0.08:
                        continue
                    d_ = bowl_disc_frac(img, calib.pose_to_T(s.feedback()["pose"]), c_, rim_z, bowl_r)
                    if d_ is None:
                        continue
                    r_ = bowl_disc_frac.red_frac
                    if alt is None or (r_ if vmed is not None and vmed < DARK_V else d_) > (alt[2] if vmed is not None and vmed < DARK_V else alt[1]):
                        alt = (np.round(c_, 3).tolist(), round(d_, 3), round(r_, 3))
                info["alt_centre"] = alt
            info["bowl_disc_frac"] = None if df is None else round(df, 3)
            info["wrist_v_median"] = None if vmed is None else round(vmed, 1)
            info["bowl_red_frac"] = None if rf is None else round(rf, 3)
            dark = df is not None and vmed is not None and vmed < DARK_V
            checks.append(info)
            s.summary(release_checks=checks)
            if s.dry:
                s.decide("dry run: release confirmation not judged"); confirmed = True; break
            if dark and not s.dry and vmed < DARK_MIN_V:
                dark_msg = f"wrist image too dark to confirm the bee in the bowl (median V {vmed:.0f} < {DARK_MIN_V})"
                s.decide(f"confirm {k}: " + dark_msg)
                break
            if dark:             # att. 204: in the dark the empty bowl itself read 0.164 colour share; red dominance stays 0.000
                in_bowl = rf is not None and rf > DARK_RED_FRAC and not info["held"]
                s.decide(f"confirm {k} (dark, median V {vmed:.0f}): red-dominant share inside the rim {info['bowl_red_frac']} "
                         f"(>{DARK_RED_FRAC} = in bowl; colour share {info['bowl_disc_frac']} ignored)")
            else:
                in_bowl = df is not None and df > IN_BOWL_FRAC and not info["held"]
            if not in_bowl and alt is not None and not tb and not info["held"]:
                alt_ok = alt[2] > DARK_RED_FRAC if dark else alt[1] > IN_BOWL_FRAC
                if alt_ok:
                    in_bowl = True
                    s.decide(f"confirm {k}: bee seen inside the rim around another bowl estimate {alt[0]} (colour {alt[1]}, red {alt[2]}); "
                             f"no bee on the table (side camera), jaws empty -> in the bowl (att. 209: bowl pushed / centre 4 cm off)")
            s.decide(f"confirm {k}: bee share inside the rim (wrist, from above) {info['bowl_disc_frac']} (>{IN_BOWL_FRAC} = in bowl), "
                     f"jaws holding {info['held']}, side camera: {len(tb)} bee-sized blob(s) on the table, bowl height {info['side_bowl_height']}")
            if in_bowl:          # the wrist sees the bee inside the rim: a side blob elsewhere is an artefact (att. 68)
                confirmed = True
                break
            if tb:
                msg = f"bee on the table outside the bowl at {np.round(tb[0]['xy'][:2], 3).tolist()} (side camera)"
                fail["check"] = fail["check"] or msg
                s.summary(first_failed_check=msg, verify=dict(release_checks=checks, bee_released=True, bee_in_bowl=False))
                s.decide(msg + ": leaving it there, ending the run")
                s.claim("fail")
                return
        if not confirmed:
            msg = dark_msg or f"bee not confirmed in the bowl after {RELEASE_TRIES} shake-off tries (likely still on the gripper)"
            fail["check"] = fail["check"] or msg
            s.summary(first_failed_check=msg, verify=dict(release_checks=checks, bee_released=False))
            s.decide(msg + ": stopping here above the bowl (not going home with it); operator must remove it")
            s.q_home = s.feedback()["q"].copy()       # the session's final return-home becomes a no-op
            s.claim("fail")
            return
        save_small(os.path.join(A, "wrist_bowl_after.jpg"), img)

        # ------------------------------------------------------------ home + verify
        s.phase("home")
        go_home_via_ready(s, tz)
        s.phase("verify")
        color, _, _ = s.side.latest()
        save_small(os.path.join(A, "side_after.jpg"), color)
        blobs2 = s.tabletop()
        # att. 163: the bee sitting high in the bowl made the bowl read 0.099 m (> BOWL_H 0.095) -> false "bowl not
        # found". After the place, look for the bowl near where it was, allowing the bee on top (a bowl on its side
        # would read ~0.15 m, its diameter).
        near_ref = [b for b in blobs2 if b["n"] >= 2500 and in_range(b["height"], BOWL_H_AFTER)
                    and np.linalg.norm(np.subtract(b["xy"][:2], bowl_xy)) < 0.08]
        bowl2 = min(near_ref, key=lambda b: np.linalg.norm(np.subtract(b["xy"][:2], bowl_xy))) if near_ref else None
        bowl2_xy = None if bowl2 is None else bowl_circle(bowl2, tz)[0]
        near_bowl = [b for b in blobs2 if bowl2 is not None and np.linalg.norm(np.subtract(b["xy"][:2], bowl2["xy"][:2])) < 0.13]
        bee_left = [b for b in blobs2 if b is not bowl2 and b not in near_bowl and float(b["height"]) > 0.012 and b["n"] >= MIN_BEE_PTS
                    and np.linalg.norm(np.subtract(b["xy"][:2], bee_xy)) < 0.08]
        stray = [b for b in blobs2 if b is not bowl2 and b not in near_bowl and in_range(b["height"], BEE_H) and b["n"] >= MIN_BEE_PTS]
        v = dict(grip_reading=round(float(g), 1), grip_after_lift=round(g2, 1), release_reading=round(float(gr), 1), bee_released=True,
                 release_checks=len(checks), bee_in_bowl_frac=checks[-1].get("bowl_disc_frac"),
                 bowl_found=bowl2 is not None,
                 bowl_moved_m=None if bowl2 is None else round(bowl_moved(bowl, bowl2, bowl2_xy, alt_bowl_xy, bowl_xy, tz), 3),
                 bowl_height=None if bowl2 is None else round(float(bowl2["height"]), 3),
                 bee_left_at_start=len(bee_left), stray_bee_blobs=[blob_info(b) for b in stray][:3],
                 release_xy=bowl_xy.round(3).tolist(), dry=s.dry)
        s.summary(verify=v, objects_after=[blob_info(b) for b in blobs2][:5])
        ok = (bowl2 is not None and v["bowl_moved_m"] < 0.05 and in_range(v["bowl_height"], BOWL_H_AFTER)
              and not bee_left and not stray and (s.dry or (g < CLOSED_ON_NOTHING and g2 < CLOSED_ON_NOTHING and gr < 20)))
        if not ok:
            why = ("bowl not found after place" if bowl2 is None else
                   f"bowl moved {v['bowl_moved_m']} m" if v["bowl_moved_m"] >= 0.05 else
                   f"bowl height {v['bowl_height']} (tipped?)" if not in_range(v["bowl_height"], BOWL_H_AFTER) else
                   "bee still at its start spot" if bee_left else
                   "bee-sized blob outside the bowl" if stray else "gripper readings")
            fail["check"] = fail["check"] or why
            s.summary(first_failed_check=why)
        s.decide(f"verify: {json.dumps(v)[:300]}")
        s.claim("success" if ok else "fail")


if __name__ == "__main__":
    main()
