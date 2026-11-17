"""Reset for bee_in_bowl: take the bee out of the bowl if it is inside, then move bowl and bee to the targets.

usage: python programs/reset.py --targets '{"bowl": [x, y], "bee": [x, y]}'   (base frame, m)

Objects by geometry (side depth blobs): bowl = 5.5-9.5 cm tall, 11-20 cm footprint; bee = 1.5-5.8 cm tall,
3.5-13 cm footprint, >= 250 points. No bee blob next to the bowl -> the bee is in the bowl (confirmed in the wrist
image from above the bowl). The bowl is moved by pinching its rim (jaw axis radial, one finger in, one out).
Displacements use side-camera centres (the orchestrator measures with the side camera)."""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlpj.robot import calib                                    # noqa: E402
from rlpj.robot.kin import wrap180                              # noqa: E402
from rlpj.robot.plan import tool_down_T                         # noqa: E402
from rlpj.robot.session import RobotSession, Abort              # noqa: E402
from rlpj.perception import wrist as wr                         # noqa: E402
from rlpj.perception import tabletop as tt                      # noqa: E402
from rlpj.robot import kortex as _kx                            # noqa: E402


class _RetryRobot(_kx.Robot):
    """att 440/467: kortex.Robot() at session start failed with INVALID_USER_SESSION_ACCESS; the next run, seconds
    later, connected fine. Operator 19:17: the library now retries the first connection for ~10 s itself, so only one
    extra reconnect here (close the half-made connections, wait 3 s, try once more). No motion involved."""
    def __init__(self, *a, **k):
        import time as _t
        for i in range(2):
            try:
                super().__init__(*a, **k)
                return
            except Exception as e:
                for c in ("udp", "tcp"):
                    try:
                        getattr(self, c).close()
                    except Exception:
                        pass
                if i == 1:
                    raise
                print(f"robot connect failed ({e!r:.120}); one more try in 3 s", flush=True)
                _t.sleep(3)


_kx.Robot = _RetryRobot

TOL = 0.03                 # orchestrator tolerance
REDO = 0.020               # redo a bee move once when the measured (blob-centre) error is above this
REDO_BOWL = 0.024          # bowl blob centre jitters ~2 cm: a correction at 2 cm overshot to the other side (att 39)
SKIP = 0.018               # object already this close to its target: leave it (side-camera noise ~1 cm)
SKIP_BEE = SKIP            # bee: max(SKIP, 0.6 TOL) at 0.04 = 2.4 cm (att 134 re-picked a bee 2.3 cm off its target:
                           # ~60 s and one more release; the final measure + correction pass still guard the result)
HOVER_Z = 0.22             # tool z (base) for moves in open space
FAR_HOVER_Z = 0.15         # hover for objects beyond r 0.65 m
BEE_TOOL_ABOVE_TOP = 0.009 # table bee: tool 9 mm above the bee top (43 mm bee -> tool 52 mm above table; worked in execute)
BEE_HELD_MAX = 95.0        # reading below this = holding the bee (82 on hardware; 99 closed on nothing)
BOWL_HELD_MAX = 98.4       # thin plastic wall ~96.5, nothing ~99.1
RIM_R = 0.069              # rim radius where the jaw centre goes (bowl ~15 cm across)
BOWL_OPEN = 1 - 0.058 / 0.085   # 58 mm opening around the rim wall
BEE_BOWL_OPEN = 1 - 0.070 / 0.085  # 70 mm opening inside the bowl
CARRY = dict(name="carry", jv=12.0, lin=0.05)   # gentle carry for the bowl hanging from its rim
CLEAR_BOWL = 0.135         # bee centre at least this far from a bowl centre (rim 0.075 + half bee width + margin)
LOOK_UV = (621.0, 440.0)   # where the bowl centre goes in the wrist image (principal point is high: cy 238)

S = None
LOG = {}
RAW = []                   # raw tabletop blob lists (what the orchestrator's layout check sees)
OTHERS = []
TILTED_BOWL = []          # bowl-sized blobs 9.5-13 cm tall: the bowl tipped up on the bee (att 354/355: 0.105)
TILTED = []                # bee-sized blobs too tall for a flat bee (on its side or standing) from the last perception                # small blobs of the last perception (a bee the side camera sees badly)
PLACE_YAW = 0.0            # jaw along base x when putting the bee down: its long axis lies along y, across the
                           # side camera's depth smear (diagonal bees near x 0.3 read > 0.125 m wide)


def r3(v, n=3):
    return np.round(np.asarray(v, float), n).tolist()


def dist(a, b):
    return float(np.linalg.norm(np.subtract(a[:2], b[:2])))


# ------------------------------------------------------------------ kinematics helpers
def ik(xyz, yaw):
    """Tool-down pose -> joints, elbow-up family with q4 ~ 0 (seeded), q6 near 90."""
    a = np.degrees(np.arctan2(xyz[1], xyz[0]))
    yaw = float(yaw)
    best = None
    for y in (yaw, yaw + 180.0, yaw - 180.0):
        q6 = y - a
        if not -10.0 <= q6 <= 190.0:
            continue
        for seed in ([-a, 40, 257, 0, -37, q6], [-a, 25, 240, 0, -60, q6], [-a, 55, 280, 0, -45, q6]):
            try:
                q, _, info = S.planner.ik_exact(tool_down_T(xyz, y), np.array(seed, float))
            except RunTimeLimit:
                raise
            except Exception:
                continue
            if abs(wrap180(q[3])) < 40 and abs(q[5] - 90) < 110 and within_limits(q):
                cost = abs(q[5] - 90)
                if best is None or cost < best[0]:
                    best = (cost, q, y)
                break
    S.require(best is not None, f"no IK for tool-down pose {r3(xyz)} yaw {yaw:.0f}")
    return best[1], best[2]


def within_limits(q):
    """Gen3 limited joints (2: +-128, 3: +-147, 5: +-120 deg) with a margin; the model check let joint 5 reach 129 deg
    and the arm faulted (att 285)."""
    q = wrap180(np.asarray(q, float))
    return abs(q[1]) <= 124.0 and abs(q[2]) <= 143.0 and abs(q[4]) <= 116.0


def tool_T():
    return calib.pose_to_T(S.feedback()["pose"])


def hover_z(xy):
    """Open-space tool height: 0.22 m, but tool-down IK at 0.22 only reaches r ~0.69 m (operator's check), so far
    objects use 0.15 m."""
    if np.hypot(xy[0], xy[1]) <= 0.65:
        return HOVER_Z
    for z in (HOVER_Z, FAR_HOVER_Z, 0.10):     # highest tool-down height the IK reaches here
        if reach_ok(xy, z):
            return z
    return None


def reach_ok(xy, z, yaw=90.0):
    """Tool-down IK (our own ik(), all yaw options) solves at (xy, z)."""
    try:
        ik([xy[0], xy[1], z], yaw)
        return True
    except RunTimeLimit:
        raise
    except Abort:
        return False


def to_hover(name, xy, yaw, z=None, speed="normal"):
    z = (hover_z(xy) or 0.10) if z is None else z
    q, y = ik([xy[0], xy[1], z], yaw)
    S.move_joint(name, q, speed)
    return y


def _line(name, T, speed, tip_margin):
    """Straight line; when the line planner fails (IK jump), a joint move to the same pose (short, nearly straight)."""
    try:
        S.move_line(name, T, speed=speed, tip_margin=tip_margin)
    except RuntimeError as e:
        S.decide(f"{name}: line planning failed ({e}); joint move to the same pose instead")
        x, y = T[0, 3], T[1, 3]
        yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        q, _ = ik([x, y, T[2, 3]], yaw)
        sp = speed if isinstance(speed, dict) else ("slow" if speed == "slow" else "normal")
        if isinstance(sp, dict) or sp == "slow":
            sp = dict(name="slowj", jv=5.0, lin=0.02)
        from rlpj.robot.session import speed_of
        seg = S.planner.joint_move(name + "_j", S.measured_q(q), q, speed_of(sp))
        S._execute(seg, tip_margin=tip_margin)


def line_to(name, xy, z, yaw, speed="slow", tip_margin=0.008):
    _line(name, tool_down_T([xy[0], xy[1], z], yaw), speed, tip_margin)


def vertical(name, z, speed="slow", tip_margin=0.008):
    T = tool_T()
    T[2, 3] = (hover_z(T[:2, 3]) or 0.10) if z is None else z
    # keep the exact tool-down orientation of the current pose
    _line(name, T, speed, tip_margin)


# ------------------------------------------------------------------ perception
DRY_SCENE = {}


def dry_move(name, dest):
    """Dry run: nothing moves, so shift the remembered blob instead of re-perceiving."""
    b = DRY_SCENE.get(name)
    if S.dry and b is not None:
        d = np.subtract(dest, b["xy"])
        DRY_SCENE[name] = dict(b, xy=[float(dest[0]), float(dest[1])], points=b["points"] + np.r_[d, 0].astype(np.float32))


def perceive(tag):
    if S.dry and DRY_SCENE:
        bowl, bee = DRY_SCENE.get("bowl"), DRY_SCENE.get("bee")
        S.decide(f"dry run: scene for {tag} from the simulated moves: bowl {_brief(bowl)}, bee {_brief(bee)}")
        return bowl, bee
    blobs = S.tabletop()
    RAW.append(blobs)
    bowls, bees, others, tilted = [], [], [], []
    blobs = [x for b in blobs for x in split_merged(b)]
    for b in blobs:
        ext = max(b["extent"]); h = b["height"]
        if 0.062 <= h <= 0.095 and 0.11 <= ext <= 0.20 and b["n"] >= 800:   # bowl reads 0.066-0.082; a bee on its
            bowls.append(b)                                                   # side 0.056 with n ~1100 (att 184)
        elif 0.038 <= h <= 0.058 and 0.035 <= ext <= 0.17 and b["n"] >= 300 and not (h < 0.040 and b["n"] < 450):
            # att 322: the fixed clutter at (0.72,-0.27) read h 0.038 n 330 -> "bee out of reach". Phantoms (att 30,
            # 166, 306, 322): h 0.032-0.038 with n 285-330; real bees below h 0.040 always had n >= 905
            # att 306: phantom at (0.32,-0.06) h 0.036 n 302 taken for the bee (bee was in the bowl, carried with it);
            # same spot/height in att 30 (0.032) and 166 (0.036); real bees at start read h >= 0.039 (85 starts)   # att 39: real bee at (0.36,-0.19) had < 450 pts;   # real bee: n 850-1350, h 0.042-0.049;
            # phantoms/shadows near x 0.26-0.36: n 63-285, h <= 0.035 (att 30 took one for the bee)
            bees.append(b)
        else:
            others.append(b)
            if 0.052 < h <= 0.10 and 0.035 <= ext <= 0.11 and b["n"] >= 300:
                tilted.append(b)          # a bee on its side (att 183-185: 0.056-0.063) or standing on its end
                                          # (att 218: 0.086 tall, ext 0.088, after the start drop from 0.10 m)
    # att 322/604: the fixed clutter at (0.72-0.74, -0.27), r 0.77, read bee-high in some frames and aborted the run
    # as "bee out of reach". Nothing beyond the arm's reach (r > 0.70) is taken for the bee: the bee stays in the
    # workspace (the task's release zone ends at r 0.63).
    # att 700: the real bee lay at (0.71, -0.008), r 0.71, and was dropped by an r > 0.70 rule: only the fixed
    # clutter corner (x > 0.66, y < -0.23) is excluded now; reach is checked later
    far = lambda b: float(b["xy"][0]) > 0.66 and float(b["xy"][1]) < -0.23
    others += [b for b in bees if far(b)]
    bees = [b for b in bees if not far(b)]
    tilted = [b for b in tilted if not far(b)]
    TILTED[:] = tilted
    for b in others:     # low flat clutter (1-3.5 cm): remembered for the whole run, it shows in some frames only
        if 0.008 < b["height"] < 0.035 and b["n"] >= 60 and max(b["extent"]) <= 0.15 and "points" in b:
            P = np.asarray(b["points"], float)[:, :2]
            CLUTTER.append(P[:: max(1, len(P) // 150)])
    TILTED_BOWL[:] = [b for b in others if 0.095 < b["height"] <= 0.13 and 0.11 <= max(b["extent"]) <= 0.20 and b["n"] >= 800]
    OTHERS[:] = [b for b in others if b["n"] >= 50 and 0.008 < b["height"] < 0.06]
    bowl = max(bowls, key=lambda b: b["n"]) if bowls else None
    if bowl is not None:
        bees = [b for b in bees if dist(b["xy"], bowl["xy"]) > 0.08]
    bee = max(bees, key=lambda b: b["n"]) if bees else None
    if bowl is not None:      # rim circle centre: steadier than the blob centre (which jitters ~2 cm)
        f = rim_fit(bowl)
        if f and 0.055 < f["r"] < 0.085 and dist(f["xy"], bowl["xy"]) < 0.04:
            bowl = dict(bowl, xy_blob=bowl["xy"], xy=f["xy"])
    if bee is not None:       # median of all bee points: steadier than the top band
        bee = dict(bee, xy_blob=bee["xy"], xy=r3(np.median(bee["points"][:, :2], 0), 4))
    info = dict(bowl=_brief(bowl), bee=_brief(bee), others=[_brief(b) for b in others + [x for x in bowls if x is not bowl]
                                                             + [x for x in bees if x is not bee]][:4])
    if bowl is not None:
        info["bowl_fit"] = rim_fit(bowl)
        info["bowl_inside"] = inside_stats(bowl)
    if bee is not None:
        info["bee_yaw_long"] = round(long_axis(bee), 1)
    LOG[tag] = info
    if tag == "start":
        S.summary(perceive_start=info)
    with open(os.path.join(S.A, "perception.jsonl"), "a") as f:
        f.write(json.dumps(dict(tag=tag, **info), default=str) + "\n")
    objs = []
    if bowl is not None:
        objs.append(dict(xy=bowl["xy"], z_top=bowl["z_top"], r=0.075, label="bowl"))
    if bee is not None:
        objs.append(dict(xy=bee["xy"], z_top=bee["z_top"], r=0.04, label="bee"))
    S.view.set_scene(objects=objs)
    if S.dry:
        DRY_SCENE.update(bowl=bowl, bee=bee)
    return bowl, bee


def split_merged(b):
    """A bowl-tall blob wider than a bowl = bowl + an object touching its shadow: split with the rim circle."""
    if not (0.055 <= b["height"] <= 0.095 and max(b["extent"]) > 0.19):   # not lower: a lone bowl reads up to 0.185 (att 135), a split could invent a bee
        return [b]
    fit = rim_fit(b)
    if fit is None or not 0.05 < fit["r"] < 0.085:
        return [b]
    P = b["points"]; c = np.array(fit["xy"])
    d = np.linalg.norm(P[:, :2] - c, axis=1)
    A, B = P[d < fit["r"] + 0.012], P[d >= fit["r"] + 0.028]
    tz = b["z_top"] - b["height"]
    parts = [(A, True)]
    if len(B):
        lo = B[:, :2].min(0) - 0.01
        shape = tuple(((B[:, :2].max(0) + 0.01 - lo) / 0.005).astype(int) + 2)
        lab, n = tt._label(B[:, :2], lo, shape, 0.005)
        parts += [(B[lab == k], False) for k in range(1, n + 1)]
    out = []
    for Q, is_bowl in parts:
        if len(Q) < 250:
            continue
        zt = float(np.percentile(Q[:, 2], 98))
        top = Q[Q[:, 2] > zt - 0.012, :2]
        xy = c if is_bowl else top.mean(0)
        ext = np.percentile(Q[:, :2], 98, 0) - np.percentile(Q[:, :2], 2, 0)
        out.append(dict(xy=r3(xy, 4), z_top=round(zt, 4), height=round(zt - tz, 4), extent=r3(ext, 4), n=int(len(Q)), points=Q))
    S.log(f"split merged blob {r3(b['xy'])} ext {b['extent']} -> {[(_brief(o)['xy'], _brief(o)['height']) for o in out]}")
    return out or [b]


def perceive_retry(tag, need_bee=True, tries=3):
    """Side depth sometimes drops an object for a frame set: perceive again before concluding it is missing."""
    for i in range(tries):
        bowl, bee = perceive(tag if i == 0 else f"{tag}_r{i}")
        if bowl is not None and (bee is not None or not need_bee):
            break
        S.log(f"perceive {tag}: bowl {bowl is not None} bee {bee is not None}; retry {i + 1}")
    return bowl, bee


def _brief(b):
    if b is None:
        return None
    return dict(xy=r3(b["xy"]), xy_blob=r3(b.get("xy_blob", b["xy"])), z_top=round(b["z_top"], 3), height=round(b["height"], 3), extent=r3(b["extent"]), n=b["n"])


def rim_fit(bowl):
    """Circle through the top band of the bowl points (the rim) -> centre, radius."""
    P = bowl["points"]; z = P[:, 2]
    top = P[z > np.percentile(z, 98) - 0.012, :2]
    if len(top) < 30:
        return None
    c, r, res = tt._fit_circle(top)
    return dict(xy=r3(c, 4), r=round(float(r), 4), res=round(float(res), 4), n=int(len(top)))


def inside_stats(bowl):
    """Side points inside the rim (r < 4.5 cm of the centre): something in the bowl raises them."""
    P = bowl["points"]; c = np.array(bowl["xy"])
    d = np.linalg.norm(P[:, :2] - c, axis=1)
    Q = P[d < 0.045]
    if len(Q) < 5:
        return dict(n=int(len(Q)))
    return dict(n=int(len(Q)), z50=round(float(np.median(Q[:, 2])), 4), z95=round(float(np.percentile(Q[:, 2], 95)), 4))


def long_axis(b):
    P = b["points"][:, :2].astype(float)
    P = P - P.mean(0)
    w, v = np.linalg.eigh(P.T @ P / max(len(P), 1))
    return float(np.degrees(np.arctan2(v[1, 1], v[0, 1])))


# ------------------------------------------------------------------ wrist look over the bowl
def look_over(xy, z_obj, yaw=90.0, tool_z=0.30):
    """Move so the point (xy, z_obj) appears at LOOK_UV in the wrist image; return (img, T_base_tool)."""
    K, dist_c = calib.wrist_intrinsics()
    for tz in (tool_z, 0.24, 0.18):           # far objects: look from lower (IK reach)
        txy = _look_xy(xy, z_obj, yaw, tz, K)
        if reach_ok(txy, tz, yaw):
            tool_z = tz
            break
    to_hover("look_bowl", txy, yaw, z=tool_z)
    import time
    time.sleep(0.8)
    img, _ = S.wrist.latest()
    return img, tool_T()


def _look_xy(xy, z_obj, yaw, tool_z, K):
    txy = np.array(xy, float)
    for _ in range(3):
        T = tool_down_T([txy[0], txy[1], tool_z], yaw)
        uv = wr.project_base_to_wrist(np.array([[xy[0], xy[1], z_obj]]), T)[0]
        if not np.all(np.isfinite(uv)):
            break
        Zc = (T @ calib.T_tool_cam())[2, 3] - z_obj
        d = (np.array(LOOK_UV) - uv) * Zc / K[0, 0]          # metres in the camera x/y
        Rc = (T @ calib.T_tool_cam())[:3, :3]
        txy = txy - (Rc[:2, 0] * d[0] + Rc[:2, 1] * d[1])
    return txy


def refine_bowl(img, T, side_c, rim_z, r):
    if S.dry or img is None:
        return None, dict(err="dry run" if S.dry else "no image")
    best = None
    for rr in (r, 0.066, 0.072):
        xy, info = wr.circle_refine(img, T, side_c, rim_z, rr)
        if xy is not None and dist(xy, side_c) < 0.03 and (best is None or info.get("coverage", 0) > best[1].get("coverage", 0)):
            best = (xy, info)
    if best is None:
        return None, info
    return np.asarray(best[0], float), best[1]


def bee_in_wrist(img, T, bowl_c, rim_z, plane_z, r_in=0.056, z_in=None, r_ref=0.064, z_ref=None, exclude=None,
                 dbg_name="wrist_bowl_bee.jpg", keep_xy=None, min_frac=0.08):
    """Bee inside the bowl from the wrist image: pixels inside the rim that differ from the bowl wall's median
    colour (contrast, not identity), largest component -> base xy on plane_z, long axis, area."""
    out = dict(found=False)
    if img is None:
        out["err"] = "no image"; return None, out
    K, dc = calib.wrist_intrinsics()
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = lambda rad, z: wr.project_base_to_wrist(np.c_[bowl_c[0] + rad * np.cos(ang), bowl_c[1] + rad * np.sin(ang), np.full(64, z)], T)
    inner = ring(r_in, rim_z - 0.01 if z_in is None else z_in); wall = ring(r_ref, rim_z - 0.004 if z_ref is None else z_ref)
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(wall))):
        out["err"] = "rim not in view"; return None, out
    h, w = img.shape[:2]
    m_in = np.zeros((h, w), np.uint8); cv2.fillPoly(m_in, [inner.astype(np.int32)], 1)
    m_wall = np.zeros((h, w), np.uint8); cv2.fillPoly(m_wall, [wall.astype(np.int32)], 1)
    m_ring = (m_wall > 0) & (m_in == 0)
    if exclude is not None:               # e.g. the bowl next to a bee on the table
        ex = wr.project_base_to_wrist(np.c_[exclude[0][0] + exclude[1] * np.cos(ang), exclude[0][1] + exclude[1] * np.sin(ang),
                                            np.full(64, exclude[2])], T)
        if np.all(np.isfinite(ex)):
            m_ex = np.zeros((h, w), np.uint8); cv2.fillPoly(m_ex, [ex.astype(np.int32)], 1)
            m_in[m_ex > 0] = 0; m_ring &= m_ex == 0
    m_in[int(h * 0.80):, :int(w * 0.22)] = 0; m_in[int(h * 0.80):, int(w * 0.78):] = 0   # fingertips
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 2), cv2.COLOR_BGR2LAB).astype(np.float32)
    if m_ring.sum() < 200 or m_in.sum() < 2000:
        out["err"] = "rim mask too small"; return None, out
    ref = np.median(lab[m_ring], axis=0)
    dlab = np.linalg.norm((lab - ref) * np.array([0.5, 1.0, 1.0]), axis=2)
    vals = dlab[m_in > 0]
    thr = max(18.0, float(np.percentile(vals, 30)) + 12.0)
    fg = ((dlab > thr) & (m_in > 0)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(fg)
    frac = 0.0
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        frac = float(st[k, cv2.CC_STAT_AREA]) / float(m_in.sum())
    out.update(thr=round(thr, 1), ref=r3(ref, 1), area_frac=round(frac, 3))
    dbg = img.copy()
    cv2.polylines(dbg, [inner.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(dbg, [wall.astype(np.int32)], True, (255, 255, 0), 2)
    if n > 1:
        dbg[lab_cc == k] = (0.5 * dbg[lab_cc == k] + np.array([0, 0, 127])).astype(np.uint8)
    cv2.imwrite(os.path.join(S.A, dbg_name), cv2.resize(dbg, (640, 360)))
    if n <= 1 or frac < min_frac or frac > 0.85:
        out["err"] = f"no bee-sized region (area {frac:.2f} of the bowl interior)"; return None, out
    ys, xs = np.nonzero(lab_cc == k)
    sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 3000), replace=False)
    uv = np.c_[xs[sel], ys[sel]].astype(np.float64)
    rays = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dc).reshape(-1, 2)
    Tc = T @ calib.T_tool_cam()
    rb = np.c_[rays, np.ones(len(rays))] @ Tc[:3, :3].T
    o = Tc[:3, 3]
    s_ = (plane_z - o[2]) / rb[:, 2]
    Pb = o[None, :] + s_[:, None] * rb
    c, lng, L, W = _shape(Pb[:, :2])
    if keep_xy is not None and len(keep_xy) >= 50 and (L > 0.125 or W > 0.08):
        # region too big for a bee: a flat shadow or the bowl wall joined it (att 151/152: 0.144 x 0.05, 0.12 x 0.081,
        # four misses). Keep the part that lies over the side camera's raised points (geometry: shadows are flat)
        g = 0.005
        lo = np.min(keep_xy, 0) - 0.03
        occ = np.zeros(tuple(((np.max(keep_xy, 0) + 0.03 - lo) / g).astype(int) + 2), np.uint8)
        ij = ((np.asarray(keep_xy) - lo) / g).astype(int)
        occ[ij[:, 0], ij[:, 1]] = 1
        occ = cv2.dilate(occ, np.ones((5, 5), np.uint8))          # +-1 cm for the side/wrist calibration offset
        q = ((Pb[:, :2] - lo) / g).astype(int)
        ins = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < occ.shape[0]) & (q[:, 1] < occ.shape[1])
        ins[ins] = occ[q[ins, 0], q[ins, 1]] > 0
        out.update(raw=dict(xy=r3(c, 4), long_axis=round(lng, 1), len=round(float(L), 3), wid=round(float(W), 3)),
                   kept=round(float(ins.mean()), 2))
        if ins.sum() >= 200:
            c, lng, L, W = _shape(Pb[ins, :2])
    out.update(found=True, xy=r3(c, 4), long_axis=round(lng, 1), len=round(float(L), 3), wid=round(float(W), 3),
               off_centre=round(dist(c, bowl_c), 3))
    return dict(xy=c, long=lng, len=L, wid=W), out


def _shape(P):
    """Centre, long-axis angle (deg), length and width (4 sigma) of 2-D points."""
    c = P.mean(0)
    Q = P - c
    ev, evec = np.linalg.eigh(Q.T @ Q / len(Q))
    lng = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
    return c, lng, 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0])


# ------------------------------------------------------------------ grasps
def grip_ok(reading, max_held, what):
    if S.dry:
        S.decide(f"{what}: dry run, gripper reading simulated ({reading:.1f}) -> treated as held")
        return True
    return reading < max_held


def open_jaws(frac, what):
    """Open an EMPTY gripper to frac before a grasp. att 716: opening at the hover over the bee in the bowl (0.236 m up)
    was refused by the library ('object on the fingers': it saw the bee below in the bottom of the wrist image).
    No command if the jaws are already that open; a refusal returns False (the caller opens lower down)."""
    cur = getattr(S, "grip_frac", None)
    if cur is not None and getattr(S, "grip_cmd_sent", False) and frac >= cur - 0.02:
        return True
    try:
        S.gripper(frac, what)
        return True
    except RunTimeLimit:
        raise
    except Abort as e:
        if "release refused" not in str(e):
            raise
        S.decide(f"{what}: opening here was refused ({str(e)[:80]}...) -> open lower, inside the zone")
        return False


def pick_bee(xy, jaw_yaw, grasp_z, open_frac, what, wall_min=None, held_max=None):
    """Open (before the hover: from there the camera sees the bee below), hover, slow descent to grasp_z (tool),
    close, check, lift to HOVER_Z. True if held."""
    opened = open_jaws(open_frac, f"open for {what}")
    y = to_hover(f"above_{what}", xy, jaw_yaw)
    if not opened:
        opened = open_jaws(open_frac, f"open for {what}")
    line_to(f"approach_{what}", xy, grasp_z + 0.06, y, speed="normal")
    if not opened:
        S.gripper(open_frac, f"open for {what} (low)")
    line_to(f"descend_{what}", xy, grasp_z, y, speed="slow")
    g = S.gripper(1.0, f"grasp {what}")
    held = grip_ok(g, BEE_HELD_MAX, what)
    S.decide(f"{what}: gripper reading {g:.1f} -> {'held' if held else 'EMPTY'}")
    if held and wall_min is not None and g < wall_min and not S.dry:
        # bee alone reads 75-90; 67-71 next to the bowl (att 113/114) = a finger has the bowl wall too: don't lift
        S.gripper(open_frac, "let go: the grip includes the bowl wall")
        vertical(f"retreat_wall_{what}", None, speed="slow")
        S.require(False, f"bee lies against the bowl; the grip caught the bowl wall (reading {g:.1f}): reset by hand")
    if held and held_max is not None and g > held_max and not S.dry:
        # in-bowl grips reading 89-90 went wrong: att 360 lost the bee in transit, att 380 carried the BOWL 13 cm
        # with it (wall in the jaws). Recent good in-bowl grips read 71-81: let go here and look again
        S.decide(f"{what}: reading {g:.1f} > {held_max}: a marginal grip (bowl wall / bee tip) -> let go, look again")
        held = False
    if not held:
        S.gripper(open_frac, "open after a miss")
        vertical(f"retreat_{what}", None, speed="slow")
        return False, g, y
    vertical(f"lift_{what}", grasp_z + 0.05, speed="slow", tip_margin=0.0)
    vertical(f"lift_high_{what}", None, speed="normal", tip_margin=0.0)
    if not S.dry:
        g2 = S.feedback()["gripper"]
        S.decide(f"{what}: reading after lift {g2:.1f}")
        if g2 > BEE_HELD_MAX:
            S.gripper(0.0, "open: lost the bee")
            return False, g2, y
    return True, g, y


BEE_SIDE_DX = 0.0     # was 0.016 for the point-median metric; the blob-centre metric (orchestrator) reads a placed bee ~2 cm lower in x (att 103: -0.021, 104: -0.024), so no offset
                      # (att 25, 26, 28, 29: +0.020, +0.011, +0.020, +0.018): put it that much short


REL_X = (0.38, 0.58)      # release spots: the library's release zone x 0.36-0.60, y -0.22..0.08 with 2 cm to spare
REL_Y = (-0.20, 0.06)


def place_bee(xy, yaw, release_z, what, check_empty=True):
    xy = [float(xy[0]) - BEE_SIDE_DX, float(xy[1])]
    cl = [float(np.clip(xy[0], *REL_X)), float(np.clip(xy[1], *REL_Y))]
    if dist(cl, xy) > 1e-4:
        S.decide(f"{what}: release spot {r3(xy)} outside the release zone -> {r3(cl)}")
        xy = cl
    release_z = min(release_z, float(S.table_z) + 0.15)
    yaw = to_hover(f"carry_{what}", xy, yaw)       # the yaw IK chose (0 or 180): keep it for the lines below
    if check_empty and not S.dry and S.feedback()["gripper"] >= EMPTY_MIN:
        S.decide(f"{what}: the jaws are empty after the carry (reading {S.feedback()['gripper']:.1f}): bee lost in transit")
        S.gripper(0.0, "open (empty)")
        return False
    line_to(f"lower_{what}", xy, release_z + 0.05, yaw, speed="normal", tip_margin=0.0)
    line_to(f"place_{what}", xy, release_z, yaw, speed="slow", tip_margin=0.0)
    S.gripper(0.0, f"release {what}")
    try:
        ensure_released(xy, yaw, release_z, what)      # stays low until the side camera sees the bee on the table
    except BeeMissing as e:
        S.decide(f"{e}: treat it as lost in transit")
        vertical(f"rise_missing_{what}", None, speed="normal")
        return False
    vertical(f"rise_high_{what}", None, speed="normal")
    return True


def stuck_frac(tag, full=False):
    """The foam bee can cling to a fingertip after the jaws open (att 72/73: carried away on the finger). From the
    wrist camera the fingertips fill the bottom corners: a bee stuck there is a large saturated-yellow area in the
    bottom 15 % of the image (a released bee on the table appears near the image centre, not there)."""
    if S.dry:
        return 0.0
    import time
    time.sleep(0.4)
    img, _ = S.wrist.latest()
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img[int(0.85 * h):], cv2.COLOR_BGR2HSV)
    m = bee_mask(hsv)
    # only the fingertip corners: a released bee under the gripper can reach the bottom centre (att 87: 0.051 full
    # width, 0.0 in the corners; stuck bee 0.12-0.13 either way)
    # full width everywhere (operator's scoring of att 74-101: stuck 0.21-0.22, released below <= 0.052,
    # empty <= 0.027; the corner mask never fired)
    # full=True at home: the camera looks across the room, so any yellow at the bottom is a bee hanging below the
    # fingertips (att 94: bottom centre, squeeze test empty)
    f = float(m.mean())
    hw = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mw = bee_mask(hw)
    fw = float(mw.mean())
    # a bee stuck on one fingertip can sit mostly above the bottom strip (att 119: bottom 0.126, whole image 0.080);
    # released bees below the gripper never exceeded 0.036 over the whole image (att 102-118)
    S.log(f"gripper-clean check {tag}: yellow fraction bottom strip {f:.3f}, whole image {fw:.3f}")
    # att 705: tool-down after the start drop, the bee lying in the bowl below filled 0.065-0.084 of the image and
    # the whole-image rule called it stuck 5 times (bottom strip 0.000-0.088). A bee on the fingers always reaches the
    # last image rows (last 4 %: stuck 0.11-0.37, held 0.12-0.18); one on the table below <= 0.016 there.
    edge = float(mw[int(0.96 * h):].mean())
    if fw >= WHOLE_STUCK and edge >= 0.05:
        f = max(f, 1.0)
    elif fw >= WHOLE_STUCK:
        S.log(f"gripper-clean check {tag}: whole-image colour is below the gripper (last rows {edge:.3f}), not on it")
    cv2.imwrite(os.path.join(S.A, f"wrist_fingers_{tag}.jpg"), cv2.resize(img, (640, 360)))
    return f


WHOLE_STUCK = 0.06        # whole-image bee colour: stuck 0.08+ (att 119), released <= 0.036
STUCK_FRAC = 0.10         # bottom 15 % at home, orange range (operator 2026-11-06): bee hanging 0.27, empty <= 0.006


def bee_mask(hsv):
    """Bee colour (OpenCV HSV). Since 2026-11-05 the bee is wrapped in orange: hue 0-40, S > 120, V > 110 (operator;
    the old yellow range 15-40/100/90 read 0.016 for a hanging bee)."""
    return (hsv[..., 0] <= 40) & (hsv[..., 1] > 120) & (hsv[..., 2] > 110)


START_DROP_XY = (0.58, 0.00)   # straight below home, inside the release zone (operator)
DROP_PTS = [np.zeros((0, 2))]   # table points seen when the start drop spot was chosen
SQUEEZE_EMPTY = 99.0           # start: any squeeze reading below this = something in the jaws (operator)


def free_drop_spot(table_z):
    """(0.58, 0.00) if no side blob lies within 0.10 m, else the nearest free point of the zone (x 0.40-0.58,
    y -0.18..0.05)."""
    pts = []
    if not S.dry:
        for b in S.tabletop():
            P = np.asarray(b["points"], float)
            pts.append(P[(P[:, 2] - table_z) < 0.30, :2])
    P = np.concatenate(pts) if pts else np.zeros((0, 2))
    DROP_PTS[0] = P
    free = lambda q: len(P) == 0 or float(np.min(np.linalg.norm(P - np.asarray(q), axis=1))) >= 0.10
    if free(START_DROP_XY):
        return list(START_DROP_XY)
    # att 293/294: bowl at (0.49, 0.01) + bee at (0.52, -0.15) left no point of x 0.40-0.58 / y -0.18..0.05 with 0.10 m
    # clearance -> search the whole operator zone (x 0.37-0.59, y -0.21..0.07) and accept down to 0.06 m clearance
    for clear in (0.10, 0.08, 0.06):
        free = lambda q, c=clear: len(P) == 0 or float(np.min(np.linalg.norm(P - np.asarray(q), axis=1))) >= c
        best = None
        for x in np.arange(0.37, 0.591, 0.01):
            for y in np.arange(-0.21, 0.071, 0.01):
                if free((x, y)):
                    d = dist((x, y), START_DROP_XY)
                    if best is None or d < best[0]:
                        best = (d, [float(x), float(y)])
        if best is not None:
            S.decide(f"start drop spot {r3(best[1])} (clearance >= {clear:.2f} m)")
            return best[1]
    S.require(False, "no free spot in the release zone to put down the bee on the gripper")


def release_at_start(table_z):
    """Bee in or on the gripper at start (operator 2026-11-06): no carry (a bee held by its tip falls out on any
    longer carry, rollout 119 / att 209). From home lower straight down over a free spot of the zone to 0.10 m above
    the table, open, rise; repeat with a jaw wiggle while the bee is still seen on the gripper."""
    spot = free_drop_spot(table_z)
    S.decide(f"bee on the gripper at start: turn tool-down in place over {r3(spot)}, lower to 0.10 m, open, rise")
    # att 285: lowering in the home orientation (tool pointing forward) drove joint 5 to 129 deg and the arm faulted;
    # turn the tool down in place first (no carry), then lower vertically tool-down
    high = float(table_z) + 0.30
    q_up, yaw = ik([spot[0], spot[1], high], 0.0)
    S.require(within_limits(q_up), f"no in-limit tool-down pose over {r3(spot)}")
    S.move_joint("start_drop_turn", q_up, "slow")
    for k in range(5):
        if k == 0:
            line_to(f"start_drop_lower{k}", spot, table_z + 0.10, yaw, speed="slow", tip_margin=0.0)
            S.gripper(0.0, "release the bee at start")
        else:
            # att 521/522: the bee hung on the gripper (tape snag) with the jaws open; opening and wiggling at 0.10 m
            # never freed it. att 523: low (tool tz+0.06) with a 4 cm slide along y: the bee rested on the table but
            # stayed on the finger. Now each try presses lower (tool tz+0.05, fingertips ~2.3 cm up), wiggles, and
            # drags the bee a different way: across the jaws (it can leave between the pads), along the jaw axis,
            # and a 45 deg twist, then across the other way (operator: stuck bee -> go back low, wiggle, slide).
            P = DROP_PTS[0]

            def clear_of(q):
                return 1.0 if len(P) == 0 else float(np.min(np.linalg.norm(P - np.asarray(q), axis=1)))

            def slide_end(ang_deg, length=0.06, prefer=None):
                a = np.radians(ang_deg)
                u = np.array([np.cos(a), np.sin(a)])
                best = None
                for L in (length, 0.04, 0.03):
                    for sg in ((prefer,) if prefer else (1.0, -1.0)):
                        q = [spot[0] + sg * L * u[0], spot[1] + sg * L * u[1]]
                        if 0.36 <= q[0] <= 0.60 and -0.22 <= q[1] <= 0.08:
                            c = clear_of(q)
                            if c >= 0.05 and (best is None or c > best[1]):
                                best = (q, c)
                    if best:
                        return best[0]
                return None

            low = table_z + 0.05
            line_to(f"start_drop_lower{k}", spot, table_z + 0.09, yaw, speed="slow", tip_margin=0.0)
            line_to(f"start_drop_low{k}", spot, low, yaw, speed="slow", tip_margin=0.0)
            S.gripper(0.0, "release the bee at start")
            for v in (0.6, 0.0, 0.6, 0.0):
                S.gripper(v, "wiggle the jaws")
            mode = ("across", "along", "twist")[(k - 1) % 3]
            if mode == "twist":
                for d in (45.0, -45.0, 0.0):
                    line_to(f"start_drop_twist{k}_{int(d)}", spot, low, yaw + d, speed="slow", tip_margin=0.0)
                side = slide_end(yaw + 90.0)
            else:
                side = slide_end(yaw + (90.0 if mode == "across" else 0.0))
            S.decide(f"the bee is on the gripper: low over {r3(spot)} (jaw yaw {yaw:.0f}), wiggle, {mode}"
                     + (f", slide to {r3(side)} (clearance {clear_of(side):.3f} m)" if side else ", no free slide"))
            if side is not None:
                line_to(f"start_drop_slide{k}", side, low, yaw, speed="slow", tip_margin=0.0)
                vertical(f"start_drop_up{k}", table_z + 0.09, speed="slow")
                line_to(f"start_drop_back{k}", spot, table_z + 0.09, yaw, speed="slow", tip_margin=0.0)
        vertical(f"start_drop_rise{k}", high, speed="slow")
        f = stuck_frac(f"start_drop{k}", full=True)
        if f < STUCK_FRAC:
            g = squeeze_test(f"start_drop{k}")
            # after an open-and-drop, a clean colour check with a squeeze of 94+ is an empty gripper whose closed
            # reading drifted (att 285 read 95.2 at start with the colour check clean)
            if g >= SQUEEZE_EMPTY or (g >= 94.0) or S.dry:
                if g < EMPTY_MIN and not S.dry:
                    S.gripper(0.0, "open again")
                S.decide(f"the gripper is empty after the start drop (colour {f:.3f}, squeeze {g:.1f})")
                return spot
        S.decide(f"the bee is still on the gripper after the start drop (colour {f:.3f}): try {k + 2}")
    S.require(False, "the bee stays on the gripper after 5 drops at start (4 low with slides / a twist)")
EMPTY_MIN = 97.5
BEE_WITH_WALL = 74.0       # bee alone reads 75-90; bee + bowl wall 67-71           # closed on nothing reads ~99.1; on the bee ~78-91


def squeeze_test(tag):
    """Close the open gripper once: a reading well below 99 means something is between the fingers. Leaves it open
    when empty, closed (holding) otherwise."""
    if S.dry:
        return 100.0
    g = S.gripper(1.0, f"squeeze test {tag}")
    if g >= EMPTY_MIN:
        S.gripper(0.0, "open again (empty)")
    return g


def side_bee_at(xy, tag):
    """Side-camera confirmation that the bee lies on the table at xy: bee-height points (1-6.5 cm above the table,
    so neither the raised gripper nor a bee hanging on it counts) within 5 cm of xy, centred within 4 cm, >= 150
    points, top >= 3 cm. Bowl blobs are skipped. Returns (ok, info)."""
    if S.dry:
        return True, dict(dry=True)
    tz = float(S.table_z)
    info = {}
    for i in range(2):                          # one more frame set if the depth drops the bee once
        pts = []
        for b in S.tabletop():
            ext = max(b["extent"])
            if 0.062 <= b["height"] <= 0.095 and 0.11 <= ext <= 0.20 and b["n"] >= 800:
                continue                        # the bowl
            P = np.asarray(b["points"], float)
            h = P[:, 2] - tz
            m = (h > 0.010) & (h < 0.065) & (np.linalg.norm(P[:, :2] - np.asarray(xy[:2], float), axis=1) < 0.07)
            if m.any():
                pts.append(P[m])
        P = np.concatenate(pts) if pts else np.zeros((0, 3))
        n = len(P)
        c = np.median(P[:, :2], 0) if n else None
        top = float(np.percentile(P[:, 2] - tz, 95)) if n else 0.0
        # a released bee may bounce a few cm (att 259: centre 4.6 cm off, 97-128 points near the e-stop edge)
        ok = n >= 80 and dist(c, xy) <= 0.06 and top >= 0.025
        info = dict(n=n, centre=None if c is None else r3(c), top=round(top, 3), ok=ok, frame=i)
        S.log(f"side check {tag} at {r3(xy)}: {info}")
        if ok:
            break
    S.summary(**{f"side_check_{tag}": info})
    return info["ok"], info


class BeeMissing(Exception):
    """After a release: the bee is neither at the release spot nor on the gripper (it fell off earlier)."""


def ensure_released(xy, yaw, release_z, what):
    """After a release, before travelling high or going home (operator, att 119): squeeze test right above the spot,
    then rise ~10 cm, step ~8 cm toward the base so the gripper does not hide the spot, and look with the side camera.
    No bee within 4 cm of the spot -> it is stuck: back down low, wiggle the jaws, slide sideways, check again."""
    tz = float(S.table_z)
    look_z = min(release_z + 0.10, tz + 0.19)
    u = -np.asarray(xy[:2], float) / max(np.hypot(*xy[:2]), 1e-6)
    look_xy = np.asarray(xy[:2], float) + 0.08 * u
    clean, squeezes = [], []
    for k in range(3):
        vertical(f"rise_{what}{k}", look_z, speed="slow")
        fy = stuck_frac(f"{what}{k}", full=True)   # logged; wrist colour alone cannot tell (operator)
        g = squeeze_test(f"{what}{k}")
        clean.append(fy < STUCK_FRAC and g >= EMPTY_MIN)
        squeezes.append(g >= EMPTY_MIN)
        if g < EMPTY_MIN:                      # bee wedged between the fingers (att 92)
            S.decide(f"{what}: closing the empty gripper read {g:.1f} (< {EMPTY_MIN}): the bee is still in the jaws")
            seen = False
        else:
            line_to(f"look_back_{what}{k}", look_xy, look_z, yaw, speed="normal", tip_margin=0.0)
            seen, info = side_bee_at(xy, f"{what}{k}")
            if seen:
                if k:
                    S.decide(f"{what}: bee on the table after {k} unstick tries ({info})")
                return True
            if k == 0 and not S.dry:
                # att 653: jaws closed empty (99.6) but the side camera missed the bee at the spot; the unstick slides
                # then pushed it 11 cm away and the run stopped as "stuck". Look down with the wrist camera first,
                # before any unstick move disturbs the bee.
                w = wrist_bee_on_table(np.asarray(xy[:2], float), tz, None, 30, max_off=0.10)
                if w is not None:
                    S.decide(f"{what}: jaws empty, wrist camera sees the bee on the table at {r3(w['xy'])} "
                             f"(released {r3(xy)}): released")
                    return True
            S.decide(f"{what}: side camera sees no bee at the release spot {r3(xy)} ({info}): treat it as stuck")
            line_to(f"back_over_{what}{k}", xy, look_z, yaw, speed="normal", tip_margin=0.0)
        S.decide(f"{what}: unstick try {k + 1}")
        line_to(f"unstick_down_{what}{k}", xy, release_z + 0.05, yaw, speed="normal", tip_margin=0.0)
        line_to(f"unstick_low_{what}{k}", xy, release_z, yaw, speed="slow", tip_margin=0.0)
        for v in (0.6, 0.0, 0.6, 0.0):
            S.gripper(v, "wiggle the jaws")
        a = np.radians(yaw + 90.0 * (1 if k % 2 == 0 else -1))
        side = [xy[0] + 0.03 * np.cos(a), xy[1] + 0.03 * np.sin(a)]
        line_to(f"unstick_slide_{what}{k}", side, release_z + 0.004, yaw, speed="slow", tip_margin=0.0)
        vertical(f"unstick_rise_{what}{k}", release_z + 0.06, speed="slow")   # stay low: the next check looks first
    if not S.dry and all(g_ok for g_ok in squeezes):
        # the jaws close empty: before calling the bee stuck, look straight down with the wrist camera at the release
        # area (att 259: the bee had bounced 5 cm and lay half hidden from the side camera; the bee below the gripper
        # also put orange into the wrist image, so the colour checks did not say "clean")
        w = wrist_bee_on_table(np.asarray(xy[:2], float), tz, None, 20, max_off=0.10)
        if w is not None:
            S.decide(f"{what}: wrist camera sees the bee on the table at {r3(w['xy'])} (released {r3(xy)}): released")
            return True
        # att 425: the bee bounced 11 cm (to y 0.16) out of the side check radius and the wrist view; the jaws closed
        # empty every time. A bee-sized side blob anywhere on the table means it was released: the next step takes it
        # (looked at from where the arm is: no going home before the bee is confirmed off the gripper, operator rule)
        _, b2 = perceive_retry(f"released_{what}", need_bee=False)
        if b2 is not None:
            S.decide(f"{what}: jaws empty and a bee lies on the table at {r3(b2['xy'])}: released (bounced away)")
            return True
    if not S.dry and all(clean):
        # every check said the gripper is empty (wrist yellow low, squeeze 99) and the bee is not on the table here:
        # it was never released here (att 181: an in-bowl grip at 74.1 lost the bee back into the bowl on the lift)
        raise BeeMissing(f"{what}: no bee at the release spot after 3 unstick tries and the gripper is empty")
    S.require(False, "the bee stays stuck on the gripper after 3 unstick tries")


def wrist_bee_on_table(xy, table_z, bowl_xy, k, side_pts=None, tall=False, max_off=0.045):
    """Look straight down at the side estimate; bee = pixels within 9 cm that differ from the table ring around it."""
    if S.dry:
        return None
    try:
        img, T = look_over(xy, table_z + 0.03, tool_z=0.30)
        ex = None if bowl_xy is None else (bowl_xy, 0.085, table_z + 0.03)
        b, info = bee_in_wrist(img, T, xy, table_z, table_z + 0.035, r_in=0.09, z_in=table_z, r_ref=0.115,
                               z_ref=table_z, exclude=ex, dbg_name=f"wrist_bee_table{k + 1}.jpg",
                               keep_xy=None if side_pts is None else side_pts[side_pts[:, 2] > table_z + 0.015, :2])
    except RunTimeLimit:
        raise
    except Exception as e:
        S.log(f"wrist bee refine failed: {e!r}"); return None
    S.summary(**{f"bee_wrist{k + 1}": info})
    # bee 0.10-0.12 x 0.05-0.07 in every good grasp; 0.144 x 0.05 / 0.12 x 0.081 (shadow joined) missed 4 times (att 151/152)
    lmin = 0.035 if tall else 0.06            # a standing bee seen from above is ~6 x 4.5 cm
    if b is None or dist(b["xy"], xy) > max_off or not (lmin < b["len"] < 0.13) or not (0.03 < b["wid"] < 0.085):
        S.log(f"wrist bee refine rejected: {info}")
        return None
    return b


class BeeBlocked(Exception):
    """The bee cannot be gripped where it lies: the bowl is in the way of a finger."""


FINGER_R = 0.012          # finger centre beyond the opening edge (half pad width + margin)
RIM_CLEAR = 0.088         # finger centre at least this far from the bowl centre (rim ~0.075 + pad)


def grasp_opening(xy, jaw, bowl_xy):
    """Gripper command for a table grasp of the bee at xy across jaw yaw `jaw`: the widest opening (85..70 mm) whose
    fingers stay clear of the bowl rim; None when even 70 mm puts a finger on the rim."""
    if bowl_xy is None or dist(xy, bowl_xy) > 0.20:
        return 0.0
    a = np.radians(jaw); u = np.array([np.cos(a), np.sin(a)])
    for o in (0.085, 0.080, 0.075, 0.070):
        f = [np.asarray(xy[:2], float) + sg * (o / 2 + FINGER_R) * u for sg in (1, -1)]
        if all(dist(p, bowl_xy) >= RIM_CLEAR for p in f):
            return 1.0 - o / 0.085
    return None


def jaw_gap_to(xy, jaw, opening, c):
    """Distance from c to the segment between the open fingers (where a bowl wall would get gripped)."""
    a = np.radians(jaw); u = np.array([np.cos(a), np.sin(a)])
    p = np.asarray(xy[:2], float); d = np.asarray(c[:2], float) - p
    t = float(np.clip(d @ u, -opening / 2, opening / 2))
    return float(np.linalg.norm(d - t * u))


def bowl_spot(bee_xy, bee_dest, bowl_target):
    """Bowl spot away from the bee (and where the bee goes) inside the bowl layout, near its target."""
    best = None
    for x in np.arange(0.40, 0.541, 0.01):
        for y in np.arange(-0.17, 0.051, 0.01):
            m = min(dist((x, y), bee_xy), dist((x, y), bee_dest))
            sc = min(m, 0.22) - 0.1 * dist((x, y), bowl_target)
            if best is None or sc > best[0]:
                best = (sc, [float(x), float(y)], m)
    return best[1], best[2]


def move_bee_from_table(bee, dest, table_z, bowl_xy, place_yaw=None, drop=0.0, force=False):
    """Side blob -> grasp across the short axis at the side centre, place with the tool at dest."""
    for k in range(2):
        jaw = long_axis(bee) + 90.0
        xy = np.asarray(bee["xy"], float)
        wx = wrist_bee_on_table(xy, table_z, bowl_xy, k, side_pts=np.asarray(bee["points"], float) if "points" in bee else None,
                                tall=float(bee.get("height", 0.0)) > 0.06)
        if wx is not None:
            S.decide(f"wrist camera: bee centre {r3(xy)} -> {r3(wx['xy'])}, long axis {wx['long']:.0f} (side {jaw - 90:.0f})")
            xy, jaw = np.asarray(wx["xy"], float), wx["long"] + 90.0
            bee = dict(bee, xy=r3(xy, 4))
            if k == 0 and not force and dist(xy, dest) < 0.020:   # 1.5 -> 2.0 cm: re-puts at 1.7/1.9 cm ended 1.9-2.0 cm anyway (att 164, 175)
                # the wrist sees the bee already on the spot (att 149: blob 2.45 cm, wrist 0.6 cm): a re-put would
                # land in the same place and only adds a release
                S.decide(f"wrist camera: bee already within {dist(xy, dest):.3f} m of {r3(dest)}: not moved")
                return True
        # >= 0.042: the bee is 4.1-4.9 cm tall; a 3.7 cm side reading near x 0.35 put the tool at 3.0 cm and the
        # closed fingertips below the table, so the lift was refused (att 182)
        r_b = float(np.hypot(*xy[:2]))
        if r_b > 0.69:
            # att 701: bee at r 0.72; the approach stretched the arm to r 0.806 (> 0.80, refused). Grip the bee's near
            # end (shift <= 3 cm toward the base, the bee is 7-10 cm across) with the jaws closing tangentially
            sh = min(0.03, r_b - 0.69)
            xy = xy * (1.0 - sh / r_b)
            jaw = float(np.degrees(np.arctan2(xy[1], xy[0]))) + 90.0
            bee = dict(bee, xy=r3(xy, 4))
            S.decide(f"bee at r {r_b:.3f} is at the reach limit: grip its near end at {r3(xy)} (r {np.hypot(*xy):.3f}), "
                     f"jaw yaw {jaw:.0f}")
        h = float(np.clip(bee["height"], 0.042, 0.055))
        gz = table_z + h + BEE_TOOL_ABOVE_TOP
        near_bowl = bowl_xy is not None and dist(bee["xy"], bowl_xy) < 0.14
        # opening from the finger positions, no grip offset: the old 12 mm shift away from the bowl ran along the jaw
        # when the bee pointed at the bowl, put a finger on the 6 cm wide bee (70 mm opening) and pushed bee and
        # bowl (att 151-153: 6 misses, bowl pushed 10 cm)
        op = grasp_opening(bee["xy"], jaw, bowl_xy)
        if op is None:
            raise BeeBlocked(f"bee at {r3(bee['xy'])} lies with its jaw axis ({jaw:.0f}) toward the bowl at "
                             f"{r3(bowl_xy)}: a finger would land on the rim")
        S.decide(f"bee on table at {r3(bee['xy'])} h {h:.3f}: jaw yaw {jaw:.0f}, tool z {gz:.3f}, try {k + 1}, "
                 f"opening {(1 - op) * 85:.0f} mm")
        # the wall check (reading < 74 = bee + wall) only where the wall can be between the jaws: the bee alone read
        # 73.7 with the jaw segment 9.9 cm from the bowl centre (att 154 stopped for nothing)
        wall_possible = near_bowl and jaw_gap_to(bee["xy"], jaw, (1 - op) * 0.085, bowl_xy) < 0.085
        ok, g, y = pick_bee(bee["xy"], jaw, gz, op, f"bee{k + 1}",
                            wall_min=BEE_WITH_WALL if wall_possible else None)
        if ok:
            # "rotate": put it down turned 90 deg (jaw along its old long axis), att 336
            py = (jaw - 90.0) if place_yaw == "rotate" else (PLACE_YAW if place_yaw is None else place_yaw)
            if not place_bee(dest, py, gz + 0.006 + drop, "bee"):
                S.home("home_after_drop")
                _, bee = perceive_retry(f"bee_dropped{k + 1}")
                S.require(bee is not None, "bee dropped in transit and not found on the table")
                continue
            dry_move("bee", dest)
            return True
        S.home("home_reperceive")
        _, bee = perceive_retry(f"bee_retry{k + 1}")
        S.require(bee is not None, "bee not found after a missed grasp")
    return False


def in_zone(xy, m=0.0):
    """Inside the task release zone (x 0.36-0.60, y -0.22..0.08), m metres from its edges."""
    return 0.36 + m <= float(xy[0]) <= 0.60 - m and -0.22 + m <= float(xy[1]) <= 0.08 - m


def move_bowl(bowl, side_c, dest, table_z, bee_xy, disp_ref=None, gain=1.0, cap=None, rim_min=None):
    """Pinch the rim on the side that keeps clear of the bee, carry the bowl by (dest - side centre)."""
    rim_z = float(bowl["z_top"])
    fit = rim_fit(bowl)
    c_grasp = np.array(fit["xy"] if fit and 0.05 < fit["r"] < 0.085 and dist(fit["xy"], side_c) < 0.02 else side_c, float)
    img, T = look_over(c_grasp, rim_z)
    if img is not None:
        cv2.imwrite(os.path.join(S.A, "wrist_bowl_look.jpg"), cv2.resize(img, (640, 360)))
    ref, info = refine_bowl(img, T, c_grasp, rim_z, 0.066)
    S.summary(bowl_wrist=info)
    if ref is not None:
        S.decide(f"wrist rim circle: grasp centre {r3(c_grasp)} -> {r3(ref)}")
        c_grasp = ref
    disp = np.subtract(dest, side_c if disp_ref is None else disp_ref)   # disp_ref: the orchestrator's blob centre
    if disp_ref is not None and ref is not None:
        # att 416: wrist rim said the bowl sat on its target (0.437 vs 0.439) but the blob median read 2.6 cm short;
        # the full blob move put it 3 cm off the other way. Blob and rim disagreeing by > 2 cm = blob jitter: split it
        wdisp = np.subtract(dest, ref)
        if float(np.linalg.norm(disp - wdisp)) > 0.02:
            S.decide(f"bowl carry: blob says {r3(disp)}, wrist rim says {r3(wdisp)}: use the mean")
            disp = 0.5 * (disp + wdisp)
    if gain != 1.0 or cap is not None:
        # correction: the blob centre near x 0.37-0.41 jumps by 4 cm between frames (att 346: rim at 0.400 read
        # 0.365/0.365/0.408, the full correction moved it to read 3.8 cm off the other way) -> part of it, capped
        disp = gain * disp
        n = float(np.linalg.norm(disp))
        if cap is not None and n > cap:
            disp = disp * (cap / n)
    # grasp side: base-facing first, rotate away from the bee
    base_dir = np.degrees(np.arctan2(-c_grasp[1], -c_grasp[0]))
    best = None
    for da in (0, 30, -30, 60, -60, 90, -90):
        a = np.radians(base_dir + da); u = np.array([np.cos(a), np.sin(a)])
        g0 = c_grasp + RIM_R * u; g1 = g0 + disp
        if not (0.28 < np.hypot(*g0) < 0.62 and 0.28 < np.hypot(*g1) < 0.62):
            continue
        zone_ok = in_zone(g0, 0.008) and in_zone(g1, 0.008)
        if rim_min is not None and not zone_ok:
            # att 650: the unclean-pinch let-go at (0.358, 0.025) was refused (outside the release zone, reading 90.8
            # counts as holding the bee): with rim_min the grasp point must allow opening there. att 704: the rim
            # release after the carry at g1 (0.421, 0.087) was refused too (bee in the jaws with the rim, reading
            # 94.7, colour on the fingers) and the arm went home holding the bee -> g1 must be in the zone as well
            continue
        clear = 1.0 if bee_xy is None else min(dist(bee_xy, c_grasp + 0.10 * u), dist(bee_xy, g1 + 0.03 * u))
        # any carry: the bee in the bowl can show at the bottom of the wrist image and the safeguard then refuses to
        # open outside the zone -> strongly prefer pick and release points inside it
        score = min(clear, 0.12) - 0.0002 * abs(da) + (0.2 if zone_ok else 0.0)
        if best is None or score > best[0]:
            best = (score, da, u, g0, g1, clear)
    if best is None and rim_min is not None:
        S.decide("bowl carry with the bee: no rim grasp inside the release zone -> not carried")
        return False
    S.require(best is not None, "no reachable rim grasp for the bowl")
    _, da, u, g0, g1, clear = best
    yaw = float(np.degrees(np.arctan2(u[1], u[0])))
    gz = rim_z + 0.009
    lift = 0.03
    if bee_xy is not None:     # bee near the carry path: lift the bowl over it
        p = np.array(bee_xy) - c_grasp; L = np.linalg.norm(disp)
        tpar = np.clip(p @ disp / max(L * L, 1e-9), 0, 1)
        if np.linalg.norm(p - tpar * disp) < 0.14:
            lift = 0.08
    S.decide(f"bowl: rim grasp at {r3(g0)} (angle {da:+.0f} from base-facing, jaw yaw {yaw:.0f}, tool z {gz:.3f}), "
             f"carry by {r3(disp)} to {r3(g1)}, lift {lift:.2f}, bee clearance {clear:.3f}")
    for k in range(2):
        opened = open_jaws(BOWL_OPEN, "open 58 mm around the rim")     # before the hover (att 716)
        y = to_hover(f"above_rim{k + 1}", g0, yaw)
        if not opened:
            opened = open_jaws(BOWL_OPEN, "open 58 mm around the rim")
        line_to(f"approach_rim{k + 1}", g0, gz + 0.06, y, speed="normal")
        if not opened:
            S.gripper(BOWL_OPEN, "open 58 mm around the rim (low)")
        line_to(f"descend_rim{k + 1}", g0, gz, y, speed="slow")
        g = S.gripper(1.0, "pinch the rim")
        held = grip_ok(g, BOWL_HELD_MAX, "bowl rim")
        S.decide(f"bowl rim: gripper reading {g:.1f} -> {'held' if held else 'EMPTY'}")
        if held and rim_min is not None and g < rim_min and not S.dry:
            # att 419: carrying the bowl with the bee, the pinch read 87.3 (rim alone 93.4-95.2): the bee was in the
            # jaws with the wall and fell out at the target. Let go and leave the bowl where it is
            S.decide(f"bowl rim reading {g:.1f} < {rim_min}: something besides the wall is in the jaws -> let go")
            S.gripper(BOWL_OPEN, "let go: not a clean rim pinch")
            vertical("retreat_rim_unclean", None, speed="slow")
            return False
        if held:
            break
        S.gripper(BOWL_OPEN, "open after a miss")
        vertical(f"retreat_rim{k + 1}", None, speed="slow")
        g0 = g0 - 0.006 * u; g1 = g1 - 0.006 * u     # a miss usually means the jaws sat outside the wall
    else:
        raise Abort("bowl rim not held after two tries")
    vertical("lift_bowl", gz + lift, speed="slow", tip_margin=0.0)
    line_to("carry_bowl", g1, gz + lift, y, speed=CARRY, tip_margin=0.0)
    line_to("lower_bowl", g1, gz + 0.001, y, speed="slow", tip_margin=0.0)
    S.gripper(BOWL_OPEN, "release the rim")
    vertical("rise_bowl", gz + 0.06, speed="slow")
    vertical("rise_bowl_high", None, speed="normal")
    dry_move("bowl", np.asarray(bowl["xy"]) + disp)
    return True


def fix_tilted_bowl(b, table_z, bowl_target):
    """The bowl rests tipped on the bee (att 354/355: blob 10.5 cm tall, bee under the raised side). Pinch the rim at
    its highest point (the propped-up side), lift it off, carry it 12 cm away from that side, set it down level,
    release; the normal reset follows."""
    P = np.asarray(b["points"], float)
    zt = float(b["z_top"])
    hi = P[P[:, 2] > zt - 0.012, :2]
    S.require(len(hi) >= 20, "tilted bowl: no clear top band")
    c = np.asarray(b["xy"][:2], float)
    p_hi = hi.mean(0)
    v = p_hi - c
    S.require(np.linalg.norm(v) > 0.02, f"tilted bowl at {r3(c)}: highest side unclear")
    u = v / np.linalg.norm(v)
    g0 = c + RIM_R * u                     # rim point on the raised side
    # where the bee is cannot be told (merged with the bowl): put the bowl at its target when that is >= 13 cm away,
    # else 12 cm off its raised side; carried 12 cm up so the bowl hanging from one rim point clears a 5 cm bee
    t = np.asarray(bowl_target[:2], float)
    c1 = t if dist(t, c) >= 0.13 else c - 0.12 * u
    g1 = c1 + RIM_R * u
    S.require(0.28 < np.hypot(*g0) < 0.62 and 0.28 < np.hypot(*g1) < 0.62, "tilted bowl: rim out of reach")
    S.require(0.36 <= c1[0] <= 0.60 and -0.24 <= c1[1] <= 0.10, f"tilted bowl: no room to set it down at {r3(c1)}")
    yaw = float(np.degrees(np.arctan2(u[1], u[0])))
    gz = zt + 0.009
    S.decide(f"bowl tipped (height {b['height']:.3f}, top side {r3(p_hi)}): pinch the raised rim at {r3(g0)} "
             f"z {gz:.3f}, lift 12 cm, set it down level at {r3(c1)}")
    y = to_hover("above_tilt_rim", g0, yaw)
    S.gripper(BOWL_OPEN, "open 58 mm around the raised rim")
    line_to("approach_tilt_rim", g0, gz + 0.06, y, speed="normal")
    line_to("descend_tilt_rim", g0, gz, y, speed="slow")
    g = S.gripper(1.0, "pinch the raised rim")
    held = grip_ok(g, BOWL_HELD_MAX, "tilted bowl rim")
    S.decide(f"tilted bowl rim: gripper reading {g:.1f} -> {'held' if held else 'EMPTY'}")
    if not held:
        S.gripper(BOWL_OPEN, "open after a miss")
        vertical("retreat_tilt_rim", None, speed="slow")
        S.require(False, "tilted bowl: rim not held")
    vertical("lift_tilt_bowl", gz + 0.12, speed="slow", tip_margin=0.0)
    line_to("carry_tilt_bowl", g1, gz + 0.12, y, speed=CARRY, tip_margin=0.0)
    put_z = float(table_z) + 0.069 + 0.009 + 0.004      # rim height of a level bowl (+4 mm: let it settle)
    line_to("lower_tilt_bowl", g1, put_z, y, speed="slow", tip_margin=0.0)
    S.gripper(BOWL_OPEN, "release the rim")
    vertical("rise_tilt_bowl", put_z + 0.06, speed="slow")
    S.home("home_after_tilt")


def bee_in_bowl_xy(bowl, side_c, table_z):
    """Wrist look into the bowl: bee centre xy, or None (no carry with the bee then)."""
    rim_z = float(bowl["z_top"])
    fit = rim_fit(bowl)
    c = np.array(fit["xy"] if fit and 0.05 < fit["r"] < 0.085 and dist(fit["xy"], side_c) < 0.02 else side_c, float)
    img, T = look_over(c, rim_z)
    ref, _ = refine_bowl(img, T, c, rim_z, 0.066)
    if ref is not None:
        c = ref
    bee, info = bee_in_wrist(img, T, c, rim_z, table_z + 0.018 + 0.03, dbg_name="wrist_bowl_bee_carry.jpg")
    S.summary(bee_in_bowl_carry=info)
    if bee is None or bee["len"] < 0.06:
        return None
    return list(map(float, bee["xy"][:2]))


def pick_bee_from_bowl(bowl, side_c, table_z):
    """Look into the bowl, find the bee, grasp across its short axis; after a miss look again (the bee or the
    bowl may have been nudged, att 32) instead of guessing a rotated grasp."""
    rim_z = float(bowl["z_top"])
    fit = rim_fit(bowl)
    c = np.array(fit["xy"] if fit and 0.05 < fit["r"] < 0.085 and dist(fit["xy"], side_c) < 0.02 else side_c, float)
    floor_hi, floor_lo = table_z + 0.018, table_z + 0.005
    gz = floor_hi + 0.046   # 6 mm lower than before: a high grip (reading 93.4) lost the bee in transit (att 110/111)
    xy, jaw = None, None
    for k in range(2):
        img, T = look_over(c, rim_z)
        if img is not None:
            cv2.imwrite(os.path.join(S.A, f"wrist_bowl_look{k + 1}.jpg"), cv2.resize(img, (640, 360)))
        ref, info = refine_bowl(img, T, c, rim_z, 0.066)
        S.summary(**{f"bowl_wrist{k + 1}": info})
        if ref is not None:
            c = ref
        bee, binfo = (None, dict(err="dry run")) if S.dry else bee_in_wrist(img, T, c, rim_z, floor_hi + 0.03,
                                                                           dbg_name=f"wrist_bowl_bee{k + 1}.jpg")
        S.summary(**{f"bee_in_bowl_wrist{k + 1}": binfo})
        standing = False
        if bee is None and not S.dry and binfo.get("area_frac", 0.0) >= 0.025:
            # att 412/413: the bee stood / leaned against the wall in the bowl, seen end-on: area 0.04-0.05 of the
            # interior (< 0.08) -> "not found", run stopped. Small region = standing bee: locate its top near the rim
            b2, i2 = bee_in_wrist(img, T, c, rim_z, rim_z - 0.02, dbg_name=f"wrist_bowl_bee{k + 1}s.jpg", min_frac=0.025)
            S.summary(**{f"bee_in_bowl_wrist{k + 1}s": i2})
            if b2 is not None:
                bee, binfo, standing = b2, i2, True
        # att 347/348 (dim evening light): the bee lay in the bowl centre but only a 0.088 x 0.032 strip of it was
        # segmented (off centre 0.015) -> rejected as "thin", run stopped. A thin strip near the bowl centre with a
        # bee's length is the bee; thin regions off centre (rim crescents) stay rejected
        dim_ok = bee is not None and bee["wid"] >= 0.025 and 0.07 <= bee["len"] <= 0.13 and dist(bee["xy"], c) <= 0.03
        if bee is not None and (bee["wid"] < 0.035 or bee["len"] < 0.06) and not dim_ok and not standing:
            binfo["err"] = f"no bee-sized region (thin {bee['len']:.3f} x {bee['wid']:.3f}: not a bee)"
            S.summary(**{f"bee_in_bowl_wrist{k + 1}": binfo})
            bee = None
        if bee is not None and standing:
            # grip the upper end: jaws tangential (fingers clear of the wall at <= 4 cm off centre), tool near rim height
            xy = np.asarray(bee["xy"], float)
            if dist(xy, c) > 0.04:
                xy = c + (xy - c) * 0.04 / dist(xy, c)
            rad = np.degrees(np.arctan2(xy[1] - c[1], xy[0] - c[0])) if dist(xy, c) > 0.005 else bee["long"]
            jaw = rad + 90.0
            gz = max(gz, rim_z - 0.005)
            S.decide(f"bee stands in the bowl (end-on region {binfo.get('area_frac')}): grip its top at {r3(xy)}")
        elif bee is not None:
            xy = np.asarray(bee["xy"], float)
            if dist(xy, c) > 0.03:                    # keep the fingers off the wall
                xy = c + (xy - c) * 0.03 / dist(xy, c)
            jaw = bee["long"] + 90.0
        elif xy is None:
            S.require(S.dry or not ("no bee-sized region" in str(binfo.get("err"))),
                      f"bee not seen on the table nor in the bowl ({binfo.get('err')})")
            xy, jaw = c, np.degrees(np.arctan2(-c[1], -c[0])) + 90.0
        elif k > 0:
            jaw += 30.0                               # no fresh detection: vary the angle a little
        S.decide(f"bee in bowl: grasp at {r3(xy)} jaw yaw {jaw:.0f} tool z {gz:.3f} (floor <= {floor_hi:.3f}), try {k + 1}")
        ok, g, y = pick_bee(xy, jaw, gz, BEE_BOWL_OPEN, f"bee_bowl{k + 1}", held_max=88.0)
        if ok:
            # gripped at its top a standing bee hangs ~3.5 cm lower than a lying one: release that much higher so its
            # lower end does not push into the table
            return True, g, y, gz - floor_lo + (0.035 if standing else 0.0)
        gz -= 0.004
    return False, None, None, None


def tilted_bee(b):
    """A tilted / standing bee blob as the bee: xy from its top band (directly above its base; the side depth smear
    pulled the blob centre of a standing bee 5 cm away from where it was dropped, att 218)."""
    P = np.asarray(b["points"], float)
    top = P[P[:, 2] > float(b["z_top"]) - 0.02, :2]
    xy = r3(np.median(top, 0), 4) if len(top) >= 20 else b["xy"]
    return dict(b, xy=xy, xy_blob=b["xy"])


MERGE_PUSH = 0.035       # att 725: 2.5 cm aimed out landed 1.0 cm from target (1.5 short), still merged


def merged_layout(lay, err):
    """Every layout-vote frame fails both objects while our own split measurement has both within tolerance: bowl and
    bee read as one side blob (att 229/230: extent 0.23, the orchestrator found neither)."""
    bad = lay.get("bad", {})
    n = lay.get("n", 0)
    # att 333: votes 1/3 (both failed in 2 of 3 frames) -> not "merged", the plain correction found the bee already
    # within 1.6 cm and did not move it; the next votes were 0/3. Most frames failing both objects counts as merged.
    # att 724: votes 2/3 (both failed in 1 frame) and the orchestrator found neither -> any frame failing both counts
    return bool(n >= 2 and bad.get("bowl", 0) >= 1 and bad.get("bee", 0) >= 1
                and all(err.get(k) is not None and err[k] <= TOL for k in ("bowl", "bee")))


def near_bowl_yaw(targets):
    """Jaw yaw for the final bee placement. Default PLACE_YAW (long axis along y, across the side camera's smear); when
    bowl and bee targets are under 22 cm apart, the jaw points at the bowl so the bee's long axis lies across that
    direction: its 10 cm length then no longer closes the gap to the rim (att 229/230: merged side blobs)."""
    v = np.subtract(targets["bowl"], targets["bee"])
    if float(np.linalg.norm(v)) >= 0.22:
        return None
    return float(np.degrees(np.arctan2(v[1], v[0])))


def bowl_aim(t):
    """Bowl target, 1.5 cm further from the base when it is near the base (x < 0.43): there the side blob of the bowl
    joins low clutter near the base in some frames and its centre jumps 3-4 cm toward the base (att 346: 0.365 vs
    0.408; att 350: 0.388 vs 0.428, 3.7 cm off). 1.5 cm costs little when the blob reads true."""
    # att 352: target 0.407, aimed 0.422, the blob read 0.440 in every frame (3.3 cm; ~1.8 cm without the nudge):
    # the near-base bias goes either way -> nudge off (kept as a hook)
    t = list(map(float, t[:2]))
    # att 366/367: target (0.408, 0.047) put the rim 0-1 cm from flat clutter at (0.33-0.34, -0.04): the blobs merged
    # in ~1/3 of the frames (bowl read (0.373, 0.021), orchestrator "4.4 cm off"). Keep a 3 cm gap from remembered
    # low clutter: shift the aim away from the nearest clutter point, at most 1.5 cm.
    # att 641: one 84-point blob seen in a single frame shifted the aim 1.5 cm (final 2.45 cm off). Use only clutter
    # seen in >= 2 frames (centroids within 3 cm of each other).
    cents = [np.mean(c, 0) for c in CLUTTER]
    steady = [c for i, c in enumerate(CLUTTER)
              if sum(1 for j, d in enumerate(cents) if j != i and np.linalg.norm(d - cents[i]) < 0.03) >= 1]
    if steady:
        P = np.concatenate(steady)
        d = np.linalg.norm(P - np.asarray(t), axis=1)
        i = int(np.argmin(d))
        gap = float(d[i]) - 0.075
        # att 641/664/672/677: the full 3 cm gap shifted aims 1.5 cm and the bowls ended 2.2-2.9 cm off; merges were
        # seen at 0-1 cm gaps (att 366/367): keep 2 cm
        if gap < 0.02 and d[i] > 1e-6:
            u = (np.asarray(t) - P[i]) / d[i]
            sh = min(0.015, 0.02 - gap)
            t = [t[0] + sh * u[0], t[1] + sh * u[1]]
    return t


BOWL_NUDGE = 0.0
CLUTTER = []      # xy points of low flat blobs seen in this run (side camera)


def bee_aim(targets):
    """Bee target, nudged 1.5 cm away from the bowl target when the two are under 20 cm apart: the side camera then
    joins bowl and bee into one blob (extent 0.28) and the orchestrator's check misreads both (att 190: bowl 4.1 cm
    off by its blob, 0.7 cm by ours). 1.5 cm uses little of the 4 cm tolerance (placements land 1-2 cm off)."""
    t, b = np.asarray(targets["bee"], float), np.asarray(targets["bowl"], float)
    d = float(np.linalg.norm(t - b))
    # att 724/725 (targets 18.3 cm apart, bee at -120 deg from the bowl): 1.5 cm was not enough, the blobs merged at
    # 18.6-19.4 cm; merges were seen up to 0.199 at -110..-130 deg -> 2.5 cm below 0.21 m
    if d >= 0.21 or d < 1e-6:
        return list(targets["bee"])
    return r3(t + 0.025 * (t - b) / d, 4)


def temp_yaw(dest, bowl_target):
    """Jaw yaw for putting the bee down at a temporary spot: the later pick uses the same jaw axis, so make it
    tangential to the bowl's target (att 156: a bee parked with its jaw axis toward the bowl target was blocked
    after the bowl moved there; clearing the bowl twice cost ~4 min)."""
    v = np.subtract(bowl_target[:2], dest[:2])
    return float(np.degrees(np.arctan2(v[1], v[0]))) + 90.0


def temp_spot(avoid, table_bounds=((0.40, 0.56), (-0.18, 0.05))):
    """Free spot for the bee, well inside the library's release zone (x 0.36-0.60, y -0.22..0.08): two bees were
    lost off the curtain-side table edge (y +0.19..+0.25) from (0.30, 0.10) and (0.36, 0.09) (operator, att 119)."""
    best = None
    for x in np.arange(table_bounds[0][0], table_bounds[0][1] + 1e-6, 0.01):
        for y in np.arange(table_bounds[1][0], table_bounds[1][1] + 1e-6, 0.01):
            r = np.hypot(x, y)
            if not 0.31 <= r <= 0.58:
                continue
            m = min(dist((x, y), a) for a in avoid)
            score = min(m, 0.20) + 0.05 * x          # enough clearance, then nearer the side camera (steadier depth)
            if m >= CLEAR_BOWL and any(x < a[0] - 0.03 and abs(y - a[1]) < 0.09 for a in avoid):
                score -= 0.06                        # hidden behind the bowl from the side camera (att 253): take
                                                     # an unhidden spot with clearance >= CLEAR_BOWL if one exists
            if best is None or score > best[2]:
                best = (m, (float(x), float(y)), score)
    return best[:2]


# ------------------------------------------------------------------ main
BEE_EXT = [0.0]   # largest bee side-blob extent seen in the last measure()
BEE_EXT_MAX = 0.145   # the orchestrator's bee prior allows extent <= 0.15 (config/tasks/bee_in_bowl.yaml)


def measure(targets, tag):
    S.home(f"home_{tag}")
    obs = {"bowl": [], "bee": []}
    BEE_EXT[0] = 0.0
    RAW.clear()
    last = {}
    for i in range(3):
        bowl, bee = perceive_retry(f"{tag}_{i}")
        # keep the last frame that saw each object: a correction needs the bee even when the final frame missed it
        # (att 335: "correction pass for ['bee']" did nothing, the last frame had no bee)
        for name, b in (("bowl", bowl), ("bee", bee)):
            if b is not None:
                obs[name].append(b.get("xy_blob", b["xy"]))   # the orchestrator's metric (blob centre) for both
        if bee is not None:
            BEE_EXT[0] = max(BEE_EXT[0], float(max(bee["extent"])))
        last.update({k: v for k, v in (("bowl", bowl), ("bee", bee)) if v is not None})
        bowl, bee = last.get("bowl"), last.get("bee")
    S.summary(**{f"measure_{tag}": {k: [r3(v) for v in o] for k, o in obs.items()}})
    meas, err = {}, {}
    for name in ("bowl", "bee"):
        b = np.median(obs[name], 0) if obs[name] else None
        meas[name] = None if b is None else r3(b)
        if b is not None:
            b = {"xy": b}
            if name == "bowl" and bowl is not None:
                bowl = dict(bowl, xy_blob=r3(b["xy"], 4))   # keep the rim centre in xy for grasping
            if name == "bee" and bee is not None:
                bee = dict(bee, xy_blob=r3(b["xy"], 4))         # keep the point-median xy for grasping
        err[name] = None if b is None else round(dist(b["xy"], targets[name]), 4)
    lay = layout_votes(targets)
    return bowl, bee, meas, err, lay


def layout_votes(targets):
    """The orchestrator's own check (rlpj.perception.layout.check with the task's size priors) on every raw
    side measurement taken: fraction passing and the objects that failed."""
    out = dict(n=0, ok=0, bad={})
    if S.dry:
        out.update(n=1, ok=1); return out
    try:
        from rlpj import config
        from rlpj.perception import layout
        task = config.load()["task"]
        for blobs in RAW:
            ok, rep = layout.check(task, targets, blobs)
            out["n"] += 1; out["ok"] += int(ok)
            for o, v in rep["objects"].items():
                if not v["ok"]:
                    out["bad"][o] = out["bad"].get(o, 0) + 1
    except RunTimeLimit:
        raise
    except Exception as e:
        out["err"] = repr(e)[:200]
    S.log(f"layout votes: {out}")
    return out


ORACLE_TIMEOUT = 3.0       # s per controller FK call (att 220 hung 2 h 45 min at full CPU with no log line before the
                           # first motion: the only unbounded call there is this RPC, which also holds the client lock)


def _robust_oracle(s):
    """The robot's FK service failed once mid-run (KServerException, attempt 27): retry it before giving up. Every call
    runs in a worker thread with a time limit; after a timeout the oracle is switched off for the rest of the run
    (the planner then uses its fitted model, as it does when the controller refuses a joint set)."""
    import time
    import concurrent.futures as cf
    orig = s.planner.oracle
    if orig is None:
        return
    pool = cf.ThreadPoolExecutor(max_workers=1)
    state = dict(off=False)

    def oracle(q):
        if state["off"]:
            raise RuntimeError("FK oracle switched off after a timeout")
        for i in range(4):
            try:
                return pool.submit(orig, q).result(timeout=ORACLE_TIMEOUT)
            except cf.TimeoutError:
                state["off"] = True
                s.log(f"FK oracle did not answer within {ORACLE_TIMEOUT:.0f} s: switched off, fitted model from now on")
                raise RuntimeError("FK oracle timeout")
            except RunTimeLimit:
                raise
            except Exception as e:
                if i == 3:
                    raise
                s.log(f"FK oracle failed ({e!r}); retry {i + 1}")
                time.sleep(0.3 * (i + 1))
    s.planner.oracle = oracle


class RunTimeLimit(Abort):
    """The run's own time limit (never swallowed by the handlers that turn an Abort into 'not reachable')."""


RUN_LIMIT_S = 17 * 60      # the orchestrator kills a run at 20 min (collection.attempt_timeout_s); stop cleanly before


def _time_limits(s):
    """Bound the whole run: an Abort after RUN_LIMIT_S (raised in the main thread between bytecodes, so it also ends a
    pure-Python loop; the session then returns the arm home). No periodic faulthandler stack dump: dumping the other
    threads while the kortex client threads run crashed the process (att 280: segfault 2 s after the 5-min dump, the
    watchdog could not take the arm home and the next run started over the table)."""
    import signal

    def _alarm(signum, frame):
        signal.alarm(5)        # fire again until it reaches main (a library handler may swallow one, e.g. the oracle's)
        raise RunTimeLimit(f"run time limit {RUN_LIMIT_S} s reached")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(RUN_LIMIT_S)


def main():
    global S, TOL, REDO, REDO_BOWL, SKIP_BEE
    try:                                   # the orchestrator's tolerance (task yaml); was 0.03, now 0.04
        from rlpj import config
        TOL = float(config.load()["task"]["layout"]["tolerance"])
    except RunTimeLimit:
        raise
    except Exception:
        pass
    # corrections only when the error eats most of the margin (each costs ~100 s); 0.03 -> 0.020 / 0.024 as before
    REDO = max(REDO, 0.65 * TOL)
    REDO_BOWL = max(REDO_BOWL, 0.75 * TOL)
    SKIP_BEE = max(SKIP, 0.6 * TOL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()
    tg = a.targets
    targets = json.load(open(tg)) if os.path.exists(tg) else json.loads(tg)
    targets = {k: [float(v[0]), float(v[1])] for k, v in targets.items()}
    with RobotSession(program="reset") as s:
        S = s
        _robust_oracle(s)
        _time_limits(s)
        s.summary(targets=targets, measured=None, errors=None, claimed_outcome="fail", first_failed_check=None)
        s.view.set_scene(targets=[dict(xy=v, z_top=0.0, r=0.03, label=f"{k} target") for k, v in targets.items()])
        try:
            run(targets)
        except Abort as e:
            import signal
            signal.alarm(0)    # the session's return home must not be interrupted by the re-armed time limit
            s.summary(first_failed_check=str(e)[:300])
            raise


def unpack(s, bowl, bowl_c, targets, table_z):
    """Bee in the bowl: take it out, put it at its target (or a free spot if the bowl blocks it)."""
    s.phase("unpack_bee")
    s.decide(f"no bee-sized blob on the table -> bee assumed in the bowl at {r3(bowl_c)}")
    if (dist(targets["bee"], bowl_c) < CLEAR_BOWL and dist(targets["bee"], targets["bowl"]) >= CLEAR_BOWL + 0.01
            and dist(bowl_c, targets["bowl"]) > SKIP and not s.dry):
        # att 320/328/374 (500+ s): a temp spot for the bee ended up next to the bowl's target and blocked the bee's
        # pick -> the bowl went away and back. Carry the bowl with the bee in it to its target first (the bee stays
        # in: att 306/356), rim grasp on the side away from the bee, then take the bee out straight to its target
        bxy = bee_in_bowl_xy(bowl, bowl_c, table_z)
        if bxy is not None:
            s.decide(f"bee target blocked by the bowl: carry the bowl with the bee (at {r3(bxy)}) to its target first")
            moved = move_bowl(bowl, bowl_c, bowl_aim(targets["bowl"]), table_z, bxy, disp_ref=bowl.get("xy_blob"),
                              rim_min=92.0)
            s.home("home_after_bowl_with_bee")
            b2, e2 = perceive_retry("after_bowl_with_bee", need_bee=False)
            s.require(b2 is not None, "bowl lost after carrying it with the bee")
            bowl, bowl_c = b2, list(b2["xy"])
            if e2 is not None and dist(e2["xy"], bowl_c) > 0.10:
                # the bee is on the table now (fell out / was never in): the placing steps take it from there
                s.decide(f"bee on the table at {r3(e2['xy'])} after the bowl move: no unpack")
                return list(e2["xy"])
    if dist(targets["bee"], bowl_c) < CLEAR_BOWL:
        m, dest = temp_spot([bowl_c, targets["bowl"]])
        s.decide(f"bee target blocked by the bowl: bee to free spot {r3(dest)} (clearance {m:.3f})")
        pyaw = temp_yaw(dest, targets["bowl"])
    else:
        # straight to the target: same aim and yaw as a table placement (att 333: put with PLACE_YAW, long axis
        # toward the bowl target 19 cm away -> one merged side blob, the orchestrator found no bee)
        dest = bee_aim(targets)
        pyaw = near_bowl_yaw(targets)
        pyaw = PLACE_YAW if pyaw is None else pyaw
    for attempt in range(2):
        ok, g, yaw, d_bottom = pick_bee_from_bowl(bowl, bowl_c, table_z)
        s.require(ok, "could not grasp the bee in the bowl after two tries")
        release = table_z + d_bottom + 0.012
        if place_bee(dest, pyaw, release, "bee_from_bowl"):
            break
        s.decide("the bee fell out during the carry (probably back into the bowl): take it out again")
    else:
        s.require(False, "the bee fell out of the gripper twice while carrying it out of the bowl")
    bee_c = list(dest)
    if s.dry:      # simulated bee on the table at dest
        P = np.random.default_rng(1).normal(0, 1, (600, 3)).astype(np.float32) * np.array([0.025, 0.014, 0.008], np.float32)
        P += np.array([dest[0], dest[1], table_z + 0.03], np.float32)
        DRY_SCENE["bee"] = dict(xy=list(dest), z_top=table_z + 0.043, height=0.043, extent=[0.1, 0.06], n=600, points=P)
    return list(dest)


def synthetic_bee(w, table_z):
    """Blob-like dict for a bee found with the wrist camera (points along its long axis for long_axis())."""
    a = np.radians(w["long"]); u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]])
    P = [np.r_[np.asarray(w["xy"]) + t * u + d * v, table_z + 0.03] for t in np.linspace(-0.045, 0.045, 10) for d in (-0.02, 0, 0.02)]
    return dict(xy=r3(w["xy"], 4), xy_blob=r3(w["xy"], 4), z_top=table_z + 0.045, height=0.045, extent=[0.1, 0.06], n=999,
                points=np.array(P, np.float32))


def search_bee(spots, table_z, bowl_xy):
    """Look straight down with the wrist camera at candidate spots (where we put the bee, small side blobs)."""
    seen = []
    spots = [q for q in spots if q is not None][:3]
    if bowl_xy is not None:
        # the side camera looks along -x from the far end: a bee just behind the bowl (smaller x) is hidden from it
        # (att 253/254: parked at (0.40, -0.10) behind the bowl at (0.53, -0.16), never seen again)
        b = np.asarray(bowl_xy, float)
        spots += [list(b + d) for d in ((-0.12, 0.0), (-0.11, -0.06), (-0.11, 0.06))]
    for xy in spots:
        if any(dist(xy, q) < 0.04 for q in seen) or not 0.30 <= np.hypot(*xy) <= 0.62:
            continue
        if bowl_xy is not None and dist(xy, bowl_xy) < 0.08:
            continue
        seen.append(xy)
        # a looked-at spot is only a guess: accept a bee region up to 8 cm from it (att 253 rejected the bee at
        # 4.5 cm from the parked spot)
        w = wrist_bee_on_table(np.asarray(xy, float), table_z, bowl_xy, 8 + len(seen), max_off=0.08)
        if w is not None and bowl_xy is not None and dist(w["xy"], bowl_xy) < 0.135:
            S.decide(f"wrist search: region at {r3(w['xy'])} is {dist(w['xy'], bowl_xy):.3f} from the bowl: its shadow, skipped")
            w = None
        if w is not None:
            S.decide(f"wrist search: bee found at {r3(w['xy'])} (looked at {r3(xy)})")
            return synthetic_bee(w, table_z)
        if len(seen) >= 5:
            break
    return None


def sweep_bee(table_z, bowl_xy, max_looks=26):
    """att 679-691: the bee vanished from the side camera's view for 13 runs. Before giving up, look straight down with
    the wrist camera over the whole reachable table (10 cm grid, each look covers ~9 cm around its point), nearest
    first, skipping the bowl. Returns a synthetic bee or None."""
    pts = []
    for x in np.arange(0.30, 0.621, 0.10):
        for y in np.arange(-0.28, 0.241, 0.10):
            r = float(np.hypot(x, y))
            if not 0.30 <= r <= 0.62:
                continue
            if bowl_xy is not None and dist((x, y), bowl_xy) < 0.07:
                continue
            pts.append([float(x), float(y)])
    pts.sort(key=lambda q: dist(q, START_DROP_XY))
    S.decide(f"wrist sweep of the table for the bee: {len(pts[:max_looks])} looks")
    for i, xy in enumerate(pts[:max_looks]):
        w = wrist_bee_on_table(np.asarray(xy, float), table_z, bowl_xy, 40 + i, max_off=0.09)
        if w is not None and bowl_xy is not None and dist(w["xy"], bowl_xy) < 0.135:
            # att 692-695: the "bee" found 9-12 cm from the bowl centre followed the bowl from run to run (its
            # shadow/rim on the curtain side) and a grasp there closed empty: not a bee
            S.decide(f"wrist sweep: region at {r3(w['xy'])} is {dist(w['xy'], bowl_xy):.3f} from the bowl: its shadow, skipped")
            w = None
        if w is not None:
            S.decide(f"wrist sweep: bee found at {r3(w['xy'])} (look {i + 1} at {r3(xy)})")
            return synthetic_bee(w, table_z)
    return None


STD_HOME_Q = np.array([0.0, 15.0, 230.0, 0.0, 55.3, 90.0])   # the usual start (camera view over the table)


def go_standard_home(s):
    """A run that starts away from the usual home (att 281: a crashed run left the arm over the table, the side
    camera saw it merged with the bowl) first moves there, and that pose becomes the run's home."""
    q = np.asarray(s.feedback()["q"], float)
    d = float(np.max(np.abs(wrap180(q - STD_HOME_Q))))
    if d <= 5.0:
        return
    s.decide(f"start joints {r3(q, 1)} are {d:.0f} deg from the usual home: move there first")
    if s.feedback()["gripper"] < 50.0 or s.dry:
        T = tool_T()
        if T[2, 3] < float(s.table_z) + 0.18:              # low over the table: straight up first
            vertical("start_rise", float(s.table_z) + 0.22, speed="slow")
    if s.feedback()["gripper"] < 50.0 and not s.dry:
        # att 405/406: the arm was left with the open fingers at the curtain-side keep-in limit (y <= 0.30); the
        # model check counts the open fingertips -> every move from there failed. Closed fingers sit ~1 cm further in
        s.gripper(1.0, "close the empty gripper: narrower fingers for the move home (keep-in)")
    s.move_joint("go_standard_home", STD_HOME_Q, "slow")
    s.q_home = STD_HOME_Q.copy()


def run(targets):
    s = S
    table_z = float(s.table_z)
    s.phase("perceive")
    go_standard_home(s)
    g_start = float(s.feedback()["gripper"])
    # bee in or on the gripper (operator 2026-11-06): closed with a reading < 99.0, the bee colour at the fingers, or a
    # squeeze reading < 99.0 -> put it down straight below home before anything else (no carry)
    fy = stuck_frac("start", full=True)
    on = (g_start > 50.0 and g_start < SQUEEZE_EMPTY) or fy >= STUCK_FRAC
    if on and fy < STUCK_FRAC and not s.dry:
        # att 293/294: after the fault was cleared the empty gripper sat half closed (61.4; a held bee reads 67-90,
        # the bee was on the table, no colour at the fingers). Closing never releases anything: close fully and
        # judge by that reading (empty closed = 99.1)
        s.gripper(1.0, "close fully to test the jaws (start reading below 99)")
        g2 = float(s.feedback()["gripper"])
        s.decide(f"start: reading {g_start:.1f}, fully closed {g2:.1f}")
        on = g2 < SQUEEZE_EMPTY
    if not on and g_start <= 50.0:
        gq = squeeze_test("start")
        on = gq < SQUEEZE_EMPTY
        if on and gq >= EMPTY_MIN:
            S.gripper(1.0, "close again: something thin in the jaws")
    if on and not s.dry:
        s.phase("start_drop")
        release_at_start(table_z)
        s.home("home_after_start_drop")
    holding = False
    color, _, _ = s.side.latest()
    if color is not None:
        cv2.imwrite(os.path.join(s.A, "side_before.jpg"), cv2.resize(color, (640, 360)))
    bowl, bee = perceive_retry("start", need_bee=False)
    if bee is None and bowl is not None:
        b2, e2 = perceive_retry("start_check", need_bee=True, tries=2)
        if b2 is not None:
            bowl, bee = b2, e2
    if bowl is None and TILTED_BOWL and not s.dry:
        fix_tilted_bowl(max(TILTED_BOWL, key=lambda b: b["n"]), table_z, targets["bowl"])
        bowl, bee = perceive_retry("after_tilt", need_bee=False)
    s.require(bowl is not None, f"bowl not found (blobs: {LOG['start']})")
    if bee is None and TILTED and not holding:
        # a bee on its side / standing (0.052-0.075 tall) is still the bee on the table, not one in the bowl (att 186
        # grasped an empty bowl twice); it goes to its target and the flip step at the end lays it flat
        bee = tilted_bee(TILTED[0])
        s.decide(f"no flat bee, but a tilted bee-sized blob at {r3(bee['xy'])} (height {bee['height']:.3f}): that is the bee")
    bowl_c = list(bowl["xy"])
    bee_c = None if bee is None else list(bee["xy"])
    s.require(max(bowl["extent"]) < 0.19, f"bowl blob {bowl['extent']} too large: bee touching the bowl? reset by hand")
    for name, t in targets.items():
        r = np.hypot(*t)
        s.require(0.28 <= r <= 0.60, f"target {name} {t} out of reach (r {r:.2f})")
    for name, b in (("bowl", bowl), ("bee", bee)):
        if b is not None:
            xy = b["xy"][:2]
            s.require(np.hypot(*xy) <= 0.72 and reach_ok(xy, table_z + 0.04) and hover_z(xy) is not None,
                      f"{name} at {r3(xy)} is out of reach (tool-down IK fails at grasp/hover height): reset by hand")
    s.summary(objects=dict(bowl=_brief(bowl), bee=_brief(bee)))

    # ---- bee still in the gripper (an earlier run stopped while carrying it)
    if holding and bee is None:
        s.phase("place_held_bee")
        dest = targets["bee"]
        if dist(dest, bowl_c) < CLEAR_BOWL:
            m, dest = temp_spot([bowl_c, targets["bowl"]])
        s.decide(f"gripper reads {g_start:.1f} at start: holding the bee -> put it down at {r3(dest)}")
        # in the jaws (reading 40-97): bee bottom ~1 cm above the table at +0.065; the old +0.09 dropped it 3-4 cm and
        # it landed on its side (att 183). +0.09 stays for a bee hanging below the tips (found by the stuck check)
        rz = table_z + (0.065 if 40.0 < g_start < 97.0 else 0.09)
        place_bee(dest, PLACE_YAW if dest is targets["bee"] else temp_yaw(dest, targets["bowl"]), rz, "held_bee",
                  check_empty=False)   # a bee on the OUTSIDE of a closed finger reads 98.7 (att 209: declared lost, the
                                       # arm went home with it on the finger and it fell off somewhere)   # a bee hanging below the tips touches down here
        bee_c = list(dest)
        if s.dry:
            P = np.random.default_rng(1).normal(0, 1, (600, 3)).astype(np.float32) * np.array([0.025, 0.014, 0.008], np.float32)
            P += np.array([dest[0], dest[1], table_z + 0.03], np.float32)
            DRY_SCENE["bee"] = dict(xy=list(dest), z_top=table_z + 0.043, height=0.043, extent=[0.1, 0.06], n=600, points=P)
    # ---- undo the task: bee in the bowl
    elif bee is None:
        cands = [list(b["xy"]) for b in sorted(OTHERS, key=lambda b: -b["n"])]
        try:
            bee_c = unpack(s, bowl, bowl_c, targets, table_z)
        except Abort as e:
            # att 693: a rim highlight taken for a standing bee, both grasps closed empty (99.6): the bee is not in
            # the bowl -> search the table (it lay near the curtain-side edge, unseen by the side camera)
            if "not seen" not in str(e) and "could not grasp the bee in the bowl" not in str(e):
                raise
            s.decide(f"no bee in the bowl ({e}); search small side blobs with the wrist camera: {cands[:3]}")
            bee = search_bee(cands, table_z, bowl_c)
            if bee is None:
                bee = sweep_bee(table_z, bowl_c)
            s.require(bee is not None, "bee not found on the table, in the bowl, nor by the wrist search and sweep")
            bee_c = list(bee["xy"])
            s.home("home_after_search")      # side perception below needs the arm out of the way
        bee = None            # position known only from the place; re-perceive before grasping it again

    # ---- order the moves
    s.phase("place")
    need_bowl = dist(bowl_c, targets["bowl"]) > SKIP
    blocked_bowl_t = dist(bee_c, targets["bowl"]) < CLEAR_BOWL
    blocked_bee_t = dist(bowl_c, targets["bee"]) < CLEAR_BOWL
    plan = []
    if need_bowl and blocked_bowl_t:
        if blocked_bee_t:
            plan = ["bee_temp", "bowl", "bee"]
        else:
            plan = ["bee", "bowl"]
    else:
        plan = ["bowl", "bee"]
    s.decide(f"order {plan}: bowl {r3(bowl_c)} -> {targets['bowl']}, bee {r3(bee_c)} -> {targets['bee']}")
    s.summary(plan=plan)

    moved = bee_c is not None and bee is None      # bee was just put down: look again
    for step in plan:
        if moved or bee is None:
            s.home(f"home_before_{step}")
            bowl, bee = perceive_retry(f"before_{step}")
        moved = False
        s.require(bowl is not None, "bowl lost")
        if bee is None and TILTED:   # a bee on its side / standing (att 186)
            bee = tilted_bee(TILTED[0])
            s.decide(f"tilted bee-sized blob at {r3(bee['xy'])} (height {bee['height']:.3f}) taken as the bee")
        if bee is None:              # not on the table: it may be in the bowl after all (att 30)
            cands = ([bee_c] if bee_c is not None else []) + [list(b["xy"]) for b in sorted(OTHERS, key=lambda b: -b["n"])]
            s.decide(f"bee not seen by the side camera: wrist search at {[r3(c) for c in cands[:3]]}")
            bee = search_bee(cands, table_z, None if bowl is None else bowl["xy"])
            if bee is not None:
                s.home(f"home_after_search_{step}")
            if bee is None:
                s.decide("bee not on the table: look into the bowl")
                unpack(s, bowl, list(bowl["xy"]), {"bowl": targets["bowl"], "bee": list(temp_spot([bowl["xy"], targets["bowl"]])[1])}, table_z)
                s.home(f"home_after_unpack_{step}")
                bowl, bee = perceive_retry(f"after_unpack_{step}")
        s.require(bowl is not None, "bowl lost")
        s.require(bee is not None, "bee lost (not on the table nor in the bowl)")
        bowl_c, bee_c = list(bowl["xy"]), list(bee["xy"])
        if step == "bowl":
            if dist(bowl_c, targets["bowl"]) > SKIP and not s.dry:     # arm is at home here (start or after a move)
                # the side centre jitters 1-2 cm between frames (att 35 placed the bowl 2.6 cm off): median of 3
                # the orchestrator measures the blob centre (xy_blob), which sits up to ~2 cm from the rim
                # circle (att 36: rim 7.8 mm, orchestrator 3.2 cm): carry by target - median blob centre
                xs, xb = [bowl_c], [bowl.get("xy_blob", bowl_c)]
                for i in range(6):          # blob centre jitters 2-3 cm in x (att 56 landed 3.4 cm off): median of 7
                    b2, _ = perceive(f"bowl_centre_{i}")
                    if b2 is not None and dist(b2["xy"], bowl_c) < 0.05:
                        xs.append(list(b2["xy"])); xb.append(list(b2.get("xy_blob", b2["xy"])))
                bowl_c = r3(np.median(xs, 0), 4)
                blob_c = r3(np.median(xb, 0), 4)
                s.decide(f"bowl centre median of {len(xs)}: rim {bowl_c}, blob {blob_c}")
            else:
                blob_c = bowl.get("xy_blob", bowl_c)
            # att 434: a clutter shift moved the aim 1.5 cm toward the bowl: 1.5 cm from the aim but 2.7 cm from the
            # target -> skipped, single frames then read 4 cm off. Skip only when close to both.
            # att 613: blob median 1.0 cm from the target but rim median 3.4 cm; the final frames then read the blob
            # 1.3-3.4 cm off (3.0 cm measured). When blob and rim disagree by > 2 cm, judge the skip by their mean
            # (as move_bowl does for the carry).
            skip_c = blob_c
            if dist(blob_c, bowl_c) > 0.02:
                skip_c = r3((np.asarray(blob_c[:2], float) + np.asarray(bowl_c[:2], float)) / 2, 4)
                s.decide(f"bowl: blob {r3(blob_c)} and rim {r3(bowl_c)} disagree by {dist(blob_c, bowl_c):.3f}: "
                         f"judge the skip by the mean {skip_c}")
            if dist(skip_c, bowl_aim(targets["bowl"])) > SKIP or dist(skip_c, targets["bowl"]) > 0.022:
                move_bowl(bowl, bowl_c, bowl_aim(targets["bowl"]), table_z, bee_c, disp_ref=blob_c)
                moved = True
        else:
            pyaw = None
            if step == "bee_temp":
                m, dest = temp_spot([bowl_c, targets["bowl"]])
                pyaw = temp_yaw(dest, targets["bowl"])
                s.decide(f"bee to a free spot {r3(dest)} first (clearance {m:.3f}, jaw yaw {pyaw:.0f})")
            else:
                dest = bee_aim(targets)
                pyaw = near_bowl_yaw(targets)
            # skip by the blob centre the orchestrator measures: the point median read 2.6-2.8 cm off where the
            # blob centre read 0.9-2.1 cm (att 141/142: both re-picks gained nothing)
            if dist(bee.get("xy_blob", bee_c), dest) > (SKIP_BEE if step == "bee" else SKIP):
                try:
                    ok = move_bee_from_table(bee, dest, table_z, bowl_c, place_yaw=pyaw)
                except BeeBlocked as e:
                    # move the bowl out of the way (rim grasp on the side away from the bee), then grip the bee
                    s.decide(f"{e}: move the bowl away first")
                    s.home(f"home_blocked_{step}")
                    prev_bee = bee
                    bowl, bee = perceive_retry(f"blocked_{step}")
                    if bee is None and prev_bee is not None:
                        # att 692: the bee lay near the curtain-side edge (0.55, 0.15), seen only by the wrist camera:
                        # keep that position (nothing has moved it)
                        s.decide(f"side camera does not see the bee: keep the wrist position {r3(prev_bee['xy'])}")
                        bee = prev_bee
                    s.require(bowl is not None and bee is not None, "bowl or bee lost before clearing the bowl")
                    spot, m = bowl_spot(bee["xy"], dest, targets["bowl"])
                    s.decide(f"bowl {r3(bowl['xy'])} -> spot {r3(spot)} (clearance {m:.3f} from the bee and its destination)")
                    move_bowl(bowl, list(bowl["xy"]), spot, table_z, bee["xy"], disp_ref=bowl.get("xy_blob"))
                    if "bowl" not in plan[plan.index(step) + 1:]:
                        plan.append("bowl")
                    s.home(f"home_cleared_{step}")
                    prev_bee = bee
                    bowl, bee = perceive_retry(f"cleared_{step}")
                    if bee is None and prev_bee is not None:
                        # att 694: a re-search look at the known spot missed the bee (near the reach limit); the
                        # bowl was carried away from it, so it has not moved: keep the position, the grasp refines it
                        s.decide(f"side camera does not see the bee after clearing: keep {r3(prev_bee['xy'])}")
                        bee = prev_bee
                    s.require(bowl is not None and bee is not None, "bowl or bee lost after clearing the bowl")
                    ok = move_bee_from_table(bee, dest, table_z, list(bowl["xy"]), place_yaw=pyaw)
                s.require(ok, "bee not grasped after two tries")
                moved = True
                bee_c = list(dest)    # where the bee now is: the wrist search looks here first (att 143 looked
                                      # at the old spot and stopped although the bee lay at the temp spot)

    # ---- measure, one correction pass
    s.phase("verify")
    bowl, bee, meas, err, lay = measure(targets, "final")
    s.summary(measured=meas, errors=err, layout_votes=lay)
    for t in range(3):
        # a bee on its side or standing reads 5.5-6.3 cm: over the orchestrator's prior (<= 0.055) -> "bee: not found"
        # (att 183-185). Put down as gripped it keeps that pose (a 5 cm drop at the target made it worse), so drop it
        # from ~8 cm at a free mid-table spot (well away from the edges), and bring it to the target only once flat.
        tb = bee if (bee is not None and float(bee["height"]) > 0.052) else (tilted_bee(TILTED[0]) if bee is None and TILTED else None)
        if s.dry or tb is None or (bee is not None and lay.get("bad", {}).get("bee", 0) * 2 <= lay["n"]):
            break
        s.phase(f"flip{t}")
        m, spot = temp_spot([bowl["xy"] if bowl is not None else targets["bowl"], targets["bowl"]])
        s.decide(f"bee height {tb['height']:.3f}: on its side or standing; drop it from 8 cm at {r3(spot)} (try {t + 1})")
        tb = dict(tb, xy_blob=tb.get("xy_blob", tb["xy"]))
        try:
            move_bee_from_table(tb, spot, table_z, None if bowl is None else bowl["xy"], drop=0.08, force=True)
        except BeeBlocked as e:
            s.decide(f"flip skipped: {e}"); break
        s.home(f"home_flip{t}")
        bowl, bee = perceive_retry(f"after_flip{t}", need_bee=False)
        if bee is not None and float(bee["height"]) <= 0.052:
            s.decide(f"bee flat after the drop (height {bee['height']:.3f}): to its target")
            try:
                move_bee_from_table(bee, targets["bee"], table_z, None if bowl is None else bowl["xy"])
            except BeeBlocked as e:
                s.decide(f"bee move blocked: {e}")
        bowl, bee, meas, err, lay = measure(targets, f"final_flip{t}")
        s.summary(measured=meas, errors=err, layout_votes=lay)
    redo = [k for k in ("bowl", "bee") if err[k] is not None and (REDO_BOWL if k == "bowl" else REDO) < err[k] < 0.15]
    if "bowl" in redo and lay["n"] and lay.get("bad", {}).get("bowl", 0) * 2 < lay["n"]:
        # the orchestrator's own check passes the bowl in most frames: a correction on ~2 cm blob noise only
        # risks making it worse (att 42: the correction moved it from passing to 4 cm off)
        s.decide(f"bowl error {err['bowl']} but the layout check passes it in {lay['n'] - lay['bad'].get('bowl', 0)}/{lay['n']}: no correction")
        redo.remove("bowl")
    for k, nbad in lay.get("bad", {}).items():   # the orchestrator's check fails on this object: put it down again
        if k == "bee" and k not in redo and nbad * 2 > lay["n"] and err.get(k) is not None and err[k] < 0.15:
            redo.append(k)
    merged = merged_layout(lay, err)
    if merged and "bee" not in redo:
        redo.append("bee")
    # att 336: the bee lying along y at (0.43, -0.17) read 0.155-0.165 long; the orchestrator's prior stops at 0.15 and
    # its check found no bee while our votes passed. Turn it 90 deg when it reads near that limit.
    wide = BEE_EXT[0] >= BEE_EXT_MAX and err.get("bee") is not None and err["bee"] < 0.15 and not merged
    if wide:
        s.decide(f"bee blob reads {BEE_EXT[0]:.3f} long (orchestrator prior <= 0.15): put it down turned 90 deg")
        if "bee" not in redo:
            redo.append("bee")
    if redo and not s.dry:
        s.phase("correct")
        s.decide(f"correction pass for {redo}: errors {err}")
        if "bowl" in redo:
            move_bowl(bowl, list(bowl["xy"]), bowl_aim(targets["bowl"]), table_z, None if bee is None else bee["xy"],
                      disp_ref=meas["bowl"], gain=0.6, cap=0.025)
            if "bee" in redo:
                s.home("home_mid_correct"); bowl, bee = perceive("mid_correct")
        if "bee" in redo and bee is not None:
            # aim off by the measured error (the blob centre the orchestrator uses sits 2-3 cm from where the
            # bee was put, att 88), but the offset is not always systematic (att 128: first put +3.8 cm in x, the
            # re-put at the full counter-aim landed 0.4 cm from the aim -> 3.5 cm off): aim off by half, at most 2 cm
            corr = np.clip(0.5 * np.subtract(targets["bee"], meas["bee"]), -0.02, 0.02)
            dest = list(np.add(bee_aim(targets), corr))     # keep the 1.5 cm nudge away from a close bowl
            if merged:
                # bowl and bee read as one side blob, the orchestrator finds neither (att 229/230, targets 18 cm
                # apart): put the bee 2.5 cm further from the measured bowl, within the 4 cm tolerance
                v = np.subtract(targets["bee"], meas["bowl"]); v = v / max(np.linalg.norm(v), 1e-6)
                dest = r3(np.asarray(targets["bee"]) + MERGE_PUSH * v, 4)
            s.decide(f"bee correction: aim {r3(dest)} (measured {meas['bee']} for target {targets['bee']}"
                     + (", merged with the bowl in every frame" if merged else "") + ")")
            try:
                # near the bowl the long axis goes across the bowl direction (near_bowl_yaw): along y it read 0.16
                # long (att 336), turned 90 deg it pointed at the bowl and merged with it (att 337)
                nyaw = near_bowl_yaw(targets)
                move_bee_from_table(bee, dest, table_z, None if bowl is None else bowl["xy"], force=merged or wide,
                                    place_yaw=nyaw if nyaw is not None else ("rotate" if wide else None))
            except BeeBlocked as e:
                s.decide(f"bee correction skipped: {e}")
        s.phase("verify2")
        bowl, bee, meas, err, lay = measure(targets, "final2")
        s.summary(measured=meas, errors=err, layout_votes=lay)
    if stuck_frac("final", full=True) >= STUCK_FRAC:      # arm at home here
        s.decide("a bee hangs on the gripper at the end: put it down at its target and measure again")
        place_bee(targets["bee"], PLACE_YAW, table_z + 0.09, "stuck_bee", check_empty=False)
        bowl, bee, meas, err, lay = measure(targets, "final3")
        s.summary(measured=meas, errors=err, layout_votes=lay)
    color, _, _ = s.side.latest()
    if color is not None:
        cv2.imwrite(os.path.join(s.A, "side_after.jpg"), cv2.resize(color, (640, 360)))
    bad = lay.get("bad", {})
    flat = bee is not None and float(bee.get("height", 0.0)) <= 0.053
    if merged_layout(lay, err):
        flat = False           # still one merged blob in every frame: the orchestrator cannot find either object
    # the votes catch a bee the orchestrator's prior rejects (on its side, att 183); a flat bee within tolerance passes
    # even when merged bowl+bee blobs fail the votes (att 188: errors 2.0/1.1 cm, votes 1/3, orchestrator ok)
    ok = (all(err[k] is not None and err[k] <= TOL for k in ("bowl", "bee")) and (flat or bad.get("bee", 0) * 2 <= lay["n"])) \
        or (lay["n"] > 0 and lay["ok"] * 2 > lay["n"])     # the orchestrator's own check passes in most frames
    # (the bowl's side blob can read > 0.18 m wide near x 0.38 from depth smear; re-placing it does not change that)
    s.summary(verify=dict(tolerance=TOL, errors=err, measured=meas, targets=targets))
    s.decide(f"final: measured {meas}, errors {err} -> {'success' if ok else 'fail'}")
    if not ok:
        s.summary(first_failed_check=f"objects off target: {err}")
    s.claim("success" if ok else "fail")


if __name__ == "__main__":
    main()
