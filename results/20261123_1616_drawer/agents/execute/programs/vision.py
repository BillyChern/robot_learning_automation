"""Wrist look-down perception for drawer_lying (numpy/cv2 only): rectify the wrist image onto a horizontal plane, find
the drawer's top face and its pull tab, and the target compartment's inner rim."""
import json
import os

import cv2
import numpy as np

from rlpj.robot import calib

RES = 0.0005                       # m per rectified pixel
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wrist_cam():
    """(K, dist, T_tool_cam): the robot machine's calibration, or the workspace copy (offline)."""
    try:
        K, dist = calib.wrist_intrinsics()
        return K, dist, calib.T_tool_cam()
    except OSError:
        d = os.path.join(HERE, "inputs", "calibration")
        f = json.load(open(os.path.join(d, "intrinsics_kinova_1280x720.json")))["factory"]
        K = np.array([[f["fx"], 0, f["cx"]], [0, f["fy"], f["cy"]], [0, 0, 1.0]])
        return K, np.array(f["coeffs"], float), np.array(json.load(open(os.path.join(d, "handeye_kinova.json")))["T_tool_cam"])


def axes(yaw_deg):
    y = np.radians(yaw_deg)
    return np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])


def rectify(img, T_base_tool, c, yaw_deg, half, z, origin=(0, 0), res=RES):
    """Top-down view of the plane at height z (base frame): column = s along the yaw axis, row = t along its normal,
    both increasing, centred on c. Returns (image, valid mask)."""
    K, dist, Ttc = wrist_cam()
    a, b = axes(yaw_deg)
    ss = np.arange(-half[0], half[0] + res / 2, res)
    tt = np.arange(-half[1], half[1] + res / 2, res)
    S, Tt = np.meshgrid(ss, tt)
    P = np.c_[(c[0] + S * a[0] + Tt * b[0]).ravel(), (c[1] + S * a[1] + Tt * b[1]).ravel(), np.full(S.size, z)]
    Pc = calib.apply(np.linalg.inv(np.asarray(T_base_tool, float) @ Ttc), P)
    uv = calib.project(Pc, K, dist).reshape(S.shape + (2,)).astype(np.float32)
    mx, my = uv[..., 0] - origin[0], uv[..., 1] - origin[1]
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    H, W = img.shape[:2]
    return out, (mx >= 1) & (mx < W - 2) & (my >= 1) & (my < H - 2)


def wood_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 25) | (h >= 170)) & (s > 75) & (v > 25)


def find_drawer(img, T_base_tool, c0, z, origin=(0, 0), half=0.105):
    """Top face of the drawer around c0: {c, yaw (long axis, deg in (-90, 90]), size [short, long], ok}."""
    R, valid = rectify(img, T_base_tool, c0, 0.0, (half, half), z, origin)
    m = (wood_mask(R) & valid).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return dict(ok=False, why="no wood")
    mid = np.array(m.shape[::-1]) / 2
    cand = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] * RES ** 2 > 0.003]
    if not cand:
        return dict(ok=False, why="no drawer-sized wood")
    k = min(cand, key=lambda i: np.linalg.norm(cen[i] - mid))
    cnt, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (u, v), (w, h), ang = cv2.minAreaRect(max(cnt, key=len))
    if w > h:                                    # first side = short axis
        w, h, ang = h, w, ang + 90
    yaw = ((ang + 90) + 90) % 180 - 90           # long axis direction in the base frame
    c = np.asarray(c0, float) + (np.array([u, v]) - (np.array(m.shape[::-1]) - 1) / 2) * RES
    size = [round(w * RES, 4), round(h * RES, 4)]
    touches = bool((lab == k)[[0, -1], :].any() or (lab == k)[:, [0, -1]].any() or (~valid & (lab == k)).any())
    ok = 0.060 <= size[0] <= 0.095 and 0.095 <= size[1] <= 0.135 and not touches
    return dict(ok=ok, c=np.round(c, 4).tolist(), yaw=round(float(yaw), 2), size=size, touches=touches)


def find_tab(img, T_base_tool, d, z, origin=(0, 0)):
    """Pull in drawer coordinates (s across the short axis, t along the long axis, axes from d['yaw'] - 90): the grey
    label frame's columns, the tab = grey pixels beyond them. {ok, xy, s, t, side, frame_s, tab_len_t}."""
    half = (d["size"][0] / 2 - 0.003, d["size"][1] / 2 - 0.003)
    ang = d["yaw"] - 90.0
    R, valid = rectify(img, T_base_tool, d["c"], ang, half, z, origin)
    hsv = cv2.cvtColor(R, cv2.COLOR_BGR2HSV)
    # metal: unsaturated, or dark and only weakly orange; the shadow on the wood stays strongly saturated
    grey = ((hsv[..., 1] < 95) | ((hsv[..., 1] < 135) & (hsv[..., 2] < 120))) & (hsv[..., 2] > 25) & valid
    grey = cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grey, connectivity=8)
    if n < 2:
        return dict(ok=False, why="no metal seen")
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    M = lab == k
    col = M.sum(0).astype(float)
    if col.max() < 0.03 / RES:
        return dict(ok=False, why="metal too small for the label frame")
    bars = np.where(col >= 0.55 * col.max())[0]
    f0, f1 = int(bars.min()), int(bars.max())
    W = M.shape[1]
    left, right = M[:, :max(f0 - 3, 0)], M[:, f1 + 4:]
    nl, nr = int(left.sum()), int(right.sum())
    side = 1 if nr >= nl else -1
    P = right if side > 0 else left
    if P.sum() < (0.004 / RES) ** 2:
        return dict(ok=False, why="no tab beside the frame", n=[nl, nr])
    rows, cols = np.where(P)
    cols = cols + (f1 + 4 if side > 0 else 0)
    s_px, t_px = float(np.median(cols)), float(np.median(rows))
    s = (s_px - (W - 1) / 2) * RES
    t = (t_px - (M.shape[0] - 1) / 2) * RES
    a, b = axes(ang)
    xy = np.asarray(d["c"], float) + s * a + t * b
    t_span = float(np.percentile(rows, 97) - np.percentile(rows, 3)) * RES
    s_len = float(cols.max() - cols.min()) * RES
    fw = (f1 - f0) * RES
    ok = 0.008 <= t_span <= 0.035 and 0.004 <= s_len <= 0.03 and 0.02 <= fw <= 0.05
    return dict(ok=ok, xy=np.round(xy, 4).tolist(), s=round(s, 4), t=round(t, 4), side=side, tab_t_span=round(t_span, 4),
                tab_s_len=round(s_len, 4), frame_s=[round((f0 - (W - 1) / 2) * RES, 4), round((f1 - (W - 1) / 2) * RES, 4)])


def _line(pts):
    """Robust fit y = off + slope * x of (x, y) samples -> dict(off, slope, n, resid) or None."""
    if len(pts) < 8:
        return None
    A = np.asarray(pts, float)
    keep = np.abs(A[:, 1] - np.median(A[:, 1])) < 0.006
    if keep.sum() < 6:
        return None
    for _ in range(4):
        fit = np.polyfit(A[keep, 0], A[keep, 1], 1)
        res = np.abs(np.polyval(fit, A[:, 0]) - A[:, 1])
        keep = res < max(2.5 * np.median(res[keep]), 0.0008)
        if keep.sum() < 6:
            return None
    return dict(slope=float(fit[0]), off=float(fit[1]), n=int(keep.sum()), resid=float(np.median(res[keep])))


def _outer(wood, valid, reverse=False, run=8):
    """First table -> wood step along a 1-D scan (index in the original order), or None. The pixel before the step must
    be valid table, so the border of the image's footprint is never taken for the cabinet's edge."""
    idx = np.arange(len(wood))[::-1] if reverse else np.arange(len(wood))
    w, v = wood[idx], valid[idx]
    for k in range(3, len(w) - run):
        if w[k:k + run].all() and not w[k - 3:k].any() and v[k - 3:k].all():
            return int(idx[k])
    return None


def find_opening(img, T_base_tool, c0, yaw_deg, size, z, origin=(0, 0), wall=0.012):
    """Target compartment from the rim plane, around the prior centre c0 (axes a along the row of compartments, b across).
    The cabinet's outer edges against the table (near end a-, long sides b-/b+) fix b and the yaw; the rim -> opening
    step inside the near end wall fixes a. {ok, c, yaw, da, db, dyaw, inner_a, outer, width_b, resid_mm}."""
    ha, hb = size[0] / 2, size[1] / 2
    half = (ha + 0.040, hb + 0.036)
    R, valid = rectify(img, T_base_tool, c0, yaw_deg, half, z, origin)
    wood = wood_mask(R)
    V = cv2.GaussianBlur(cv2.cvtColor(R, cv2.COLOR_BGR2HSV)[..., 2], (5, 5), 0).astype(np.float32)
    Hh, Ww = V.shape
    cs, ct = (Ww - 1) / 2, (Hh - 1) / 2
    pa, pb, pi = [], [], []
    for x in np.arange(-0.75 * hb, 0.75 * hb, 0.002):            # rows: near end wall
        r = int(round(ct + x / RES))
        k = _outer(wood[r], valid[r])
        if k is None or k > cs:
            continue
        pa.append((x, (k - cs) * RES))
        lo, hi = k + int(0.005 / RES), min(k + int(0.020 / RES), Ww - 2)
        prof = V[max(r - 3, 0):r + 4].mean(0)
        g = np.gradient(prof)[lo:hi]
        if len(g) and g.min() < -1.5:
            pi.append((x, (lo + int(np.argmin(g)) - cs) * RES))
    pbm, pbp = [], []
    for x in np.arange(-0.75 * ha, 0.75 * ha, 0.002):            # columns: the two long sides
        c = int(round(cs + x / RES))
        k = _outer(wood[:, c], valid[:, c])
        if k is not None and k < ct:
            pbm.append((x, (k - ct) * RES))
        k = _outer(wood[:, c], valid[:, c], reverse=True)
        if k is not None and k > ct:
            pbp.append((x, (k - ct) * RES))
    L = dict(a_out=_line(pa), a_in=_line(pi), b_m=_line(pbm), b_p=_line(pbp))
    if any(v is None for v in L.values()):
        return dict(ok=False, why="edge not found", n=dict(a_out=len(pa), a_in=len(pi), b_m=len(pbm), b_p=len(pbp)),
                    lines={k: None if v is None else round(v["off"], 4) for k, v in L.items()})
    da = L["a_in"]["off"] + ha
    db = 0.5 * (L["b_m"]["off"] + L["b_p"]["off"])
    width_b = L["b_p"]["off"] - L["b_m"]["off"]
    wall_a = L["a_in"]["off"] - L["a_out"]["off"]
    # an a-edge s = off + slope * t is turned by -atan(slope); a b-edge t = off + slope * s by +atan(slope)
    angs = [-np.degrees(np.arctan(L["a_out"]["slope"])), np.degrees(np.arctan(L["b_m"]["slope"])), np.degrees(np.arctan(L["b_p"]["slope"]))]
    dyaw = float(np.median(angs))
    a, b = axes(yaw_deg)
    c = np.asarray(c0, float) + da * a + db * b
    resid = max(v["resid"] for v in L.values()) * 1000
    ok = abs(width_b - (size[1] + 2 * wall)) < 0.010 and 0.006 <= wall_a <= 0.017 and float(np.ptp(angs)) < 3.0 and resid < 1.5
    return dict(ok=ok, c=np.round(c, 4).tolist(), yaw=round(yaw_deg + dyaw, 2), da=round(da, 4), db=round(db, 4), dyaw=round(dyaw, 2),
                wall_a=round(wall_a, 4), width_b=round(width_b, 4), resid_mm=round(resid, 2), angs=np.round(angs, 2).tolist())


def plane_tilt(P, c, yaw_deg, half, z_mid, z_band, exclude=None):
    """Robust plane z = z0 + gs * s + gt * t through the base-frame points P inside a rectangle around c (axes from
    yaw_deg) and z_mid +- z_band; exclude = (s, t, half_s, half_t) drops the fingers or the pull. Tilts in degrees."""
    a, b = axes(yaw_deg)
    d = np.asarray(P, float)[:, :2] - np.asarray(c, float)
    s, t, z = d @ a, d @ b, np.asarray(P, float)[:, 2]
    m = (np.abs(s) < half[0]) & (np.abs(t) < half[1]) & (np.abs(z - z_mid) < z_band)
    if exclude is not None:
        m &= ~((np.abs(s - exclude[0]) < exclude[2]) & (np.abs(t - exclude[1]) < exclude[3]))
    if m.sum() < 150:
        return dict(ok=False, n=int(m.sum()))
    s, t, z = s[m], t[m], z[m]
    A = np.c_[np.ones(len(z)), s, t]
    keep = np.ones(len(z), bool)
    for _ in range(5):
        x, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = np.abs(A @ x - z)
        keep = res < max(2.5 * np.median(res[keep]), 0.0015)
        if keep.sum() < 100:
            return dict(ok=False, n=int(keep.sum()))
    span = [float(np.ptp(s[keep])), float(np.ptp(t[keep]))]
    return dict(ok=bool(span[0] > 0.035 and span[1] > 0.05), n=int(keep.sum()), tilt_s_deg=round(float(np.degrees(np.arctan(x[1]))), 2),
                tilt_t_deg=round(float(np.degrees(np.arctan(x[2]))), 2), z0=round(float(x[0]), 4),
                resid_mm=round(float(np.median(res[keep])) * 1000, 2), span=np.round(span, 3).tolist())


def held_offset(img, T_base_tool, c_plan, yaw_deg, z, half_short=0.0368, origin=(0, 0)):
    """Where the held drawer's top face really is, from the wrist image at the hover, relative to its planned centre
    c_plan (axes a, b of the opening; the plane z is the top face). The long sides against the table beyond them give b
    and the yaw; the short edge on the camera's side gives a (the other one is out of view). {ok, da, db, dyaw, ...}."""
    R, valid = rectify(img, T_base_tool, c_plan, yaw_deg, (0.060, 0.075), z, origin)
    cam = (np.asarray(T_base_tool, float) @ wrist_cam()[2])[:2, 3] - np.asarray(c_plan, float)
    sg = 1.0 if float(cam @ axes(yaw_deg)[0]) > 0 else -1.0
    wood = wood_mask(R)
    V = cv2.GaussianBlur(cv2.cvtColor(R, cv2.COLOR_BGR2HSV)[..., 2], (5, 5), 0).astype(np.float32)
    Hh, Ww = V.shape
    cs, ct = (Ww - 1) / 2, (Hh - 1) / 2
    pm, pp, pa = [], [], []
    for x in sg * np.arange(0.006, 0.033, 0.001):                # columns on the camera side of the fingers
        c = int(round(cs + x / RES))
        k = _outer(wood[:, c], valid[:, c])
        if k is not None and k < ct:
            pm.append((x, (k - ct) * RES))
        k = _outer(wood[:, c], valid[:, c], reverse=True)
        if k is not None and k > ct:
            pp.append((x, (k - ct) * RES))
    lo, hi = int(round(cs + (sg * half_short - 0.010) / RES)), int(round(cs + (sg * half_short + 0.010) / RES))
    for x in np.arange(-0.045, 0.045, 0.002):                    # rows: the camera-side short edge of the top face
        r = int(round(ct + x / RES))
        if lo < 2 or not valid[r, lo:hi].all():
            continue
        g = np.abs(np.gradient(V[max(r - 3, 0):r + 4].mean(0)))[lo:hi]
        if g.max() > 1.0:
            pa.append((x, (lo + int(np.argmax(g)) - cs) * RES))
    L = dict(b_m=_line(pm), b_p=_line(pp), a=_line(pa))
    if any(v is None for v in L.values()):
        return dict(ok=False, why="edge not found", n=[len(pm), len(pp), len(pa)])
    db = 0.5 * (L["b_m"]["off"] + L["b_p"]["off"])
    width = L["b_p"]["off"] - L["b_m"]["off"]
    angs = [np.degrees(np.arctan(L["b_m"]["slope"])), np.degrees(np.arctan(L["b_p"]["slope"])), -np.degrees(np.arctan(L["a"]["slope"]))]
    dyaw = float(np.median(angs))
    da = L["a"]["off"] - sg * half_short
    resid = max(v["resid"] for v in L.values()) * 1000
    ok = 0.105 <= width <= 0.125 and abs(da) < 0.012 and abs(db) < 0.012 and float(np.ptp(angs)) < 5.0 and resid < 1.5
    # the long sides converge in the rectified plane when the face is tilted about b: width(s) ~ w (1 + g s / depth), g = dz/ds
    depth = float((np.asarray(T_base_tool, float) @ wrist_cam()[2])[2, 3] - z)
    tilt_s = float(np.degrees(np.arctan((L["b_p"]["slope"] - L["b_m"]["slope"]) * depth / width)))
    return dict(ok=bool(ok), tilt_s=round(tilt_s, 2), edge=int(sg), da=round(float(da), 4), db=round(float(db), 4), dyaw=round(dyaw, 2), width_b=round(float(width), 4),
                resid_mm=round(resid, 2), angs=np.round(angs, 2).tolist())


def cabinet_near(img, T_base_tool, c, yaw_deg, z, origin=(0, 0)):
    """The cabinet's near end in the rim plane, seen past the held drawer (axes of the opening, around c): outer edge of the
    near wall (a, yaw) and the two long sides at that wall (b). {ok, a_out, cb, width_b, dyaw, ...}, offsets from c."""
    R, valid = rectify(img, T_base_tool, c, yaw_deg, (0.075, 0.085), z, origin)
    wood = wood_mask(R)
    Hh, Ww = wood.shape
    cs, ct = (Ww - 1) / 2, (Hh - 1) / 2
    pa = []
    for x in np.arange(-0.055, 0.055, 0.002):
        r = int(round(ct + x / RES))
        k = _outer(wood[r], valid[r])
        if k is not None and k < cs:
            pa.append((x, (k - cs) * RES))
    La = _line(pa)                                # logged only: this edge includes the wall's outer face (moves ~8 mm)
    pm, pp = [], []
    for x in np.arange(-0.0445, -0.0394, 0.0005):  # columns on the near wall's top
        c_ = int(round(cs + x / RES))
        k = _outer(wood[:, c_], valid[:, c_])
        if k is not None and k < ct:
            pm.append((k - ct) * RES)
        k = _outer(wood[:, c_], valid[:, c_], reverse=True)
        if k is not None and k > ct:
            pp.append((k - ct) * RES)
    if len(pm) < 6 or len(pp) < 6:
        return dict(ok=False, why="long sides not found", n=[len(pm), len(pp)])
    bm, bp = float(np.median(pm)), float(np.median(pp))
    spread = max(float(np.ptp(np.clip(pm, bm - 0.003, bm + 0.003))), float(np.ptp(np.clip(pp, bp - 0.003, bp + 0.003))))
    ok = 0.125 <= bp - bm <= 0.145 and spread < 0.004
    return dict(ok=bool(ok), a_out=None if La is None else round(La["off"], 4), cb=round(0.5 * (bm + bp), 4), width_b=round(bp - bm, 4))


def drawer_vs_opening(img, T_base_tool, c, yaw_deg, z_rim, z_top, origin=(0, 0)):
    """Drawer relative to the opening from one wrist image, camera on the -a side: the drawer's top-face edges against the
    cabinet's near end, each in its own plane. b comes from the two in the same image; a and the yaw are the drawer against
    the rim measured before the grasp (the near wall's outer edge is not usable for them). {ok, ea, eb, eyaw, drawer, cabinet}."""
    d = held_offset(img, T_base_tool, c, yaw_deg, z_top, origin=origin)
    k = cabinet_near(img, T_base_tool, c, yaw_deg, z_rim, origin)
    out = dict(ok=False, drawer=d, cabinet=k)
    if d.get("edge") != -1 or d.get("da") is None or not k.get("ok") or not d.get("ok"):
        return out
    out.update(ok=True, ea=d["da"], eb=round(d["db"] - k["cb"], 4), eyaw=d["dyaw"])
    return out


def cabinet_shift_px(before, after, regions=((320, 160, 405, 240), (285, 205, 400, 290))):
    """Largest image shift (px at 640 wide) of the cabinet's parts away from the target compartment between two side images, by
    phase correlation; None when no region correlates."""
    a, b = (cv2.cvtColor(cv2.resize(i, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for i in (before, after))
    out = []
    for x0, y0, x1, y1 in regions:
        g0, g1 = np.float32(a[y0:y1, x0:x1]), np.float32(b[y0:y1, x0:x1])
        (dx, dy), r = cv2.phaseCorrelate(g0, g1, cv2.createHanningWindow(g0.shape[::-1], cv2.CV_32F))
        if r > 0.2:
            out.append(float(np.hypot(dx, dy)))
    return max(out) if out else None


def outline_xy(blob, yaw_deg):
    """Centre of a blob's outline: 2nd-98th percentile midpoints along and across the axis yaw_deg (operator 18:56). Same definition as
    rlpj.success.checks.outline_xy, used only where the installed checks module does not have it."""
    a, b = axes(yaw_deg)
    xy = np.asarray(blob["points"], float)[:, :2]
    s0, s1 = np.percentile(xy @ a, [2, 98])
    t0, t1 = np.percentile(xy @ b, [2, 98])
    return 0.5 * (s0 + s1) * a + 0.5 * (t0 + t1) * b
