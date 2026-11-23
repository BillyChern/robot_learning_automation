"""Find the drawer's metal pull in a look-down wrist image: the knob is the thickest piece of metal, the label frame is
the thin ring beside it. Needs only the image scale at the drawer's top face."""
import cv2
import numpy as np


def metal_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    return ((S >= 22) & (S < 165) & (V > 70)).astype(np.uint8)   # table S ~15, wood S > 200, dark gaps V < 65


def detect(img, px_per_m, centre_uv=None, radius_m=0.06):
    """dict(knob_uv, frame_uv, axis, r_m, ...) or None. axis: unit image vector square to the frame's long side, toward
    the knob. Only metal within radius_m of centre_uv counts."""
    H, W = img.shape[:2]
    m = metal_mask(img)
    if centre_uv is not None:
        roi = np.zeros((H, W), np.uint8)
        cv2.circle(roi, (int(centre_uv[0]), int(centre_uv[1])), int(radius_m * px_per_m), 1, -1)
        m = m & roi
    # only metal on the wood counts: dark table edges and shadows outside the drawer also have a low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wood = ((hsv[..., 1] >= 165) & (hsv[..., 2] >= 45)).astype(np.uint8)
    if centre_uv is not None:
        wood = wood & roi
    nw, lw, sw, _ = cv2.connectedComponentsWithStats(wood, connectivity=8)
    if nw < 2:
        return None
    big = np.column_stack(np.nonzero(lw == 1 + int(np.argmax(sw[1:, cv2.CC_STAT_AREA]))))[:, ::-1]
    hull = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(big.astype(np.int32)), 1)
    m = m & hull
    kc = max(3, int(0.0015 * px_per_m) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    m[0, :] = m[-1, :] = 0; m[:, 0] = m[:, -1] = 0       # the distance transform must see the image edge as background
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    r = float(dt.max())
    if not 0.0045 <= r / px_per_m <= 0.016:              # knob half-width ~8 mm, frame bars ~3 mm
        return None
    core = (dt > 0.55 * r).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(core, connectivity=8)
    i = int(lab[np.unravel_index(int(np.argmax(dt)), dt.shape)])
    ku = np.array(cen[i], float)
    blob = cv2.dilate((lab == i).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 5, 2 * int(r) + 5)))
    rest = ((m > 0) & (blob == 0)).astype(np.uint8)
    nr, lr, sr, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    if nr < 2:
        return None
    j = 1 + int(np.argmax(sr[1:, cv2.CC_STAT_AREA]))
    if sr[j, cv2.CC_STAT_AREA] < (0.012 * px_per_m) ** 2:
        return None
    pts = np.column_stack(np.nonzero(lr == j))[:, ::-1].astype(np.float32)
    (fx, fy), (fw, fh), fa = cv2.minAreaRect(pts)
    fu = np.array([fx, fy], float)
    d = ku - fu
    dist = float(np.linalg.norm(d))
    if not (0.008 < dist / px_per_m < 0.05) or not (0.025 < max(fw, fh) / px_per_m < 0.075):
        return None
    a = np.radians(fa if fw >= fh else fa + 90.0)        # direction of the frame's long side
    e = np.array([-np.sin(a), np.cos(a)])
    e = e if e @ d >= 0 else -e
    return dict(knob_uv=ku, frame_uv=fu, axis=e, r_m=round(r / px_per_m, 4), area_frame=int(sr[j, cv2.CC_STAT_AREA]),
                frame_size_m=[round(max(fw, fh) / px_per_m, 4), round(min(fw, fh) / px_per_m, 4)], dist_m=round(dist / px_per_m, 4))
