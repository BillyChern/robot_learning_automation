"""Which drawer wall carries the pull (grey metal label frame + tab)? Side-camera colour on each wall's outer face:
wood is saturated orange-brown, the frame is grey. Pure helpers (numpy/cv2), used by execute.py and orient_probe.py."""
import cv2
import numpy as np

from rlpj.robot import calib


def side_cam():
    K, dist = calib.side_intrinsics()
    Tbs = calib.T_base_side()
    return K, dist, Tbs, np.linalg.inv(Tbs)


def faces_of(c, d_in, L, W):
    """Four walls of a drawer with centre c (xy), d_in = unit direction from wall A (the grasped one) into the drawer.
    Returns dict name -> (outer face centre xy, outward normal xy, half width along the face)."""
    d = np.asarray(d_in, float); p = np.array([-d[1], d[0]])
    return dict(A=(c - d * L / 2, -d, W / 2), C=(c + d * L / 2, d, W / 2), B=(c + p * W / 2, p, L / 2), D=(c - p * W / 2, -p, L / 2))


def score_faces(img, faces, z_bottom, h_band=(0.015, 0.06), margin=0.6):
    """Per wall: visibility (cos between its outward normal and the direction to the camera), and colour stats of the
    image samples on its outer face (inner `margin` of its width, heights h_band above z_bottom)."""
    K, dist, Tbs, Tsb = side_cam()
    cam = Tbs[:3, 3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {}
    for k, (fc, nrm, half) in faces.items():
        tang = np.array([-nrm[1], nrm[0]])
        mid = np.r_[fc + 0.002 * nrm, z_bottom + np.mean(h_band)]
        v = cam - mid
        vis = float(np.r_[nrm, 0.0] @ v / np.linalg.norm(v))
        P = np.array([np.r_[fc + 0.002 * nrm + a * tang, z_bottom + hh]
                      for a in np.linspace(-margin * half, margin * half, 25) for hh in np.linspace(h_band[0], h_band[1], 10)])
        uv = calib.project(calib.apply(Tsb, P), K, dist)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0] - 1)
        if ok.sum() < 20 or vis < 0.15:
            out[k] = dict(vis=round(vis, 2), n=int(ok.sum()))
            continue
        px = hsv[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].astype(float)
        S, V = px[:, 1], px[:, 2]
        grey = (S < 70) & (V > 40)
        out[k] = dict(vis=round(vis, 2), n=int(ok.sum()), grey=round(float(grey.mean()), 3), s_med=round(float(np.median(S)), 1),
                      v_med=round(float(np.median(V)), 1), uv=np.round(uv[ok].mean(0)).astype(int).tolist())
    return out


def crop_around(img, P_base, half=(220, 150)):
    """Crop of the full-res side image around the projection of the base point(s) P_base (<= 440 x 300 px)."""
    K, dist, Tbs, Tsb = side_cam()
    uv = calib.project(calib.apply(Tsb, np.atleast_2d(P_base)), K, dist).mean(0)
    u, v = int(uv[0]), int(uv[1])
    x0, y0 = max(0, u - half[0]), max(0, v - half[1])
    return img[y0:min(img.shape[0], v + half[1]), x0:min(img.shape[1], u + half[0])].copy()


def decide(scores_list, min_grey=0.08, ratio=2.0):
    """scores_list: [{face: stats}] from several views. Best grey fraction per face over the views where it is visible;
    pull = argmax when it beats the runner-up by `ratio` and exceeds min_grey. Returns (face or None, best dict)."""
    best = {}
    for sc in scores_list:
        for k, st in sc.items():
            if "grey" in st and st["vis"] >= 0.25:
                best[k] = max(best.get(k, 0.0), st["grey"])
    if not best:
        return None, best
    order = sorted(best.items(), key=lambda kv: -kv[1])
    top = order[0]
    second = order[1][1] if len(order) > 1 else 0.0
    if top[1] >= min_grey and top[1] >= ratio * max(second, 0.01):
        return top[0], best
    return None, best


def wood_hull(img, uv0, rad=260):
    """Filled convex hull of the wood (orange, saturated) component nearest to pixel uv0 (the drawer), or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wood = ((H <= 25) | (H >= 170)) & (S > 90) & (V > 40)
    m = np.zeros(img.shape[:2], np.uint8)
    u0, v0 = int(uv0[0]), int(uv0[1])
    y0, y1, x0, x1 = max(0, v0 - rad), min(img.shape[0], v0 + rad), max(0, u0 - rad), min(img.shape[1], u0 + rad)
    m[y0:y1, x0:x1] = wood[y0:y1, x0:x1].astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 300]
    if not big:
        return None
    # the drawer: the big component nearest the expected centre (not the cabinet beside it)
    i0 = min(big, key=lambda i: np.hypot(*(cent[i] - [u0, v0])))
    pts = np.argwhere(lab == i0)[:, ::-1]
    if len(pts) < 50:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    out = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def score_faces2(img, faces, z_bottom, uv_center, exclude=(), h_band=(0.012, 0.062), margin=0.8):
    """Like score_faces, but only samples inside the drawer's wood hull count (background excluded); grey = low
    saturation and not dark (metal frame / knob, not the black gripper). A face needs >= 25 % of its samples on the
    drawer to be scored."""
    K, dist, Tbs, Tsb = side_cam()
    cam = Tbs[:3, 3]
    hull = wood_hull(img, uv_center)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {}
    for k, (fc, nrm, half) in faces.items():
        if k in exclude:
            continue
        tang = np.array([-nrm[1], nrm[0]])
        mid = np.r_[fc, z_bottom + np.mean(h_band)]
        v = cam - mid
        vis = float(np.r_[nrm, 0.0] @ v / np.linalg.norm(v))
        P = np.array([np.r_[fc + 0.002 * nrm + a * tang, z_bottom + hh]
                      for a in np.linspace(-margin * half, margin * half, 31) for hh in np.linspace(h_band[0], h_band[1], 12)])
        uv = calib.project(calib.apply(Tsb, P), K, dist).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        if vis < 0.15 or ok.sum() < 20 or hull is None:
            out[k] = dict(vis=round(vis, 2)); continue
        uv = uv[ok]
        on = hull[uv[:, 1], uv[:, 0]]
        if on.mean() < 0.25:
            out[k] = dict(vis=round(vis, 2), on=round(float(on.mean()), 2)); continue
        px = hsv[uv[on, 1], uv[on, 0]].astype(float)
        grey = (px[:, 1] < 70) & (px[:, 2] > 70)
        out[k] = dict(vis=round(vis, 2), on=round(float(on.mean()), 2), grey=round(float(grey.mean()), 3),
                      s_med=round(float(np.median(px[:, 1])), 1))
    return out


GREY_S = 100                 # wood S 150-210; the frame in shadow (attempt 14 rot90) reaches ~100


def detect_pull(img, uv_center):
    """The pull (grey metal frame + knob) on the drawer in the side image: the largest grey (S < 70, V > 70)
    component inside the wood hull, in its lower 60 % (the gripper sits at the top). Side: left/right of the hull's
    lowest point (the vertical edge nearest the camera). Returns dict(found, ...)."""
    hull = wood_hull(img, uv_center)
    if hull is None:
        return dict(found=False, why="no wood")
    ys, xs = np.nonzero(hull)
    top, bot = ys.min(), ys.max()
    corner_u = float(np.median(xs[ys >= bot - 3]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = (hsv[..., 1] < GREY_S) & (hsv[..., 2] > 45) & hull
    grey[: int(top + 0.4 * (bot - top))] = False
    m = cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return dict(found=False, corner_u=corner_u, why="no grey")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA]); u, v = cent[i]
    hull_area = int(hull.sum())
    found = area >= max(150, 0.006 * hull_area)
    return dict(found=bool(found), u=round(float(u), 1), v=round(float(v), 1), area=area, hull_area=hull_area,
                corner_u=round(corner_u, 1), side="left" if u < corner_u else "right")


def visible_lr(faces, exclude=()):
    """(left face, right face) of the two walls facing the side camera, by the image u of their centres."""
    K, dist, Tbs, Tsb = side_cam()
    cam = Tbs[:3, 3]
    vis = []
    for k, (fc, nrm, half) in faces.items():
        v = cam[:2] - fc
        c = float(nrm @ v / np.linalg.norm(v))
        u = calib.project(calib.apply(Tsb, np.r_[fc, 0.0][None]), K, dist)[0, 0]
        vis.append((c, u, k))
    vis.sort(reverse=True)
    two = sorted(vis[:2], key=lambda x: x[1])
    return two[0][2], two[1][2]


def uv_of(P_base):
    K, dist, Tbs, Tsb = side_cam()
    return calib.project(calib.apply(Tsb, np.atleast_2d(P_base)), K, dist).mean(0)
