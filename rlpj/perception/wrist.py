"""Wrist camera helpers: project base-frame points into the wrist image, crop around them, refine a circle."""
import cv2
import numpy as np

from rlpj.robot import calib


def project_base_to_wrist(P_base, T_base_tool, K=None, dist=None, T_tool_cam=None):
    """Base-frame points (N,3) -> wrist pixel coordinates (N,2) for the tool pose T_base_tool; nan behind the camera."""
    K, dist, T_tc = _cam(K, dist, T_tool_cam)
    T_cam_base = np.linalg.inv(np.asarray(T_base_tool, float) @ T_tc)
    Pc = calib.apply(T_cam_base, np.asarray(P_base, float).reshape(-1, 3))
    uv = calib.project(Pc, K, dist)
    uv[Pc[:, 2] <= 0.01] = np.nan
    return uv


def roi(img, uv, half):
    """Square crop of img centred on pixel uv with half-size half (clipped to the image)."""
    x0, y0, x1, y1 = _box(img, uv, half)
    return img[y0:y1, x0:x1]


def circle_refine(img, T_base_tool, xy, z, r_m, K=None, dist=None, T_tool_cam=None, min_coverage=0.5):
    """Refine the base-frame centre xy of a circle of radius r_m (m) lying at height z (a rim or a ring's edge):
    project it into the wrist image, find edge points around it, RANSAC a circle within 0.8..1.2 of the expected
    pixel radius, cast the centre back onto the plane z. Returns (xy or None, info)."""
    K, dist, T_tc = _cam(K, dist, T_tool_cam)
    T_base_cam = np.asarray(T_base_tool, float) @ T_tc
    pc = calib.apply(np.linalg.inv(T_base_cam), np.array([[xy[0], xy[1], z]]))[0]
    info = dict(err=None)
    if pc[2] < 0.05:
        info["err"] = "behind camera"
        return None, info
    u0 = calib.project(pc[None], K, dist)[0]
    rpx = float(K[0, 0] * r_m / pc[2])
    box = _box(img, u0, int(1.9 * rpx))
    info.update(u0=u0.round(1).tolist(), r_px=round(rpx, 1), box=list(box))
    if box[2] - box[0] < rpx or box[3] - box[1] < rpx:
        info["err"] = "roi outside image"
        return None, info
    pts, _ = edge_points(img, box)
    res, _ = ransac_circle(pts, u0, 0.8 * rpx, 1.2 * rpx, 0.6 * rpx)
    if res is None:
        info["err"] = "no circle"
        return None, info
    (u, v), r, cov, n = res
    info.update(circle=[round(float(u), 1), round(float(v), 1), round(r, 1)], coverage=round(cov, 2), inliers=n)
    if cov < min_coverage:
        info["err"] = "weak circle"
        return None, info
    ray = cv2.undistortPoints(np.array([[[u, v]]], float), K, dist).reshape(2)
    rb = T_base_cam[:3, :3] @ np.array([ray[0], ray[1], 1.0])
    ob = T_base_cam[:3, 3]
    if abs(rb[2]) < 1e-6:
        info["err"] = "ray parallel to the table"
        return None, info
    p = ob + (z - ob[2]) / rb[2] * rb
    info["xy"] = p[:2].round(4).tolist()
    return p[:2], info


def edge_points(img, box):
    """Thinned strong colour edges inside box (x0, y0, x1, y1) -> (pixel points (N,2), gradient magnitudes)."""
    x0, y0, x1, y1 = box
    sub = cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 1.2).astype(np.float32)
    sub = sub[:, :, None] if sub.ndim == 2 else sub
    mag = np.zeros(sub.shape[:2], np.float32)
    for c in range(sub.shape[2]):
        gx = cv2.Sobel(sub[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        mag = np.maximum(mag, np.hypot(gx, gy))
    m = (mag > max(np.percentile(mag, 92), 25.0)) & (mag >= cv2.dilate(mag, np.ones((3, 3), np.uint8)))
    ys, xs = np.nonzero(m)
    return np.stack([xs + x0, ys + y0], 1).astype(np.float64), mag[ys, xs]


def ransac_circle(pts, c0, r_lo, r_hi, max_center_off, iters=2000, tol=2.5, seed=0):
    """Circle through edge points with radius in [r_lo, r_hi] and centre within max_center_off px of c0, scored by
    inliers x angular coverage; returns ((centre, r, coverage, n_inliers) or None, score)."""
    rng = np.random.default_rng(seed)
    best, bscore = None, 0.0
    if len(pts) < 10:
        return None, 0.0
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        try:
            c = np.linalg.solve(np.c_[2 * s, np.ones(3)], (s ** 2).sum(1))
        except np.linalg.LinAlgError:
            continue
        cen, r2 = c[:2], c[2] + c[:2] @ c[:2]
        if r2 <= 0:
            continue
        r = np.sqrt(r2)
        if not (r_lo <= r <= r_hi) or np.linalg.norm(cen - c0) > max_center_off:
            continue
        inl = np.abs(np.linalg.norm(pts - cen, axis=1) - r) < tol
        score = inl.sum() * _coverage(pts[inl], cen)
        if score > bscore:
            bscore, best = score, inl
    if best is None:
        return None, 0.0
    P = pts[best]
    c = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
    cen = c[:2]
    r = float(np.sqrt(max(c[2] + cen @ cen, 1e-9)))
    return (cen, r, _coverage(P, cen), int(best.sum())), float(bscore)


def _coverage(P, cen, nbins=36):
    """Fraction of angular bins around cen that contain a point."""
    if len(P) == 0:
        return 0.0
    ang = np.arctan2(P[:, 1] - cen[1], P[:, 0] - cen[0])
    return len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * nbins))) / nbins


def _cam(K, dist, T_tool_cam):
    if K is None:
        K, dist = calib.wrist_intrinsics()
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    T_tool_cam = calib.T_tool_cam() if T_tool_cam is None else T_tool_cam
    return np.asarray(K, float), dist, np.asarray(T_tool_cam, float)


def _box(img, uv, half):
    h, w = img.shape[:2]
    x0, y0 = int(max(uv[0] - half, 0)), int(max(uv[1] - half, 0))
    x1, y1 = int(min(uv[0] + half, w)), int(min(uv[1] + half, h))
    return x0, y0, x1, y1


def gripper_load(img, chk):
    """Share of object-coloured pixels in the bottom of the wrist image, where an object hanging on the fingers shows
    (chk: task gripper_check with hsv [[lo], [hi]] and bottom share); None without an image."""
    if img is None:
        return None
    h = img.shape[0]
    lo, hi = (np.array(v, np.uint8) for v in chk["hsv"])
    m = cv2.inRange(cv2.cvtColor(img[int((1.0 - chk.get("bottom", 0.15)) * h):], cv2.COLOR_BGR2HSV), lo, hi)
    return float((m > 0).mean())
