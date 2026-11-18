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
