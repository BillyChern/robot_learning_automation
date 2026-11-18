"""Wrist-camera wall refinement (position AND angle) of a drawer wall seen from above, extending the reset agent's
refine_wall: the outer edge (bright table -> darker wood, going inward along d_in) at the wall-top height is found
separately on the two lateral halves of the wall; their difference gives the wall's angle. Pure (numpy/cv2)."""
import cv2
import numpy as np

from rlpj.perception.wrist import project_base_to_wrist


def edge_profile(g, T_tool, e, d_in, z, a_vals, half=0.04, scale=1.0):
    """Position (m along d_in from e) of the strongest bright->dark step and its strength, sampling points
    e + s d_in + a w (a in a_vals) at height z. g: grey image (any size; scale = its width / 1280)."""
    w = np.array([-d_in[1], d_in[0]])
    ss = np.arange(-half, half + 1e-9, 0.001)
    prof = []
    for sv in ss:
        P = np.array([np.r_[e + sv * d_in + a * w, z] for a in a_vals])
        uv = project_base_to_wrist(P, T_tool) * scale
        ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < g.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < g.shape[0] - 1)
        prof.append(np.median(g[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]) if ok.sum() >= 2 else np.nan)
    prof = np.array(prof)
    if np.isnan(prof).mean() > 0.3:
        return None, 0.0
    prof = np.where(np.isnan(prof), np.nanmedian(prof), prof)
    grad = np.correlate(prof, [1, 1, 0, -1, -1], "same"); grad[:3] = grad[-3:] = 0
    k = int(np.argmax(grad))
    return float(ss[k]), float(grad[k])


def wall_fit(img, T_tool, e, d_in, z, lat=0.035, min_step=40.0):
    """dict(shift (m along d_in, edge at +6 mm convention of refine_wall), angle_deg (rotation of the wall from the
    expected direction, CCW about z, base frame), steps) or None."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
    scale = img.shape[1] / 1280.0
    sL, kL = edge_profile(g, T_tool, e, d_in, z, np.linspace(-lat, -0.012, 5), scale=scale)
    sR, kR = edge_profile(g, T_tool, e, d_in, z, np.linspace(0.012, lat, 5), scale=scale)
    sC, kC = edge_profile(g, T_tool, e, d_in, z, np.linspace(-0.02, 0.02, 9), scale=scale)
    out = dict(sL=sL, sR=sR, sC=sC, steps=[round(kL), round(kR), round(kC)])
    if sC is None or kC < min_step:
        return None
    out["shift"] = sC + 0.006
    if sL is not None and sR is not None and min(kL, kR) >= min_step:
        dl = lat + 0.012                                  # distance between the two lateral group centres... (centres at +-0.0235)
        dl = 2 * (lat + 0.012) / 2
        phi = np.degrees(np.arctan2(sR - sL, dl))         # the wall tangent turned from w toward d_in by phi
        out["angle_deg"] = float(-phi)                    # CCW rotation of the wall (and the drawer) about z
    return out
