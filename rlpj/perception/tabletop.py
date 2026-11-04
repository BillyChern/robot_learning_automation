"""Objects on the table from side depth: base-frame cloud, table fit, grid blobs and a ring test (tape rolls)."""
import numpy as np
from scipy import ndimage

from rlpj.robot import calib

WORKSPACE = dict(x=(0.22, 0.95), y=(-0.40, 0.20))   # base frame (m): x away from the robot, z up
TABLE_BAND = 0.006                                    # |z - table_z| below this counts as table
BLOB_KEYS = ("xy", "z_top", "height", "extent", "n", "hollow", "hole_xy", "inner_r", "outer_r")


def cloud(side_cam, n=5, K=None, dist=None, T=None, table_z=None):
    """Median of n side depth frames -> (N,3) base-frame points with the range-dependent depth error removed."""
    return from_depth(side_cam.depth_median(n), K, dist, T, table_z)


def from_depth(D, K=None, dist=None, T=None, table_z=None):
    """Side depth image (m) -> (N,3) base-frame points: deproject, depth_correction, T_base_side."""
    if K is None:
        K, dist = calib.side_intrinsics()
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    T = calib.T_base_side() if T is None else np.asarray(T, float)
    P, _ = calib.deproject(np.asarray(D, np.float32), np.asarray(K, float), dist)
    k, _ = depth_correction(P, T, table_z)
    return calib.apply(T, P * (k[0] + k[1] * P[:, 2:3]))


def depth_correction(P_cam, T, table_z=None, workspace=None):
    """Fit a depth scale f(d) = k0 + k1 d that puts the table points on table_z (the side depth error grows
    with range). Returns (k, info); k = [1, 0] when too few table points are seen."""
    tz = _table_prior() if table_z is None else float(table_z)
    if tz is None:
        raise ValueError("depth_correction needs table_z (no robot calibration available)")
    R, o = T[:3, :3], T[:3, 3]
    Pb = calib.apply(T, P_cam)
    m = (np.abs(Pb[:, 2] - tz) < 0.03) & _inside(Pb, workspace or WORKSPACE)
    d = P_cam[m, 2]
    info = dict(k=[1.0, 0.0], n=int(m.sum()), table_resid_mm=None)
    if len(d) < 200:
        return np.array([1.0, 0.0]), info
    uz = (P_cam[m] / d[:, None]) @ R[2]          # base z per metre of depth along each ray
    f = (tz - o[2]) / (uz * d)                    # scale that puts each point on the table
    inl = np.ones(len(d), bool)
    for _ in range(3):
        k, *_ = np.linalg.lstsq(np.c_[np.ones(inl.sum()), d[inl]], f[inl], rcond=None)
        res = np.abs(k[0] + k[1] * d - f)
        inl = res < max(3 * np.median(res), 0.002)
    zc = o[2] + uz * d * (k[0] + k[1] * d)
    info.update(k=np.round(k, 4).tolist(), d_range=[round(float(d.min()), 2), round(float(d.max()), 2)],
                table_resid_mm=round(float(np.std(zc[inl] - tz) * 1000), 2))
    return k, info


def table_fit(P, z0=None, win=0.04):
    """Table height: peak of a 2 mm histogram of z within win of the prior z0 (calibration, else the mode of z)."""
    z = np.asarray(P)[:, 2]
    if z0 is None:
        z0 = _table_prior()
    if z0 is None:
        h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.002, 0.002))
        z0 = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    zz = z[np.abs(z - z0) < win]
    if len(zz) < 100:
        return float(z0)
    h, e = np.histogram(zz, bins=np.arange(z0 - win, z0 + win + 0.002, 0.002))
    k = int(np.argmax(h))
    return float(np.median(zz[(zz > e[k] - 0.004) & (zz < e[k + 1] + 0.004)]))


def detect(P_base, workspace=None, min_height=0.008, max_height=0.30, grid=0.005, min_pts=60, table_z=None):
    """Grid-connected blobs of points min_height..max_height above the table inside the workspace, sorted by
    distance from the robot axis. Blob: xy, z_top, height, extent [dx, dy], n, points (M,3), hollow, hole_xy,
    inner_r, outer_r (ring test), table_xy (table points around the blob, for hole checks)."""
    ws = workspace or WORKSPACE
    P = np.asarray(P_base, np.float64)
    P = P[_inside(P, ws)]
    if len(P) == 0:
        return []
    tz = table_fit(P) if table_z is None else float(table_z)
    h = P[:, 2] - tz
    O = P[(h > min_height) & (h < max_height)]
    table_xy = P[np.abs(h) < TABLE_BAND, :2]
    lo = np.array([ws["x"][0], ws["y"][0]])
    shape = (int((ws["x"][1] - ws["x"][0]) / grid) + 2, int((ws["y"][1] - ws["y"][0]) / grid) + 2)
    lab, n = _label(O[:, :2], lo, shape, grid)
    blobs = []
    for k in range(1, n + 1):
        Q = O[lab == k]
        if len(Q) < min_pts:
            continue
        z_top = float(np.percentile(Q[:, 2], 98))
        ext = np.percentile(Q[:, :2], 98, 0) - np.percentile(Q[:, :2], 2, 0)
        b = dict(z_top=round(z_top, 4), height=round(z_top - tz, 4), extent=np.round(ext, 4).tolist(), n=int(len(Q)),
                 points=Q.astype(np.float32))
        b.update(_ring_test(Q, Q[:, 2] - tz, table_xy, grid))
        blobs.append(b)
    blobs.sort(key=lambda b: float(np.hypot(*b["xy"])))
    return blobs


def summarize(blobs):
    """JSON-safe blob list without point arrays (summary.json, prompts)."""
    return [{k: _js(b.get(k)) for k in BLOB_KEYS} for b in blobs]


def cells(xy, grid=0.005):
    """Number of distinct grid cells covered by xy points."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    return 0 if len(xy) == 0 else len(np.unique(np.floor(xy / grid).astype(np.int64), axis=0))


def _inside(P, ws):
    (x0, x1), (y0, y1) = ws["x"], ws["y"]
    return (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)


def _table_prior():
    try:
        return calib.table_z()
    except Exception:
        return None


def _label(xy, lo, shape, grid):
    """Connected components of occupied grid cells (>= 2 points, gaps of one cell closed) -> label per point."""
    if len(xy) == 0:
        return np.zeros(0, int), 0
    ij = np.clip(np.floor((xy - lo) / grid).astype(int), 0, np.array(shape) - 1)
    occ = np.zeros(shape, np.int32)
    np.add.at(occ, (ij[:, 0], ij[:, 1]), 1)
    lab, n = ndimage.label(ndimage.binary_closing(occ > 1, iterations=1))
    return lab[ij[:, 0], ij[:, 1]], n


def _ring_test(Q, h, table_xy, grid, nbins=24):
    """Centre and radii of a blob from the band at its most populated height (the flat top of a ring or cup):
    per-angle outer/inner radii, circle fits, hollow when the inner radius is consistent and the table shows
    inside the hole. Also returns table points near the blob."""
    hmax = float(np.percentile(h, 98))
    top = Q[h > hmax - max(0.008, 0.15 * hmax), :2]
    xy = top.mean(0) if len(top) >= 5 else Q[:, :2].mean(0)
    out = dict(xy=xy, hollow=False, hole_xy=None, inner_r=None, outer_r=None, table_xy=np.zeros((0, 2), np.float32))
    cnt, edges = np.histogram(h, bins=np.arange(0, hmax + 0.008, 0.004))
    k = int(np.argmax(cnt))
    band = Q[np.abs(h - 0.5 * (edges[k] + edges[k + 1])) < 0.006, :2]
    if len(band) < 10:
        return _finish(out)
    c = band.mean(0)
    ang = (np.arange(nbins) + 0.5) / nbins * 2 * np.pi - np.pi
    unit = np.c_[np.cos(ang), np.sin(ang)]
    ok = np.zeros(nbins, bool)
    for _ in range(2):
        rmin, rmax = _radial_profile(band, c, nbins)
        ok = np.isfinite(rmax)
        if ok.sum() < 0.6 * nbins:
            return _finish(out)
        c2, r_out, res = _fit_circle(c + rmax[ok, None] * unit[ok])
        if np.linalg.norm(c2 - c) > 0.03:
            return _finish(out)
        c = c2
    out["outer_r"] = float(np.median(rmax[ok]))
    out["inner_r"] = float(max(np.median(rmin[ok]), 0.0))
    if res < 0.004:                                   # round top: the circle centre beats the band centroid
        out["xy"] = c
    near = table_xy[np.linalg.norm(table_xy - c, axis=1) < out["outer_r"] + 0.01]
    out["table_xy"] = near.astype(np.float32)
    rin, rout = out["inner_r"], out["outer_r"]
    if rin < max(0.010, 0.3 * rout) or np.mean(rmin[ok] > 0.5 * rin) < 0.75:
        return _finish(out)
    c_in, r_in, _ = _fit_circle(c + rmin[ok, None] * unit[ok])
    if np.linalg.norm(c_in - c) > 0.015 or not (0.5 * rin < r_in < 1.5 * rin):
        c_in, r_in = c, rin
    if cells(near[np.linalg.norm(near - c_in, axis=1) < 0.75 * r_in], grid) >= 3:
        out.update(hollow=True, hole_xy=c_in, inner_r=float(r_in))
    return _finish(out)


def _finish(out):
    for k in ("xy", "hole_xy"):
        if out[k] is not None:
            out[k] = np.round(np.asarray(out[k], float), 4).tolist()
    for k in ("inner_r", "outer_r"):
        if out[k] is not None:
            out[k] = round(float(out[k]), 4)
    return out


def _radial_profile(xy, c, nbins):
    """Per angular bin around c: 10th and 90th percentile radius of the points (nan when under 3 points)."""
    d = xy - c
    r = np.hypot(d[:, 0], d[:, 1])
    b = np.minimum(((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * nbins).astype(int), nbins - 1)
    rmin, rmax = np.full(nbins, np.nan), np.full(nbins, np.nan)
    for i in range(nbins):
        s = b == i
        if s.sum() >= 3:
            rmin[i], rmax[i] = np.percentile(r[s], [10, 90])
    return rmin, rmax


def _fit_circle(xy):
    """Algebraic least-squares circle with one round of outlier rejection -> (centre, radius, median residual)."""
    xy = np.asarray(xy, float)
    keep = np.ones(len(xy), bool)
    c, r = xy.mean(0), 0.0
    for _ in range(2):
        if keep.sum() < 3:
            break
        A = np.c_[2 * xy[keep], np.ones(int(keep.sum()))]
        s, *_ = np.linalg.lstsq(A, (xy[keep] ** 2).sum(1), rcond=None)
        c = s[:2]
        r = float(np.sqrt(max(s[2] + c @ c, 1e-12)))
        res = np.abs(np.linalg.norm(xy - c, axis=1) - r)
        keep = res < max(2.5 * np.median(res), 0.003)
    return c, r, float(np.median(np.abs(np.linalg.norm(xy - c, axis=1) - r)))


def _js(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_js(x) for x in v]
    return v
