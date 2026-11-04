"""Calibration of the arm in use and frame helpers. Convention: T_A_B maps points in frame B into frame A (metres)."""
import functools
import hashlib
import json
import os

import numpy as np

from rlpj import config


def _j(name):
    with open(os.path.join(config.robot()["calibration_dir"], name)) as f:
        return json.load(f)


def ip():
    return config.robot()["ip"]


def side_serial():
    return config.robot()["side_camera_serial"]


def table_z():
    return float(config.robot()["table_z_m"])


def side_depth_gain():
    """side_left depth reads long: multiply raw depth by this."""
    return float(config.robot()["side_depth_gain"])


@functools.lru_cache(None)
def T_base_side():
    return np.array(_j(f"side_pose_{side_serial()}.json")["T_base_side"])


@functools.lru_cache(None)
def side_intrinsics():
    d = _j(f"intrinsics_{side_serial()}_1280x720.json")
    return np.array(d["K"]), np.array(d["dist"])


@functools.lru_cache(None)
def wrist_intrinsics():
    r = config.robot()
    d = _j(r["wrist_intrinsics"])
    if r["wrist_intrinsics_use"] == "factory":
        f = d["factory"]
        return np.array([[f["fx"], 0, f["cx"]], [0, f["fy"], f["cy"]], [0, 0, 1.0]]), np.array(f["coeffs"], float)
    return np.array(d["K"]), np.array(d["dist"], float)


@functools.lru_cache(None)
def T_tool_cam():
    return np.array(_j(config.robot()["wrist_handeye"])["T_tool_cam"])


def file_hashes():
    """sha256 (16 hex) of every calibration file of this arm, stored with each run."""
    r = config.robot()
    return {f: hashlib.sha256(open(os.path.join(r["calibration_dir"], f), "rb").read()).hexdigest()[:16] for f in r["files"]}


def euler_xyz_deg_to_R(tx, ty, tz):
    """Kortex tool_pose angles: extrinsic XYZ, R = Rz Ry Rx."""
    a, b, c = np.radians([tx, ty, tz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler_xyz_deg(R):
    ty = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), ty, np.arctan2(R[1, 0], R[0, 0])])


def pose_to_T(p):
    T = np.eye(4)
    T[:3, :3] = euler_xyz_deg_to_R(*p[3:6])
    T[:3, 3] = p[:3]
    return T


def T_to_pose(T):
    return np.concatenate([T[:3, 3], R_to_euler_xyz_deg(T[:3, :3])])


def apply(T, P):
    P = np.asarray(P, dtype=np.float64)
    return P @ T[:3, :3].T + T[:3, 3]


def deproject(depth_m, K, dist, mask=None, step=2):
    """Depth image (m) -> Nx3 points in the camera frame and their pixel coordinates."""
    import cv2
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    z = depth_m[::step, ::step]
    ok = z > 0.1
    if mask is not None:
        ok &= mask[::step, ::step]
    uv = np.stack([us[ok], vs[ok]], 1).astype(np.float64)
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    zz = z[ok]
    return np.stack([und[:, 0] * zz, und[:, 1] * zz, zz], 1), uv


def project(P_cam, K, dist):
    import cv2
    P = np.asarray(P_cam, dtype=np.float64).reshape(-1, 3)
    uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)
