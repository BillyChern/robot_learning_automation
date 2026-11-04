"""Fit the kinematic model to the robot's FK oracle (read-only, ~1 min): python -m rlpj.robot.fit_kin [--n 300]
Collects (q, tool_pose) pairs with Base.ComputeForwardKinematics, fits per-joint origin perturbations on 70% of them
and reports the held-out error; writes runs/calib/<robot>/kin_fit.json (rlpj.robot.kin.fit_file()), pairs.json and
report.txt. Refuses to write when the held-out median exceeds MAX_HELDOUT_MM."""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from . import calib
from .kin import Kin, fit_file, rotvec

MAX_HELDOUT_MM = 3.0
X_SCALE = np.tile([0.005] * 3 + [0.01] * 3, 7)


def sample_joints(rng, near_work=False):
    """Random joint set (deg); near_work: typical working poses (gripper down)."""
    if near_work:
        return np.array([rng.uniform(-60, 60), rng.uniform(-40, 70), rng.uniform(40, 145) * rng.choice([-1, 1]),
                         rng.uniform(-30, 30), rng.uniform(-110, 110), rng.uniform(-180, 180)])
    return np.array([rng.uniform(-180, 180), rng.uniform(-110, 110), rng.uniform(-140, 140),
                     rng.uniform(-180, 180), rng.uniform(-115, 115), rng.uniform(-180, 180)])


def collect_pairs(robot, n=300, seed=1, log=print):
    """Measured (q, tool_pose) plus n oracle pairs (half near working poses). Read-only."""
    fb = robot.feedback()
    log(f"arm state {robot.arm_state()} q {fb['q'].round(2).tolist()} pose {fb['pose'].round(4).tolist()} gripper {fb['gripper']:.1f}")
    pairs = [dict(q=fb["q"].tolist(), pose=fb["pose"].tolist(), src="measured"),
             dict(q=fb["q"].tolist(), pose=robot.fk_oracle(fb["q"]).tolist(), src="oracle_current")]
    rng = np.random.default_rng(seed)
    fails = 0
    for k in range(n):
        q = sample_joints(rng, near_work=k < n // 2)
        try:
            pairs.append(dict(q=q.tolist(), pose=robot.fk_oracle(q).tolist(), src="oracle"))
        except Exception:
            fails += 1
    log(f"pairs {len(pairs)}, oracle failures {fails}")
    return pairs


def errors(kin, pairs, idx=None):
    """(pos mm, rot deg) of the model vs the recorded poses for the given indices."""
    idx = range(len(pairs)) if idx is None else idx
    ep, er = [], []
    for i in idx:
        T = kin.fk(pairs[i]["q"]); Tm = calib.pose_to_T(np.asarray(pairs[i]["pose"], float))
        ep.append(np.linalg.norm(T[:3, 3] - Tm[:3, 3])); er.append(np.degrees(np.linalg.norm(rotvec(T[:3, :3] @ Tm[:3, :3].T))))
    return np.array(ep) * 1000, np.array(er)


def stats(ep, er):
    return dict(n=int(len(ep)), pos_mm_median=float(np.median(ep)), pos_mm_p95=float(np.percentile(ep, 95)), pos_mm_max=float(ep.max()),
                rot_deg_median=float(np.median(er)), rot_deg_max=float(er.max()))


def fit(pairs, train_frac=0.7, seed=0, max_nfev=200, log=print):
    """Least-squares fit of the (7,6) origin perturbations; returns the kin_fit.json dict."""
    Q = [np.asarray(p["q"], float) for p in pairs]
    Tm = [calib.pose_to_T(np.asarray(p["pose"], float)) for p in pairs]
    perm = np.random.default_rng(seed).permutation(len(pairs))
    n_tr = int(train_frac * len(perm))
    train, test = perm[:n_tr], perm[n_tr:]

    def res(x):
        k = Kin(x.reshape(7, 6)); r = []
        for i in train:
            T = k.fk(Q[i])
            r.extend((T[:3, 3] - Tm[i][:3, 3]) * 1000)
            r.extend(np.degrees(rotvec(T[:3, :3] @ Tm[i][:3, :3].T)) * 2.0)
        return np.array(r)

    nominal = stats(*errors(Kin(), pairs))
    log(f"nominal URDF, all: {nominal}")
    t0 = time.time()
    x = least_squares(res, np.zeros(42), x_scale=X_SCALE, loss="soft_l1", f_scale=1.0, max_nfev=max_nfev).x
    k1 = Kin(x.reshape(7, 6))
    tr = stats(*errors(k1, pairs, train)); te = stats(*errors(k1, pairs, test)) if len(test) else tr
    meas = [i for i, p in enumerate(pairs) if p["src"] == "measured"]
    em = errors(k1, pairs, meas)[0] if meas else np.array([np.nan])
    log(f"fitted in {time.time() - t0:.0f}s: train {tr}")
    log(f"held-out: {te}; measured tool_pose {em.round(2).tolist()} mm")
    return dict(delta=x.reshape(7, 6).tolist(), n_pairs=len(pairs), n_train=int(n_tr), heldout_pos_mm_median=te["pos_mm_median"],
                heldout_pos_mm_max=te["pos_mm_max"], heldout_rot_deg_max=te["rot_deg_max"], train=tr, heldout=te, nominal=nominal,
                measured_pos_mm=float(em[0]), max_delta_pos_mm=float(np.abs(x.reshape(7, 6)[:, :3]).max() * 1000),
                max_delta_rot_deg=float(np.degrees(np.abs(x.reshape(7, 6)[:, 3:]).max())), fitted_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def write(result, pairs, path=None):
    """kin_fit.json + pairs.json + report.txt next to it; returns the fit path."""
    path = path or fit_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = dict(result, source=os.path.join(os.path.dirname(path), "pairs.json"))
    json.dump(result, open(path, "w"), indent=1)
    json.dump(pairs, open(result["source"], "w"))
    with open(os.path.join(os.path.dirname(path), "report.txt"), "w") as f:
        f.write(f"kinematic fit {result['fitted_at']}: {result['n_pairs']} pairs, {result['n_train']} train\n")
        f.write(f"nominal URDF  pos mm median {result['nominal']['pos_mm_median']:.2f} max {result['nominal']['pos_mm_max']:.2f}\n")
        f.write(f"fitted train  pos mm median {result['train']['pos_mm_median']:.2f} max {result['train']['pos_mm_max']:.2f}\n")
        f.write(f"fitted held-out pos mm median {result['heldout_pos_mm_median']:.2f} max {result['heldout_pos_mm_max']:.2f} "
                f"rot deg max {result['heldout_rot_deg_max']:.3f}\n")
        f.write(f"measured tool_pose now: {result['measured_pos_mm']:.2f} mm; max |delta| {result['max_delta_pos_mm']:.1f} mm / {result['max_delta_rot_deg']:.2f} deg\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300, help="oracle joint sets to query")
    ap.add_argument("--out", default=None, help="fit file (default: rlpj.robot.kin.fit_file())")
    ap.add_argument("--pairs", default=None, help="fit from an existing pairs.json instead of the robot")
    a = ap.parse_args(argv)
    if a.pairs:
        pairs = json.load(open(a.pairs))
    else:
        from .kortex import Robot
        robot = Robot(dry=True)
        try:
            pairs = collect_pairs(robot, a.n)
        finally:
            robot.close()
    result = fit(pairs)
    print(f"held-out: median {result['heldout_pos_mm_median']:.2f} mm, max {result['heldout_pos_mm_max']:.2f} mm "
          f"(nominal median {result['nominal']['pos_mm_median']:.2f} mm)")
    if result["heldout_pos_mm_median"] > MAX_HELDOUT_MM:
        print(f"REFUSED: held-out median {result['heldout_pos_mm_median']:.2f} mm > {MAX_HELDOUT_MM} mm; nothing written")
        return 1
    path = write(result, pairs, a.out)
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
