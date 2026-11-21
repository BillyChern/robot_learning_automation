"""Evaluate the policy and scripted baselines on fixed randomized episodes, split by start error size.

python -m sim.drawer_insert.evaluate --policy sim/drawer_insert/weights/policy.npz -n 500 --out sim/drawer_insert/results"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

T_BINS = (0, 3, 6, 10, 20, 60)        # |lateral start error| bins (mm)
Y_BINS = (0, 2, 4, 6, 10, 25)         # |yaw start error| bins (deg)


def controller(name, policy_path=None):
    from . import baselines
    from .policy import Policy
    if name == "policy":
        pol = Policy(policy_path)
        return lambda o: pol.act(o)
    c = {"straight": baselines.Straight, "wiggle": baselines.Wiggle, "straight_z": baselines.StraightZ}[name]()
    return c.act


def run(args):
    """One episode: (controller name, policy path, seed) -> result dict."""
    name, path, seed, variant, kind, k_pre, v2 = (tuple(args) + (None, 0, None))[:7]
    from . import params as P
    from .env import DrawerEnv, action_of
    env = DrawerEnv(seed)
    o = env.reset(P.sample(np.random.default_rng(seed), variant, v2), kind, k_pre)
    act = controller(name, path)
    first = None
    for k in range(10**6):
        o, r, done, info = env.step(action_of(act(o)))
        if info["success"] and first is None:
            first = k + 1
        if done:
            break
    return dict(seed=seed, success=info["success"], front_err=info["front_err"], max_f=info["max_f"],
                n_stops=info["n_stops"], e_t=info["e_t"] * 1e3, e_yaw=info["e_yaw"], first_success=first,
                clearance=env.p["clearance"] * 1e3, mu=env.p["mu"], protrusion=info["protrusion"] * 1e3,
                stop_extra=env.p["stop_extra"] * 1e3, slot_dn=env.p["slot_dn"] * 1e3, f_end=info["f_end"],
                h_mode=env.p.get("h_mode", 0.0))


def table(res, key, bins, label):
    """Success rate per bin of |key|."""
    x = np.abs([r[key] for r in res]); s = np.array([r["success"] for r in res])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        rows.append(dict(bin=f"{label} {lo}-{hi}", n=int(m.sum()), success=float(s[m].mean()) if m.any() else None))
    return rows


def summary(res):
    s = np.array([r["success"] for r in res])
    fs = [r["first_success"] for r in res if r["first_success"]]
    return dict(n=len(res), success=float(s.mean()), max_f_median=float(np.median([r["max_f"] for r in res])),
                over_65n=float(np.mean([r["max_f"] > 65.0 for r in res])),
                first_success_median=float(np.median(fs)) if fs else None,
                stops_mean=float(np.mean([r["n_stops"] for r in res])),
                by_lateral=table(res, "e_t", T_BINS, "|e_t| mm"), by_yaw=table(res, "e_yaw", Y_BINS, "|e_yaw| deg"),
                by_clearance=table(res, "clearance", (3, 5, 7.5, 10.01), "clearance mm"),
                by_height=table(res, "h_mode", (0, 0.5, 1.5, 2.5), "height good/low/high"))


def report(folder):
    """Markdown tables from the eval_*.json files in folder."""
    rows, bins = [], []
    for f in sorted(Path(folder).glob("eval_*.json")):
        for name, v in json.loads(f.read_text()).items():
            sm, tag = v["summary"], f.stem[5:]
            rows.append(f"| {tag} | {name} | {sm['success']:.3f} | {sm['max_f_median']:.0f} | {sm['over_65n']:.2f} | "
                        f"{sm['stops_mean']:.1f} | {sm['first_success_median']} |")
            if tag == "carry":
                cells = [f"{r['success']:.2f} ({r['n']})" if r["n"] else "-" for r in sm["by_lateral"] + sm["by_yaw"] + sm["by_clearance"]]
                bins.append(f"| {name} | " + " | ".join(cells) + " |")
    head = "| start | controller | success | median max F (N) | share > 65 N | force stops/ep | steps to success |\n|---|---|---|---|---|---|---|"
    labels = [f"{a}-{b} mm" for a, b in zip(T_BINS[:-1], T_BINS[1:])] + [f"{a}-{b} deg" for a, b in zip(Y_BINS[:-1], Y_BINS[1:])] + \
             ["clr 3-5", "clr 5-7.5", "clr 7.5-10"]
    print(head + "\n" + "\n".join(rows))
    print("\n| carry | " + " | ".join(labels) + " |\n" + "|---" * (len(labels) + 1) + "|\n" + "\n".join(bins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--variant", default="carry", help="start variant: carry | aligned (params.VARIANTS)")
    ap.add_argument("--preroll", default="", help="start from a scripted pre-roll, e.g. push:12 or drift:16 (jammed / partly in)")
    ap.add_argument("--out", default="")
    ap.add_argument("--v2", default="", help="vertical model: carry-height mode (params.H_MODES: mix, wedge, good)")
    ap.add_argument("--episodes", action="store_true", help="also save per-episode results")
    ap.add_argument("--report", default="", help="print markdown tables of the eval_*.json in this folder and exit")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    kind, k_pre = (a.preroll.split(":")[0], int(a.preroll.split(":")[1])) if a.preroll else (None, 0)
    names = [n for n in ("straight", "wiggle", "straight_z", "policy") if (not a.only or n in a.only.split(","))]
    if not a.v2:
        names = [n for n in names if n != "straight_z"]
    if not a.policy:
        names = [n for n in names if n != "policy"]
    out = {}
    with mp.get_context("spawn").Pool(a.workers, initializer=os.nice, initargs=(15,)) as pool:
        for name in names:
            res = pool.map(run, [(name, a.policy, a.seed0 + i, a.variant, kind, k_pre, a.v2 or None) for i in range(a.n)], chunksize=4)
            out[name] = dict(summary=summary(res), episodes=res)
            sm = out[name]["summary"]
            print(f"{name:9s} success {sm['success']:.3f}  median max force {sm['max_f_median']:.0f} N  >65 N {sm['over_65n']:.2f}  "
                  f"stops {sm['stops_mean']:.1f}  first success step {sm['first_success_median']}")
            for row in sm["by_lateral"] + sm["by_yaw"] + sm["by_clearance"] + sm["by_height"]:
                print(f"   {row['bin']:16s} n {row['n']:4d}  success {row['success'] if row['success'] is None else round(row['success'], 3)}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        tag = ("v2" + a.v2 + "_" if a.v2 else "") + a.variant + (f"_{kind}{k_pre}" if kind else "")
        keep = out if a.episodes else {k: dict(summary=v["summary"]) for k, v in out.items()}
        Path(a.out, f"eval_{tag}.json").write_text(json.dumps(keep, default=float, indent=1))


if __name__ == "__main__":
    main()
