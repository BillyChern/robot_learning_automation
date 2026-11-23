"""Evaluate the lying-cabinet policy and the scripted baselines on fixed randomized episodes, split by error size.

python -m sim.drawer_insert.lying_eval --policy sim/drawer_insert/weights/policy_lying.npz -n 500 --out sim/drawer_insert/results/lying"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from .evaluate import table

XY_BINS = (0, 3, 6, 10, 15, 30)       # |start error of the drawer's bottom centre vs the true opening centre| (mm)
YAW_BINS = (0, 1, 2, 4, 10)           # |start yaw error| (deg)
TILT_BINS = (0, 2, 4, 6, 8.01)        # in-hand tilt (deg)
CLR_BINS = (2, 3, 4, 6.01)            # min(clr_x, clr_y) (mm)
NAMES = ("straight", "spiral", "guided", "policy")


def controller(name, path=None):
    from . import lying_baselines as B
    from .lying_policy import Policy
    if name == "policy":
        return Policy(path).act
    return {"straight": B.Straight, "spiral": B.Spiral, "guided": B.Guided}[name]().act


def run(args):
    """One episode: (controller, policy path, seed, variant, preroll kind, k_pre) -> result dict."""
    name, path, seed, variant, kind, k_pre, early = args
    from . import lying_params as P
    from .lying_env import LyingEnv
    from .lying_policy import action_of
    env = LyingEnv(seed)
    p = P.sample(np.random.default_rng(seed), variant)
    o = env.reset(p, kind, k_pre)
    act, first, seated = controller(name, path), None, 0
    for k in range(10**6):
        o, r, done, info = env.step(action_of(act(o)))
        if info["success"] and first is None:
            first = k + 1
        seated = seated + 1 if (o["z"] < 0.01 and o["f"][2] > 10.0) else 0
        if early and seated >= 2 and not done:                # robot rule: seated for two decisions -> release
            info.update(max_f=env.max_f, n_stops=env.n_stops, e_x=env.e_xy[0], e_y=env.e_xy[1], e_yaw=env.e_yaw)
            break
        if done:
            break
    return dict(seed=seed, success=info["success"], top_err=info["top_err"] * 1e3, max_f=info["max_f"], f_end=info["f_end"],
                n_stops=info["n_stops"], e_xy=float(np.hypot(info["e_x"], info["e_y"]) * 1e3), e_yaw=info["e_yaw"],
                tilt=p["tilt"], clr=min(p["clr_x"], p["clr_y"]) * 1e3, open_err=float(np.hypot(p["open_dx"], p["open_dy"]) * 1e3),
                first_success=first, depth=info["depth"] * 1e3)


def summary(res):
    s = np.array([r["success"] for r in res])
    fs = [r["first_success"] for r in res if r["first_success"]]
    return dict(n=len(res), success=float(s.mean()), max_f_median=float(np.median([r["max_f"] for r in res])),
                over_65n=float(np.mean([r["max_f"] > 65.0 for r in res])), stops_mean=float(np.mean([r["n_stops"] for r in res])),
                first_success_median=float(np.median(fs)) if fs else None,
                by_xy=table(res, "e_xy", XY_BINS, "|e_xy| mm"), by_yaw=table(res, "e_yaw", YAW_BINS, "|e_yaw| deg"),
                by_tilt=table(res, "tilt", TILT_BINS, "tilt deg"), by_clr=table(res, "clr", CLR_BINS, "clr mm"))


def report(folder):
    """Markdown tables from the eval_*.json files in folder."""
    rows, bins = [], []
    for f in sorted(Path(folder).glob("eval_*.json")):
        for name, v in json.loads(f.read_text()).items():
            sm, tag = v["summary"], f.stem[5:]
            rows.append(f"| {tag} | {name} | {sm['success']:.3f} | {sm['max_f_median']:.0f} | {sm['over_65n']:.2f} | "
                        f"{sm['stops_mean']:.1f} | {sm['first_success_median']} |")
            if tag == "carry":
                cells = [f"{r['success']:.2f} ({r['n']})" if r["n"] else "-" for r in sm["by_xy"] + sm["by_yaw"] + sm["by_tilt"] + sm["by_clr"]]
                bins.append(f"| {name} | " + " | ".join(cells) + " |")
    lab = lambda b, u: [f"{x:g}-{y:g} {u}" for x, y in zip(b[:-1], b[1:])]
    labels = lab(XY_BINS, "mm") + lab(YAW_BINS, "deg") + ["tilt " + x for x in lab(TILT_BINS, "deg")] + ["clr " + x for x in lab(CLR_BINS, "mm")]
    print("| start | controller | success | median max F (N) | share > 65 N | force stops/ep | steps to success |\n|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n| carry | " + " | ".join(labels) + " |\n" + "|---" * (len(labels) + 1) + "|\n" + "\n".join(bins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--variant", default="carry", help="lying_params.VARIANTS key")
    ap.add_argument("--preroll", default="", help="start from a scripted pre-roll, e.g. descend:10 or drift:12")
    ap.add_argument("--early", action="store_true", help="end when seated (z < 1 cm, f_z > 10 N) for two decisions")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="", help="print markdown tables of the eval_*.json in this folder and exit")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    kind, k_pre = (a.preroll.split(":")[0], int(a.preroll.split(":")[1])) if a.preroll else (None, 0)
    names = [n for n in NAMES if (not a.only or n in a.only.split(",")) and (n != "policy" or a.policy)]
    out = {}
    with mp.get_context("spawn").Pool(a.workers, initializer=os.nice, initargs=(15,)) as pool:
        for name in names:
            res = pool.map(run, [(name, a.policy, a.seed0 + i, a.variant, kind, k_pre, a.early) for i in range(a.n)], chunksize=4)
            out[name] = dict(summary=summary(res), episodes=res)
            sm = out[name]["summary"]
            print(f"{name:9s} success {sm['success']:.3f}  median max force {sm['max_f_median']:.0f} N  >65 N {sm['over_65n']:.2f}  "
                  f"stops {sm['stops_mean']:.1f}  first success step {sm['first_success_median']}")
            for row in sm["by_xy"] + sm["by_yaw"] + sm["by_tilt"] + sm["by_clr"]:
                print(f"   {row['bin']:16s} n {row['n']:4d}  success {row['success'] if row['success'] is None else round(row['success'], 3)}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        tag = a.variant + (f"_{kind}{k_pre}" if kind else "") + ("_early" if a.early else "")
        f = Path(a.out, f"eval_{tag}.json")
        old = json.loads(f.read_text()) if f.exists() else {}                 # merge: baselines and policy run separately
        old.update({k: dict(summary=v["summary"]) for k, v in out.items()})
        f.write_text(json.dumps({k: old[k] for k in NAMES if k in old}, default=float, indent=1))


if __name__ == "__main__":
    main()
