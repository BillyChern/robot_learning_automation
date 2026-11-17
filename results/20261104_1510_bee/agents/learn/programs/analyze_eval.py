"""Compact summary of eval episodes vs training episodes (runs on the 5090 after the episode sync).
usage: python analyze_eval.py <run_dir> <first_ep> <last_ep> [montage_eps comma]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

run, a, b = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
mont = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
out = Path(__file__).resolve().parent.parent / "notes" / "eval_frames"
out.mkdir(parents=True, exist_ok=True)


def stats(d):
    m = json.loads((d / "meta.json").read_text())
    s = np.load(d / "steps.npz")
    t, g, gc, tp = s["t"], s["gripper"], s["gripper_cmd"], s["tool_pose"]
    dur = t[-1] - t[0]
    closed = np.where(gc > 0.5)[0]
    tclose = (t[closed[0]] - t[0]) if len(closed) else None
    zmin_i = int(np.argmin(tp[:, 2]))
    o = m.get("outcome", {})
    return dict(ep=m.get("episode"), mode=m.get("mode"), final=o.get("final"), dur=round(float(dur), 1), n=len(t),
                t_close=None if tclose is None else round(float(tclose), 1), g_max=round(float(g.max()), 1),
                g_end=round(float(g[-1]), 1), zmin=round(float(tp[zmin_i, 2]), 3), xy_zmin=np.round(tp[zmin_i, :2], 3).tolist(),
                xy_end=np.round(tp[-1, :3], 3).tolist(), target=m.get("layout_target"), reason=(o.get("reason") or "")[:90])


def montage(d, name):
    for cam in ("side", "wrist"):
        cap = cv2.VideoCapture(str(d / f"{cam}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for k in np.linspace(0, n - 1, 8).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            if ok:
                f = cv2.resize(f, (320, 180))
                cv2.putText(f, f"{k / 15:.0f}s", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tiles.append(f)
        while len(tiles) < 8:
            tiles.append(np.zeros((180, 320, 3), np.uint8))
        img = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(str(out / f"{name}_{cam}.jpg"), cv2.resize(img, (640, 180)))


eps = run / "episodes"
print("== eval episodes")
for i in range(a, b + 1):
    d = eps / f"ep_{i:06d}"
    if (d / "meta.json").exists():
        print(json.dumps(stats(d)))
        lt = (d / "log.txt").read_text().splitlines() if (d / "log.txt").exists() else []
        for line in [x for x in lt if any(k in x.lower() for k in ("abort", "refus", "deviat", "error", "stop record", "traceback", "exception", "ignored", "descend", "time limit", "stop_reason"))][-4:]:
            print("   log:", line[:160])
print("== dagger / other episodes after", b)
for d in sorted(eps.glob("ep_*")):
    if int(d.name[3:]) > b and (d / "meta.json").exists():
        print(json.dumps(stats(d)))
print("== training (collect, success) sample")
rows = []
for d in sorted(eps.glob("ep_*"))[:108]:
    try:
        r = stats(d)
    except Exception:
        continue
    if r["mode"] == "collect" and r["final"] == "success":
        rows.append(r)
for k in ("dur", "t_close", "zmin", "g_max"):
    v = np.array([r[k] for r in rows if r[k] is not None], float)
    print(k, "median", round(float(np.median(v)), 3), "p10", round(float(np.percentile(v, 10)), 3), "p90", round(float(np.percentile(v, 90)), 3))
for r in rows[:3]:
    print(json.dumps(r))
for i in mont:
    montage(eps / f"ep_{i:06d}", f"ep{i}")
montage(eps / "ep_000005", "train_ep5")
print("montages in", out)
