"""Episodes (DESIGN.md section 4) -> LeRobot dataset for openpi. Runs inside the openpi venv on the 5090:

    cd third_party/openpi && PYTHONPATH=<repo> uv run python -m rlpj.vla.convert --run-dir runs/<run> --name <name> [--overwrite]

Dataset: HF_LEROBOT_HOME = runs/<run>/datasets, repo_id rlpj/<name> -> runs/<run>/datasets/rlpj/<name>/ (+ manifest.json).
Features: image, wrist_image uint8 224x224x3 RGB (letterboxed from 640x360 like openpi's resize_with_pad; dtype
'image' = PNG in parquet, or --videos), state float32 (7) = [q rad wrapped to (-pi, pi], gripper reading / 100],
actions float32 (7) = [q_cmd rad (state + wrapped command delta), gripper_cmd 0..1], task = instruction, fps = meta.fps.
Only episodes with outcome.final == success and mode in --modes are used unless --all. Module level needs numpy only;
cv2 for video, lerobot only inside build_dataset (load_episode is testable without it)."""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .images import SIZE, bgr_to_model

STATE_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
REPO_PREFIX = "rlpj"


def wrap_pi(a):
    """Wrap radians to (-pi, pi]."""
    return -((-np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi)


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def state_actions(q_deg, gripper, q_cmd_deg, gripper_cmd):
    """(N,7) state and (N,7) actions, float32: joints rad (state wrapped, action = state + wrapped cmd delta)."""
    q = np.asarray(q_deg, float).reshape(-1, 6)
    qc = np.asarray(q_cmd_deg, float).reshape(-1, 6)
    g = np.asarray(gripper, float).reshape(-1)
    gc = np.asarray(gripper_cmd, float).reshape(-1)
    g = np.where(np.isfinite(g), g, 0.0) / 100.0
    gc = np.where(np.isfinite(gc), gc, g)
    sq = wrap_pi(np.deg2rad(q))
    aq = sq + np.deg2rad(wrap180(qc - q))
    state = np.concatenate([sq, np.clip(g, 0, 1)[:, None]], 1).astype(np.float32)
    actions = np.concatenate([aq, np.clip(gc, 0, 1)[:, None]], 1).astype(np.float32)
    return state, actions


def read_video(path, n=None, size=SIZE):
    """All (or the first n) frames of an mp4 as RGB uint8 (M, size, size, 3)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while n is None or len(frames) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr_to_model(bgr, size))
    cap.release()
    if not frames:
        return np.zeros((0, size, size, 3), np.uint8)
    return np.stack(frames)


class Skip(Exception):
    """Episode left out on purpose (reason in the message)."""


def wrist_flipped(q_deg):
    """True where joint 4 is in the flipped wrist family (90..270 deg); the rollout's look pose uses the normal one."""
    q4 = np.mod(np.asarray(q_deg, float).reshape(-1, 6)[:, 3], 360.0)
    return (q4 > 90.0) & (q4 < 270.0)


def idle_keep_mask(q_deg, q_cmd_deg, gripper, gripper_cmd, horizon=10, tol_deg=0.3, grip_tol=2.0, keep=2):
    """Bool mask of frames to keep: drop frames whose next `horizon` commands move no joint more than tol_deg while the
    gripper command stays the same and the reading changes < grip_tol (a scripted pause); keep the first `keep` frames
    of every pause. A single frame cannot tell when a pause ends, so pauses teach the policy to stall."""
    q = np.asarray(q_deg, float).reshape(-1, 6)
    qc = np.asarray(q_cmd_deg, float).reshape(-1, 6)
    g = np.nan_to_num(np.asarray(gripper, float).reshape(-1))
    gc = np.nan_to_num(np.asarray(gripper_cmd, float).reshape(-1))
    n = len(q)
    idle = np.zeros(n, bool)
    for k in range(n):
        j = min(n, k + horizon)
        move = np.abs(wrap180(qc[k:j] - q[k])).max()
        idle[k] = move < tol_deg and np.ptp(gc[k:j]) == 0 and abs(g[j - 1] - g[k]) < grip_tol
    mask = ~idle
    run = 0
    for k in range(n):
        run = run + 1 if idle[k] else 0
        if idle[k] and run <= keep:
            mask[k] = True
    return mask


def load_episode(ep_dir, size=SIZE, with_images=True, wrist=None, start_phase=None, drop_idle=False):
    """Read one episode dir -> dict(meta, n, fps, instruction, state, actions, image, wrist_image, phases, notes).
    wrist="normal": raise Skip if joint 4 ever enters the flipped family. start_phase: drop frames before the first
    phase of that name (collect episodes only; DAgger corrections start at the policy's pose and keep their recovery).
    drop_idle: drop scripted pauses (idle_keep_mask)."""
    ep = Path(ep_dir)
    meta = json.loads((ep / "meta.json").read_text())
    st = np.load(ep / "steps.npz")
    n = int(len(st["q"]))
    if wrist == "normal" and wrist_flipped(st["q"]).any():
        raise Skip(f"wrist flipped in {int(wrist_flipped(st['q']).sum())} of {n} steps")
    notes = []
    out = dict(meta=meta, fps=int(round(float(meta["fps"]))), instruction=str(meta.get("instruction", "")), notes=notes)
    if (ep / "phases.json").exists():
        out["phases"] = json.loads((ep / "phases.json").read_text())
    if with_images:
        side = read_video(ep / "side.mp4", n, size)
        wrist = read_video(ep / "wrist.mp4", n, size)
        m = min(n, len(side), len(wrist))
        if m < n:
            notes.append(f"truncated {n} steps to {m} video frames (side {len(side)}, wrist {len(wrist)})")
        out["image"], out["wrist_image"] = side[:m], wrist[:m]
        n = m
    state, actions = state_actions(st["q"][:n], st["gripper"][:n], st["q_cmd"][:n], st["gripper_cmd"][:n])
    keep = np.ones(n, bool)
    if start_phase and meta.get("mode", "collect") == "collect":      # DAgger corrections keep their recovery
        i0 = next((int(p["i0"]) for p in out.get("phases", []) if p.get("name") == start_phase), None)
        if i0 is not None:
            keep[:min(i0, n)] = False
            notes.append(f"starts at phase {start_phase} (frame {i0})")
    if drop_idle:
        idle_ok = idle_keep_mask(st["q"][:n], st["q_cmd"][:n], st["gripper"][:n], st["gripper_cmd"][:n])
        notes.append(f"dropped {int((keep & ~idle_ok).sum())} idle frames")
        keep &= idle_ok
    if not keep.all():
        idx = np.flatnonzero(keep)
        state, actions, n = state[idx], actions[idx], len(idx)
        if with_images:
            out["image"], out["wrist_image"] = out["image"][idx], out["wrist_image"][idx]
    out.update(n=n, state=state, actions=actions)
    return out


def select_episodes(run_dir, episodes_glob="episodes/ep_*", all_outcomes=False, modes=("collect",), fps=None):
    """Return (used dirs, excluded [(dir, reason)], fps) after reading only meta.json of each candidate."""
    run_dir = Path(run_dir)
    used, excluded = [], []
    for d in sorted(glob.glob(str(run_dir / episodes_glob))):
        d = Path(d)
        missing = [f for f in ("meta.json", "steps.npz", "side.mp4", "wrist.mp4") if not (d / f).exists()]
        if missing:
            excluded.append((str(d), f"missing {','.join(missing)}"))
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception as e:
            excluded.append((str(d), f"bad meta.json: {e}"))
            continue
        final = (meta.get("outcome") or {}).get("final")
        if not all_outcomes and final != "success":
            excluded.append((str(d), f"outcome.final={final!r}"))
            continue
        if modes and meta.get("mode") not in modes:
            excluded.append((str(d), f"mode={meta.get('mode')!r}"))
            continue
        ep_fps = int(round(float(meta.get("fps", 0))))
        if fps is None:
            fps = ep_fps
        if ep_fps != fps:
            excluded.append((str(d), f"fps {ep_fps} != {fps}"))
            continue
        if int(meta.get("n_steps", 2)) < 2:
            excluded.append((str(d), "n_steps < 2"))
            continue
        used.append(d)
    return used, excluded, fps


def dataset_root(run_dir, name):
    return Path(run_dir).resolve() / "datasets" / REPO_PREFIX / name


def features(size=SIZE, videos=False):
    img = {"dtype": "video" if videos else "image", "shape": (size, size, 3), "names": ["height", "width", "channel"]}
    return {
        "image": dict(img),
        "wrist_image": dict(img),
        "state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "actions": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    }


def build_dataset(run_dir, name, episode_dirs, fps, overwrite=False, videos=False, size=SIZE, log=print, wrist=None,
                  start_phase=None, drop_idle=False):
    """Write the LeRobot dataset under runs/<run>/datasets/rlpj/<name>; returns the manifest dict."""
    run_dir = Path(run_dir).resolve()
    home = run_dir / "datasets"
    os.environ["HF_LEROBOT_HOME"] = str(home)
    root = dataset_root(run_dir, name)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite to rebuild it")
        shutil.rmtree(root)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"{REPO_PREFIX}/{name}"
    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), root=root, robot_type="kinova_gen3_6dof",
                               features=features(size, videos), use_videos=videos, image_writer_threads=8)
    manifest = dict(repo_id=repo_id, root=str(root), fps=int(fps), image_size=size, image_dtype="video" if videos else "image",
                    state="[q1..q6 rad wrapped (-pi,pi], gripper reading/100]", actions="[q_cmd1..6 rad, gripper_cmd 0..1]",
                    episodes=[], excluded=[], n_frames=0, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    options=dict(wrist=wrist, start_phase=start_phase, drop_idle=drop_idle))
    for i, d in enumerate(episode_dirs):
        t0 = time.time()
        try:
            ep = load_episode(d, size, wrist=wrist, start_phase=start_phase, drop_idle=drop_idle)
        except Skip as e:
            manifest["excluded"].append(dict(dir=str(d), reason=str(e)))
            log(f"[{i + 1}/{len(episode_dirs)}] {Path(d).name}: left out ({e})")
            continue
        except Exception as e:
            manifest["excluded"].append(dict(dir=str(d), reason=f"load failed: {e!r}"))
            log(f"[{i + 1}/{len(episode_dirs)}] {d}: skipped ({e!r})")
            continue
        if ep["n"] < 2 or not ep["instruction"]:
            manifest["excluded"].append(dict(dir=str(d), reason="too short or no instruction"))
            continue
        for k in range(ep["n"]):
            ds.add_frame({"image": ep["image"][k], "wrist_image": ep["wrist_image"][k], "state": ep["state"][k],
                          "actions": ep["actions"][k], "task": ep["instruction"]})
        ds.save_episode()
        manifest["episodes"].append(dict(dir=str(d), episode=ep["meta"].get("episode"), n_frames=ep["n"],
                                         instruction=ep["instruction"], notes=ep["notes"]))
        manifest["n_frames"] += ep["n"]
        log(f"[{i + 1}/{len(episode_dirs)}] {d.name}: {ep['n']} frames ({time.time() - t0:.1f}s)")
    stop = getattr(ds, "stop_image_writer", None)
    if callable(stop):
        stop()
    manifest["n_episodes"] = len(manifest["episodes"])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--name", required=True, help="dataset name -> repo_id rlpj/<name>")
    p.add_argument("--episodes-glob", default="episodes/ep_*", help="relative to run dir")
    p.add_argument("--all", action="store_true", help="keep every outcome, not only outcome.final == success")
    p.add_argument("--modes", default="collect,dagger", help="comma list of meta.mode to keep ('' = any); dagger = expert corrections")
    p.add_argument("--overwrite", action="store_true", help="remove an existing dataset of the same name first")
    p.add_argument("--videos", action="store_true", help="store images as AV1 videos instead of PNG-in-parquet")
    p.add_argument("--size", type=int, default=SIZE)
    p.add_argument("--fps", type=int, default=None, help="expected recording fps (default: first episode's)")
    p.add_argument("--limit", type=int, default=None, help="use at most this many episodes")
    p.add_argument("--dry-run", action="store_true", help="only list the selection (no lerobot needed)")
    p.add_argument("--wrist", choices=["normal"], default=None, help="leave out episodes that use the flipped wrist (joint 4 90..270)")
    p.add_argument("--start-phase", default=None, help="drop frames before this phase, e.g. approach (= the look pose)")
    p.add_argument("--drop-idle", action="store_true", help="drop scripted pauses (keep 2 frames of each, keep gripper changes)")
    a = p.parse_args(argv)
    modes = tuple(m for m in a.modes.split(",") if m)
    used, excluded, fps = select_episodes(a.run_dir, a.episodes_glob, a.all, modes, a.fps)
    if a.limit:
        used = used[: a.limit]
    print(f"{len(used)} episodes selected, {len(excluded)} excluded, fps={fps}")
    for d, why in excluded:
        print(f"  excluded {Path(d).name}: {why}")
    if a.dry_run:
        return 0
    if not used:
        print("nothing to convert", file=sys.stderr)
        return 2
    manifest = build_dataset(a.run_dir, a.name, used, fps, a.overwrite, a.videos, a.size, wrist=a.wrist,
                             start_phase=a.start_phase, drop_idle=a.drop_idle)
    manifest["excluded"] = [dict(dir=d, reason=r) for d, r in excluded] + manifest["excluded"]
    (Path(manifest["root"]) / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {manifest['root']}: {manifest['n_episodes']} episodes, {manifest['n_frames']} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
