"""Synthetic episodes in the DESIGN.md section 4 format (for tests and pipeline checks without the robot):

    python -m rlpj.vla.synthetic_episodes --out /tmp/rlpj_synth --n 2 [--steps 60] [--fps 15]

writes <out>/episodes/ep_<n>/{meta.json, steps.npz, side.mp4, wrist.mp4, phases.json, summary.json} so that
`rlpj.vla.convert --run-dir <out>` works. Joints are Kinova style (deg in [0, 360)), q_cmd leads q by one tick."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

Q_HOME = np.array([0.0, 26.0, -95.0, 0.0, -58.0, 0.0])        # tool ~ (0.45, 0, 0.20) pointing down
INSTRUCTION = "put the foam bee in the tape roll"


def _video(path, frames_bgr, fps):
    import cv2
    fourcc = cv2.VideoWriter.fourcc(*"mp4v") if hasattr(cv2.VideoWriter, "fourcc") else cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()


def make_episode(ep_dir, n_steps=60, fps=15, seed=0, outcome="success", mode="collect", instruction=INSTRUCTION,
                 size=(640, 360), episode=None):
    """Write one synthetic episode; returns the meta dict."""
    ep = Path(ep_dir)
    ep.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) / fps
    amp = rng.uniform(5, 15, 6) * np.array([1, 1, 1, 0.5, 1, 2])
    q_unwrapped = Q_HOME + amp * np.sin(2 * np.pi * t[:, None] / max(t[-1], 1e-6) * rng.uniform(0.5, 1.0, 6))
    q_cmd_unwrapped = np.vstack([q_unwrapped[1:], q_unwrapped[-1:]])          # command = next tick's pose
    q = q_unwrapped % 360.0                                                 # Kinova feedback range
    q_cmd = q_cmd_unwrapped % 360.0
    dq = np.gradient(q_unwrapped, 1 / fps, axis=0)
    close_from = n_steps // 2
    gripper_cmd = np.where(np.arange(n_steps) >= close_from, 0.8, 0.0)
    gripper = np.clip(np.concatenate([[0.0], gripper_cmd[:-1]]) * 100 + rng.normal(0, 0.3, n_steps), 0, 100)
    tool_pose = np.zeros((n_steps, 6)); tool_pose[:, :3] = [0.45, 0.0, 0.2]
    phase = np.where(np.arange(n_steps) >= close_from, 1, 0)
    np.savez(ep / "steps.npz", t=t, q=q, dq=dq, tool_pose=tool_pose, gripper=gripper, q_cmd=q_cmd,
             gripper_cmd=gripper_cmd, phase=phase)
    w, h = size
    side, wrist = [], []
    for k in range(n_steps):
        im = np.full((h, w, 3), (40, 60, 80), np.uint8)
        x = int(w * (0.2 + 0.6 * k / max(n_steps - 1, 1)))
        im[h // 2 - 20:h // 2 + 20, max(x - 20, 0):x + 20] = (0, 0, 255)      # red block moving (BGR)
        side.append(im)
        wim = np.full((h, w, 3), (90, 90, 90), np.uint8)
        wim[:, : int(w * k / n_steps)] = (255, 128, 0)
        wrist.append(wim)
    _video(ep / "side.mp4", side, fps)
    _video(ep / "wrist.mp4", wrist, fps)
    phases = [{"name": "approach", "i0": 0, "i1": close_from}, {"name": "grasp", "i0": close_from, "i1": n_steps}]
    (ep / "phases.json").write_text(json.dumps(phases))
    meta = dict(episode=episode if episode is not None else ep.name, run_id="synthetic", task="bee_in_tape_roll",
                instruction=instruction, mode=mode, program={"name": "synthetic", "commit": "", "command": ""},
                robot={"name": "kinova_a", "ip": "192.168.2.9"}, fps=fps, layout_target={}, layout_before={},
                layout_after={}, t_start=time.time(), t_end=time.time() + t[-1], n_steps=n_steps,
                outcome={"claim": outcome, "perception": None, "vlm": None, "operator": None, "final": outcome,
                         "reason": "synthetic"}, calibration_hashes={})
    (ep / "meta.json").write_text(json.dumps(meta, indent=1))
    (ep / "summary.json").write_text(json.dumps({"claimed_outcome": outcome}))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--fail-every", type=int, default=0, help="mark every k-th episode as fail (tests selection)")
    a = p.parse_args(argv)
    for i in range(a.n):
        outcome = "fail" if a.fail_every and (i + 1) % a.fail_every == 0 else "success"
        make_episode(Path(a.out) / "episodes" / f"ep_{i:06d}", a.steps, a.fps, seed=i, outcome=outcome, episode=i)
    print(f"wrote {a.n} episodes under {Path(a.out) / 'episodes'}")


if __name__ == "__main__":
    main()
