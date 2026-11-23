"""Image strip (and optional GIF) of lying-cabinet episodes: rows = episodes, columns = decisions; oblique + side view.

MUJOCO_GL=egl python -m sim.drawer_insert.lying_render --policy sim/drawer_insert/weights/policy_lying.npz --seeds 100001,100007"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import lying_params as P
from .lying_env import LyingEnv
from .lying_eval import controller
from .lying_policy import H_STEPS, action_of

COLS = (0, 5, 10, 15, 20, 30, 45, 59)


def views(env, r):
    """Oblique view from above and a side view along +y (near wall see-through), stacked."""
    imgs = []
    for el, az, dist, look in ((-50.0, 135.0, 0.36, (0, 0, 0.08)), (-8.0, 90.0, 0.34, (0, 0, 0.09))):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.elevation, cam.azimuth, cam.distance = el, az, dist
        r.update_scene(env.d, cam)
        imgs.append(r.render())
    return np.concatenate(imgs, 0)


def episode(name, path, seed, variant="carry", size=(200, 240)):
    env = LyingEnv(seed)
    o = env.reset(P.sample(np.random.default_rng(seed), variant))
    for g in ("wall_my", "lip_my", "edge_my"):
        env.m.geom_rgba[env.m.geom(g).id, 3] = 0.2
    r = mujoco.Renderer(env.m, *size)
    act, frames, info = controller(name, path), [views(env, r)], {}
    for k in range(H_STEPS):
        o, _, done, info = env.step(action_of(act(o)))
        frames.append(views(env, r))
        if done:
            break
    r.close()
    label = (f"{name} seed {seed}: start error {np.hypot(*env.e_xy) * 1e3:.0f} mm, yaw {env.e_yaw:+.1f} deg, tilt "
             f"{env.p['tilt']:.1f} deg -> " + ("SUCCESS" if info["success"] else f"fail (top {info['top_err'] * 1e3:.0f} mm off)") +
             f", max F {info['max_f']:.0f} N")
    return frames, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--names", default="policy")
    ap.add_argument("--seeds", default="100001,100002")
    ap.add_argument("--out", default="strip_lying.jpg")
    ap.add_argument("--gif", default="")
    ap.add_argument("--variant", default="carry")
    a = ap.parse_args()
    rows, gif = [], []
    for s in map(int, a.seeds.split(",")):
        for name in a.names.split(","):
            frames, label = episode(name, a.policy, s, a.variant)
            row = Image.fromarray(np.concatenate([frames[min(c + 1, len(frames) - 1) if c else 0] for c in COLS], 1))
            canvas = Image.new("RGB", (row.width, row.height + 18), "white")
            canvas.paste(row, (0, 18)); ImageDraw.Draw(canvas).text((4, 3), label, fill="black")
            rows.append(canvas); gif += [Image.fromarray(f) for f in frames]
            print(label)
    strip = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        strip.paste(r, (0, y)); y += r.height
    strip.save(a.out, quality=85)
    if a.gif:
        gif[0].save(a.gif, save_all=True, append_images=gif[1:], duration=120, loop=0)


if __name__ == "__main__":
    main()
