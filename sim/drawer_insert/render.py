"""Image strip (and optional GIF) of a few sim episodes: rows = episodes, columns = steps, top + oblique view.

MUJOCO_GL=egl python -m sim.drawer_insert.render --policy weights/policy.npz --seeds 100001,100007 --out strip.png"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import params as P
from .env import DrawerEnv, action_of
from .evaluate import controller

COLS = (0, 6, 12, 18, 24, 32, 47)


def views(env, r):
    """Top view and oblique view of the current state, side by side."""
    imgs = []
    for el, az, dist in ((-89.0, 180.0, 0.32), (-35.0, 30.0, 0.42)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.0, 0.0, 0.04)
        cam.elevation, cam.azimuth, cam.distance = el, az, dist
        r.update_scene(env.d, cam)
        imgs.append(r.render())
    return np.concatenate(imgs, 0)


def episode(name, path, seed, variant="carry", r_size=(200, 240)):
    env = DrawerEnv(seed)
    o = env.reset(P.sample(np.random.default_rng(seed), variant))
    env.m.geom_rgba[env.m.geom("ceiling").id, 3] = 0.25
    r = mujoco.Renderer(env.m, *r_size)
    act, frames, info = controller(name, path), [views(env, r)], {}
    for k in range(48):
        o, _, done, info = env.step(action_of(act(o)))
        frames.append(views(env, r))
        if done:
            break
    r.close()
    label = f"{name} seed {seed}: e_t {env.e_t * 1e3:+.0f} mm, e_yaw {env.e_yaw:+.1f} deg, clr {env.p['clearance'] * 1e3:.1f} mm -> " + \
            ("SUCCESS" if info["success"] else f"fail (front {info['front_err'] * 1e3:.0f} mm)") + f", max F {info['max_f']:.0f} N"
    return frames, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--names", default="policy")
    ap.add_argument("--seeds", default="100001,100002,100003")
    ap.add_argument("--out", default="strip.png")
    ap.add_argument("--gif", default="")
    ap.add_argument("--variant", default="carry")
    a = ap.parse_args()
    rows, gif = [], []
    for name in a.names.split(","):
        for s in map(int, a.seeds.split(",")):
            frames, label = episode(name, a.policy, s, a.variant)
            row = Image.fromarray(np.concatenate([frames[min(c, len(frames) - 1)] for c in COLS], 1))
            canvas = Image.new("RGB", (row.width, row.height + 18), "white")
            canvas.paste(row, (0, 18)); ImageDraw.Draw(canvas).text((4, 3), label, fill="black")
            rows.append(canvas); gif += [Image.fromarray(f) for f in frames]
            print(label)
    strip = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        strip.paste(r, (0, y)); y += r.height
    strip.save(a.out)
    if a.gif:
        gif[0].save(a.gif, save_all=True, append_images=gif[1:], duration=100, loop=0)


if __name__ == "__main__":
    main()
