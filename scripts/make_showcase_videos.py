"""Showcase videos of run 20261104_1510_bee (toy bee into the bowl, Kinova Gen3, pi0.5) for posting on X.

Every clip is a real recording of the run (dashboard.mp4: side + wrist camera with the program's overlays and log);
numbers come from the run's events.jsonl, training logs and evaluations. 1920x1080, 30 fps, H.264 + silent AAC,
each video <= 2:20 (X's limit without Premium).

    third_party/openpi/.venv/bin/python scripts/make_showcase_videos.py [story agents resets collection training eval_dagger final highlights human]

"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import showcase_music
from showcase_common import (BG, DIM, FPS, H, PANEL, TXT, W, Video, Writer, balanced, blank, fade, fit, font, mono,
                             split_dash, text, wrap)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20261104_1510_bee"
OUT = RUN / "videos"
WORK = OUT / "_work"
COL = dict(execute=(88, 166, 255), reset=(255, 166, 87), collect=(88, 166, 255), policy=(63, 200, 100),
           dagger=(190, 140, 255), train=(255, 210, 80), fail=(255, 90, 90), target=(255, 210, 80), human=(255, 112, 67))


# ------------------------------------------------------------------ run data
def T(s):
    return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


EVENTS = [json.loads(l) for l in open(RUN / "events.jsonl")]
T0 = T(EVENTS[0]["t"])                                  # 2026-11-04 15:10:35, run created
EPS = [(T(e["t"]) - T0, e.get("mode"), e.get("final")) for e in EVENTS if e["kind"] == "episode"]
RESETS = [T(e["t"]) - T0 for e in EVENTS if e["kind"] == "reset"]
EVALS = [(T(e["t"]) - T0, e["id"], e["success"], e["n"]) for e in EVENTS if e["kind"] == "eval"]
GOAL_T = next(t for t, i, s, n in EVALS if i.startswith("22_"))


def stats_at(t):
    """Counters at run time t (s since the start)."""
    c = sum(1 for x in EPS if x[0] <= t and x[1] == "collect" and x[2] == "success")
    r = sum(1 for x in RESETS if x <= t)
    ev = sum(1 for x in EPS if x[0] <= t and x[1] == "eval")
    dg = sum(1 for x in EPS if x[0] <= t and x[1] == "dagger")
    return dict(demos=c, resets=r, rollouts=ev, dagger=dg)


def clock(t):
    lt = time.localtime(T0 + t)
    return f"DAY {int(t // 86400) + 1}", time.strftime("%a %b %d  %H:%M", lt)


def clip_index():
    """Every robot recording with a dashboard: start (s since T0), role, attempt dir, episode, mode, final."""
    p = WORK / "clip_index.json"
    if p.exists():
        return json.load(open(p))
    fin = {e["episode"]: (e.get("mode"), e.get("final")) for e in EVENTS if e["kind"] == "episode"}
    idx = []
    for d in glob.glob(f"{RUN}/agents/*/attempts/attempt_*") + glob.glob(f"{RUN}/episodes/ep_*/attempt"):
        m, v = d + "/attempt.json", d + "/dashboard.mp4"
        if not (os.path.exists(m) and os.path.exists(v)):
            continue
        a = json.load(open(m))
        if not a.get("moves_robot") or "exit_code" not in a:
            continue
        role = d.split("/")[-3] if "/agents/" in d else "policy"
        rd = a.get("record_dir") or (os.path.dirname(d) if role == "policy" else None)
        ep = int(os.path.basename(rd)[3:]) if rd and os.path.basename(rd).startswith("ep_") else None
        mode, final = fin.get(ep, (None, None))
        idx.append(dict(t=float(a["started"]) - T0, role=role, dir=d, what=str(a.get("what") or "")[:160],
                        outcome=a.get("outcome"), dur=float(a.get("duration_s") or 0), ep=ep, mode=mode, final=final))
    idx.sort(key=lambda x: x["t"])
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump(idx, open(p, "w"))
    return idx


IDX = clip_index()


def rec(suffix):
    return next(x for x in IDX if x["dir"].endswith(suffix))


def agent_note(rc):
    """The agent's own description of a robot run (attempt.json 'what')."""
    return "Agent's note: " + " ".join(str(json.load(open(rc["dir"] + "/attempt.json")).get("what") or "").split())


def by_ep(ep):
    return next(x for x in IDX if x["ep"] == ep and (x["mode"] == "dagger" or x["role"] == "policy"))


# ------------------------------------------------------------------ drawing helpers


@lru_cache(None)
def bee_sprite(h, right=True):
    """A small cartoon bee (RGBA), h pixels tall, facing right (or left)."""
    w0, h0 = 300, 220
    im = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (30, 30, 30, 255)
    d.ellipse([110, 8, 190, 98], fill=(205, 232, 255, 215), outline=ink, width=6)
    d.ellipse([160, 18, 236, 102], fill=(205, 232, 255, 215), outline=ink, width=6)
    body = Image.new("L", (w0, h0), 0)
    ImageDraw.Draw(body).ellipse([30, 70, 270, 210], fill=255)
    layer = Image.new("RGBA", (w0, h0), (255, 205, 40, 255))
    ld = ImageDraw.Draw(layer)
    for x in (118, 168, 218):
        ld.rectangle([x, 0, x + 26, h0], fill=ink)
    im.paste(layer, (0, 0), body)
    d.ellipse([30, 70, 270, 210], outline=ink, width=7)
    d.polygon([(266, 130), (298, 140), (266, 150)], fill=ink)
    d.ellipse([60, 108, 86, 134], fill=ink)
    d.ellipse([66, 113, 75, 122], fill=(255, 255, 255, 255))
    d.arc([52, 128, 98, 172], 25, 115, fill=ink, width=6)
    d.line([(72, 84), (44, 32)], fill=ink, width=6)
    d.ellipse([33, 21, 55, 43], fill=ink)
    d.line([(98, 76), (94, 20)], fill=ink, width=6)
    d.ellipse([83, 10, 105, 32], fill=ink)
    im = im.resize((int(w0 * h / h0), h), Image.LANCZOS)
    return ImageOps.mirror(im) if right else im


def paste_bee(pil, cx, cy, h, angle=0.0, right=True):
    spr = bee_sprite(h, right)
    if angle:
        spr = spr.rotate(angle, resample=Image.BICUBIC, expand=True)
    pil.paste(spr, (int(cx - spr.width / 2), int(cy - spr.height / 2)), spr)


def fly_bee(pil, k, n, y0=250, h=86):
    """The mascot flies across the frame on a wavy path, leaving a dotted trail."""
    def pos(kk):
        u = kk / max(1, n - 1)
        return -170 + (W + 340) * u, y0 + 38 * np.sin(2 * np.pi * 1.5 * u)
    d = ImageDraw.Draw(pil)
    for j in range(2, 40, 3):
        if k - j < 0:
            break
        x, y = pos(k - j)
        r = max(2, 6 - j // 8)
        d.ellipse([x - 70 - r, y + 18 - r, x - 70 + r, y + 18 + r], fill=(70, 78, 90))
    x, y = pos(k)
    paste_bee(pil, x, y, h, angle=-12 * np.cos(2 * np.pi * 1.5 * k / max(1, n - 1)))


# ------------------------------------------------------------------ layouts
def header(d, chapter, accent, sub=None, t=None, speed=None):
    d.rectangle([0, 0, W, 124], fill=PANEL)
    d.rectangle([0, 0, 10, 124], fill=accent)
    text(d, (40, 14), chapter, 54, "B", fill=TXT)
    if sub:
        text(d, (40, 80), sub, 30, "R", fill=DIM)
    if t is not None:
        day, ts = clock(t)
        text(d, (W - 40, 12), day, 54, "B", fill=accent, anchor="ra")
        text(d, (W - 40, 80), ts, 30, "R", fill=DIM, anchor="ra")
    if speed:
        x = W - 360 if t is not None else W - 40
        d.rounded_rectangle([x - 180, 36, x - 10, 92], 14, fill=accent)
        text(d, (x - 95, 64), f"{speed:g}x speed", 30, "B", fill=BG, anchor="mm")


def counters(d, t, y=1018, accent=TXT):
    s = stats_at(t)
    n_help = sum(1 for x in INCIDENTS if x["t"] <= t)
    items = [("demos", s["demos"], accent), ("robot resets", s["resets"], accent), ("policy tests", s["rollouts"], accent),
             ("DAgger fixes", s["dagger"], accent), ("human rescues", n_help, COL["human"])]
    d.rectangle([0, y - 14, W, H], fill=PANEL)
    for i, (k, v, col) in enumerate(items):
        x = 40 + i * 372
        text(d, (x, y), f"{v:,}", 40, "B", fill=col)
        text(d, (x + font(40, "B").getlength(f"{v:,}") + 12, y + 10), k, 28, "R", fill=DIM)


def timeline_bar(d, t, pil=None, y=990):
    """Run progress: day ticks across the 156 h, marker at t, goal flag at 80 %, a red tick per human intervention."""
    total = (EPS[-1][0] if EPS else GOAL_T) + 600
    x0, x1 = 40, W - 40
    d.rectangle([x0, y, x1, y + 6], fill=(45, 52, 62))
    d.rectangle([x0, y, x0 + (x1 - x0) * min(t, total) / total, y + 6], fill=COL["policy"])
    for k in range(1, 7):
        xx = x0 + (x1 - x0) * k * 86400 / total
        d.line([xx, y - 6, xx, y + 12], fill=(80, 88, 100), width=2)
    xg = x0 + (x1 - x0) * GOAL_T / total
    d.line([xg, y - 14, xg, y + 18], fill=COL["target"], width=3)
    for inc in INCIDENTS:
        if inc["t"] <= t:
            xi = x0 + (x1 - x0) * inc["t"] / total
            d.rectangle([xi - 3, y - 9, xi + 3, y + 15], fill=COL["human"])
    if pil is not None:                    # the mascot marks "now"
        paste_bee(pil, x0 + (x1 - x0) * min(t, total) / total, y - 4 + 3 * np.sin(t / 900.0), 40)


def dash_segment(rc, speed, chapter, sub, caption, accent, t0=0.0, t1=None, note=None, story=False, badge=None):
    """One recording, full layout: header with clock, side + wrist cameras, the program's log, caption."""
    v = Video(rc["dir"] + "/dashboard.mp4")
    t0 = max(t0, 0.25)
    k_badge = None
    span = (min(t1, v.dur) if t1 else v.dur) - t0
    for k, fr in enumerate(v.frames(speed, t0, t1)):
        side, wrist, log = split_dash(fr)
        img = blank()
        img[130:670, 0:960] = fit(side, 960, 540)
        img[130:670, 960:1920] = fit(wrist, 960, 540)
        img[676:856, 0:1920] = fit(log, 1920, 180)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        t_run = rc["t"] + t0 + k * speed / FPS
        header(d, chapter, accent, sub, t_run, speed)
        text(d, (16, 140), "side camera", 26, "M", fill=TXT, shadow=True)
        text(d, (976, 140), "wrist camera", 26, "M", fill=TXT, shadow=True)
        if badge and k * speed / FPS > span * 0.8:
            k_badge = k if k_badge is None else k_badge
            f = k - k_badge
            sc = 0.5 + 0.7 * f / 5 if f < 5 else 1.2 - 0.2 * min(1.0, (f - 5) / 5)
            col = {"ok": COL["policy"], "fail": COL["fail"], "human": COL["human"]}[badge[0]]
            d.rounded_rectangle([1610 - 290 * sc, 766 - 74 * sc, 1610 + 290 * sc, 766 + 74 * sc], int(24 * sc), fill=col)
            text(d, (1610, 766), badge[1], max(12, int(50 * sc)), "B", fill=BG, anchor="mm")
        y = 864
        for line in wrap(caption, 44, W - 80, "M")[:2]:
            text(d, (40, y), line, 44, "M", fill=TXT)
            y += 54
        if note and not story:
            lines = []
            for part in ([note] if isinstance(note, str) else note):
                lines += wrap(part, 30, W - 80)
            if len(lines) > 2:
                cut = lines[1].split(" ")
                while cut and font(30).getlength(" ".join(cut) + " …") > W - 80:
                    cut.pop()
                lines[1] = " ".join(cut).rstrip(",;:") + " …"
            for line in lines[:2]:
                text(d, (40, y + 6), line, 30, "R", fill=DIM)
                y += 38
        if story:
            counters(d, t_run, accent=accent)
            timeline_bar(d, t_run, pil)
        yield np.asarray(pil)


def card(lines, dur, accent=TXT, first=False, bee=None):
    """Title card: lines of (text, size, weight, color), centred; text slides up into place; small text is enlarged for
    phones. first: no fade-in (the first frame is what X shows before playing). bee: y of a mascot fly-by."""
    rows = []
    for s, size, weight, col in lines:
        size = size if size >= 40 else size + 6
        rows += [(line, size, weight, col) for line in balanced(s, size, W - 240, weight)]
    n = int(dur * FPS)

    def render(k):
        pil = Image.fromarray(blank())
        d = ImageDraw.Draw(pil)
        off = int(46 * (1 - min(1.0, k / 10)) ** 2)
        y = (H - sum(size + 24 for _, size, _, _ in rows)) // 2
        for line, size, weight, col in rows:
            text(d, (W // 2, y + size // 2 + off), line, size, weight, fill=col, anchor="mm")
            y += size + 24
        d.rectangle([W // 2 - 80, H - 120, W // 2 + 80, H - 114], fill=accent)
        if bee is not None:
            fly_bee(pil, k, n, y0=bee)
        return np.asarray(pil)

    def frames():
        still = None
        for k in range(n):
            if k <= 10 or bee is not None:
                yield render(k)
            else:
                still = render(10) if still is None else still
                yield still
    return fade(frames(), 0 if first else 10, 10)


def grid_segment(recs, cols, rows, speed, chapter, sub, caption, accent, t_story=None, badges=False, dur=None,
                 counter=None):
    """Many recordings at once (side camera), each tile freezing on its last frame (+ a pass/fail badge)."""
    tw, th = W // cols, int((W // cols) * 9 / 16)
    gy = 130
    gens = [Video(r["dir"] + "/dashboard.mp4").frames(speed) for r in recs]
    last = [None] * len(recs)
    done = [False] * len(recs)
    done_k = [None] * len(recs)
    n_frames = int(dur * FPS) if dur else int(max(Video(r["dir"] + "/dashboard.mp4").dur for r in recs) / speed * FPS) + 45
    for k in range(n_frames):
        img = blank()
        for i, g in enumerate(gens):
            if not done[i]:
                try:
                    last[i] = next(g)
                except StopIteration:
                    done[i] = True
                    done_k[i] = k
            if last[i] is None:
                continue
            r, c = divmod(i, cols)
            x, y = c * tw, gy + r * th
            img[y:y + th - 4, x:x + tw - 4] = fit(split_dash(last[i])[0], tw - 4, th - 4)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub, None if t_story is None else t_story, speed)
        n_ok = 0
        for i, rc in enumerate(recs):
            r, c = divmod(i, cols)
            x, y = c * tw, gy + r * th
            if badges and done[i]:
                ok = rc.get("final") == "success"
                n_ok += ok
                col = COL["policy"] if ok else COL["fail"]
                f = k - done_k[i]
                sc = 0.4 + 0.9 * f / 5 if f < 5 else 1.3 - 0.3 * min(1.0, (f - 5) / 5)
                cx, cy = x + tw - 42, y + 38
                d.rounded_rectangle([cx - 28 * sc, cy - 28 * sc, cx + 28 * sc, cy + 28 * sc], int(12 * sc), fill=col)
                text(d, (cx, cy), "✓" if ok else "✗", max(10, int(40 * sc)), "B", fill=BG, anchor="mm")
        y_cap = gy + rows * th + 14
        if badges:
            fin = sum(done)
            head = f"{n_ok} / {fin} in the bowl" if fin else f"{len(recs)} runs, side by side"
            text(d, (40, y_cap), head, 44, "B", fill=COL["policy"])
            text(d, (40 + font(44, "B").getlength(head) + 24, y_cap + 12), caption, 32, "R", fill=DIM)
        elif counter:
            text(d, (40, y_cap), counter(k / FPS), 44, "B", fill=accent)
            text(d, (40, y_cap + 58), caption, 32, "R", fill=DIM)
        else:
            for j, line in enumerate(wrap(caption, 40, W - 80, "M")[:2]):
                text(d, (40, y_cap + j * 50), line, 40, "M", fill=TXT)
        yield np.asarray(pil)


def wall_segment(dur, chapter, sub, accent):
    """All 1,422 robot runs as tiles, appearing in time order over the 6.5 days (colour = role)."""
    th_dir = WORK / "thumbs"
    cols, tw, th = 42, 44, 25
    gx, gy = (W - cols * tw) // 2, 132
    marked = {inc["idx"] for inc in INCIDENTS if inc.get("idx") is not None}
    thumbs = []
    for i, r in enumerate(IDX):
        p = th_dir / f"{i:05d}.jpg"
        im = cv2.imread(str(p)) if p.exists() else None
        thumbs.append(None if im is None else cv2.cvtColor(cv2.resize(im, (tw - 1, th - 1)), cv2.COLOR_BGR2RGB))
    t_end = IDX[-1]["t"]
    n = int(dur * FPS)
    img = blank()
    shown = 0
    for k in range(n):
        t = t_end * min(1.0, (k + 1) / (n * 0.85))
        while shown < len(IDX) and IDX[shown]["t"] <= t:
            r, c = divmod(shown, cols)
            x, y = gx + c * tw, gy + r * th
            role = IDX[shown]["mode"] if IDX[shown]["mode"] in ("dagger", "collect") else IDX[shown]["role"]
            col = {"reset": COL["reset"], "collect": COL["collect"], "policy": COL["policy"], "dagger": COL["dagger"]}.get(role, DIM)
            tile = np.empty((th - 1, tw - 1, 3), np.float32)
            tile[:] = col
            if thumbs[shown] is not None:
                tile = 0.55 * thumbs[shown][:th - 1, :tw - 1] + 0.45 * tile
                tile[:3] = col
            img[y:y + th - 1, x:x + tw - 1] = tile.astype(np.uint8)
            shown += 1
        pil = Image.fromarray(img.copy())
        d = ImageDraw.Draw(pil)
        for i in marked:
            if i < shown:
                r, c = divmod(i, cols)
                x, y = gx + c * tw, gy + r * th
                d.rectangle([x - 3, y - 3, x + tw + 1, y + th + 1], outline=COL["human"], width=4)
        header(d, chapter, accent, sub, t)
        y = gy + 34 * th + 12
        text(d, (40, y), f"{shown:,} robot runs", 44, "B", fill=TXT)
        lx = 400
        for name, key in (("reset", "reset"), ("demo", "collect"), ("policy test", "policy"), ("DAgger", "dagger"), ("agent test", None),
                          ("human to the rescue", "human")):
            if key == "human":
                d.rectangle([lx, y + 10, lx + 28, y + 40], outline=COL["human"], width=4)
            else:
                d.rectangle([lx, y + 12, lx + 26, y + 38], fill=COL[key] if key else DIM)
            text(d, (lx + 38, y + 8), name, 30, "R", fill=COL["human"] if key == "human" else DIM)
            lx += 60 + font(30).getlength(name) + 30
        yield np.asarray(pil)


def make_thumbs():
    th_dir = WORK / "thumbs"
    th_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, r in enumerate(IDX):
        p = th_dir / f"{i:05d}.jpg"
        if not p.exists():
            jobs.append(f"ffmpeg -v error -y -ss {max(0.0, r['dur'] * 0.55):.1f} -i '{r['dir']}/dashboard.mp4' "
                        f"-vf crop=640:360:0:0,scale=92:52 -frames:v 1 '{p}'")
    if jobs:
        (WORK / "thumb_jobs.sh").write_text("\n".join(jobs) + "\n")
        subprocess.run(f"xargs -P 12 -I CMD bash -c CMD < '{WORK / 'thumb_jobs.sh'}'", shell=True, check=False)


# ------------------------------------------------------------------ charts
def loss_curves():
    """Per-step training loss from the progress bar of each full fine-tune, median per 150-step bin (checkpoint-save
    spikes in the periodic log lines are left out)."""
    out = {}
    for v in ("full_v1", "full_v5", "full_v7"):
        p = WORK / f"train_{v}.log"
        if not p.exists():
            continue
        txt = p.read_text(errors="ignore").replace("\r", "\n")
        pts = {}
        for m in re.finditer(r"loss=([0-9.]+), lr=[^,]+, step=([0-9.e+]+)\]", txt):
            pts[int(float(m.group(2)))] = float(m.group(1))
        if len(pts) < 100:
            continue
        s = np.array(sorted(pts.items()), float)
        bins = np.arange(0, 30001, 150)
        xs, ys = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = s[(s[:, 0] >= a) & (s[:, 0] < b), 1]
            if len(sel):
                xs.append((a + b) / 2)
                ys.append(float(np.median(sel)))
        out[v] = (np.array(xs), np.array(ys))
    return out


def chart_segment(dur, chapter, accent, sub, kind):
    """kind 'loss': training loss (binned median) of three full fine-tunes; kind 'success': success rate per model vs the 70 % goal."""
    n = int(dur * FPS)
    curves = loss_curves() if kind == "loss" else None
    models = [("LoRA ×3\nNov 6-7\nraw data", 0, 54), ("full_v1\nNov 8\nraw data", 0, 40), ("full_v5\nNov 9\ncleaned data", 25, 40),
              ("full_v6\nNov 9\nfrom v5, LR 5e-5", 3, 20), ("full_v7\nNov 10\nfrom v5, LR 1e-5", 16, 20), ("full_v8\nNov 11\nv7 + 64 demos", 11, 20)]
    for k in range(n):
        p = min(1.0, (k + 1) / (n * (0.7 if kind == "loss" else 0.5)))
        pil = Image.fromarray(blank())
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub)
        x0, y0, x1, y1 = 180, 190, W - 120, 860
        d.line([x0, y1, x1, y1], fill=(80, 88, 100), width=3)
        d.line([x0, y0, x0, y1], fill=(80, 88, 100), width=3)
        if kind == "loss":
            cols = {"full_v1": (120, 130, 145), "full_v5": COL["execute"], "full_v7": COL["policy"]}
            lo, hi = np.log10(5e-4), np.log10(1e-1)
            for v_ in (1e-3, 1e-2):
                yy = y1 - (np.log10(v_) - lo) / (hi - lo) * (y1 - y0)
                d.line([x0, yy, x1, yy], fill=(40, 46, 56), width=1)
                text(d, (x0 - 16, yy), f"{v_:g}", 30, "R", fill=DIM, anchor="rm")
            for i, (name, (st, ls)) in enumerate(curves.items()):
                m = max(2, int(len(st) * p))
                pts = [(x0 + st[j] / 30000 * (x1 - x0), y1 - (np.log10(max(ls[j], 5e-4)) - lo) / (hi - lo) * (y1 - y0)) for j in range(m)]
                d.line(pts, fill=cols[name], width=5)
                text(d, (x1 - 320, y0 + 20 + i * 50), name, 38, "B", fill=cols[name])
            for s_ in (0, 10000, 20000, 30000):
                text(d, (x0 + s_ / 30000 * (x1 - x0), y1 + 32), f"{s_ // 1000}k", 30, "R", fill=DIM, anchor="mm")
            text(d, ((x0 + x1) // 2, y1 + 82), "training step  ·  full_v5 starts from full_v1, and full_v7 from full_v5", 32, "R", fill=DIM, anchor="mm")
            text(d, (x0 - 120, (y0 + y1) // 2), "loss", 32, "R", fill=DIM, anchor="mm")
        else:
            bw = (x1 - x0) / len(models)
            yg = y1 - 0.70 * (y1 - y0)
            d.line([x0, yg, x1, yg], fill=COL["target"], width=3)
            text(d, (x0 + 16, yg - 12), "goal 70 %", 36, "B", fill=COL["target"], anchor="lb")
            for i, (name, s_, n_) in enumerate(models):
                rate = s_ / n_
                q = min(1.0, max(0.0, p * len(models) - i))
                hh = rate * q * (y1 - y0)
                xa = x0 + i * bw + bw * 0.18
                xb = x0 + (i + 1) * bw - bw * 0.18
                col = COL["policy"] if rate >= 0.7 else (COL["execute"] if rate > 0.3 else (120, 130, 145))
                d.rectangle([xa, y1 - hh, xb, y1], fill=col)
                if q > 0.95:
                    text(d, ((xa + xb) / 2, y1 - hh - 14), f"{int(rate * 100 + 0.5)} %", 48, "B", fill=TXT, anchor="mb")
                    text(d, ((xa + xb) / 2, y1 - hh - 70), f"{s_}/{n_}", 30, "R", fill=DIM, anchor="mb")
                for j, line in enumerate(name.split("\n")):
                    if j == 2 and q < 0.95:
                        continue
                    col_j = (TXT, DIM, {"from v5, LR 5e-5": COL["fail"], "from v5, LR 1e-5": COL["policy"], "v7 + 64 demos": COL["execute"]}.get(line, DIM))[j]
                    text(d, ((xa + xb) / 2, y1 + 22 + j * 40), line, (36, 30, 30)[j], ("B", "R", "M")[j], fill=col_j, anchor="mt")
        yield np.asarray(pil)


def code_segment(dur, chapter, accent, sub):
    """The two programs the agents wrote, scrolling side by side."""
    files = [(RUN / "agents/execute/programs/execute.py", "Agent 1  ·  execute.py", "878 lines · 32 commits", COL["execute"]),
             (RUN / "agents/reset/programs/reset.py", "Agent 2  ·  reset.py", "1,915 lines · 127 commits", COL["reset"])]
    srcs = [p.read_text().splitlines() for p, *_ in files]
    n = int(dur * FPS)
    lh, rows = 24, 33
    for k in range(n):
        pil = Image.fromarray(blank())
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub)
        for c, ((p, title, stats, col), lines) in enumerate(zip(files, srcs)):
            x = 40 + c * 940
            d.rounded_rectangle([x, 136, x + 900, 1050], 16, fill=PANEL)
            text(d, (x + 24, 152), title, 34, "B", fill=col)
            text(d, (x + 876, 160), stats, 26, "R", fill=DIM, anchor="ra")
            start = int((len(lines) - rows) * k / n)
            for j in range(rows):
                i = start + j
                if i >= len(lines):
                    break
                d.text((x + 24, 214 + j * lh), f"{i + 1:5d}  {lines[i][:66]}", font=mono(20), fill=col if lines[i].lstrip().startswith(("def ", "class ")) else (200, 208, 218))
        yield np.asarray(pil)


# ------------------------------------------------------------------ output
def chapter_card(num, title, sub, accent, dur=2.0):
    return card([(f"STEP {num}", 34, "B", accent), (title, 72, "B", TXT), (sub, 34, "R", DIM)], dur, accent, bee=300)


def faded(gen, n_in=6, n_out=6):
    return fade(gen, n_in, n_out)


# ------------------------------------------------------------------ clip choices (all real recordings of this run)
FIRST_EXEC = rec("execute/attempts/attempt_3")         # Agent 1's first robot run, success (Nov 4 15:14)
RESET_FAIL = rec("reset/attempts/attempt_2")            # Agent 2's first robot run, error
RESET_OK = rec("reset/attempts/attempt_4")              # its second robot run, success
RESET_HERO = rec("reset/attempts/attempt_474")          # a collection-time reset: bee out of the bowl, both to new targets
RESET_GRID = [rec(f"reset/attempts/attempt_{n}") for n in (666, 469, 485, 314, 368, 454)]   # bowl moved 12-18 cm
_collect = [x for x in IDX if x["mode"] == "collect" and x["final"] == "success" and x["dur"] < 190]
COLLECT_GRID = [_collect[i] for i in np.linspace(0, len(_collect) - 1, 24).astype(int)]
COLLECT_HERO = next(x for x in _collect[len(_collect) // 2:] if 120 < x["dur"] < 170)   # a mid-run demonstration
V1_FAIL = next(x for x in IDX if x["role"] == "policy" and x["ep"] == 284)        # full_v1, eval 08: 0/20
DAG_FAIL = next(x for x in IDX if x["role"] == "policy" and x["ep"] == 520)       # full_v5, eval 18: grasps miss, time out
DAG_FIX = next(x for x in IDX if x["mode"] == "dagger" and x["ep"] == 521)        # its correction, in full_v7's training data
FINAL20 = [x for x in IDX if x["role"] == "policy" and x["ep"] and 626 <= x["ep"] <= 647]
HERO = next(x for x in IDX if x["role"] == "policy" and x["ep"] == 644)
HIGHLIGHTS = [next(x for x in FINAL20 if x["ep"] == e) for e in (629, 633, 637, 645)]
assert len(FINAL20) == 20 and sum(x["final"] == "success" for x in FINAL20) == 16

POLICY_NOTE = "Every try starts with a scripted move that puts the camera over the bowl. After that, the policy drives."
END_NOTE = "The policy never says it's done, so a safety check stops the arm after it lets go. The cameras decide if it worked."

# Every time a person had to step in (OPERATOR_LOG.md, "Human interventions" and milestones). clips: (attempt dir
# suffix, speed, t0, t1, caption); the first clip's run is marked on the wall of all runs.
INCIDENTS = [
    dict(when="2026-11-04T20:39", title="Hide and seek", person="A person put the bee back on the table.",
         fix="After that, the bee only goes where the camera can see it well.",
         clips=[("reset/attempts/attempt_60", 12, 30, 130, "A reset parked the bee near the robot's base, right where the camera "
                 "can barely see it. Six resets later: \"bee not found\". Run stopped.")]),
    dict(when="2026-11-04T21:55", title="Clingy bee", person="A person pulled the bee off the gripper.",
         fix="The reset agent added a check for a stuck bee.",
         clips=[("reset/attempts/attempt_72", 14, 60, 172, "The gripper opened, but the bee would not let go of a fingertip. "
                 "It rode along for two whole resets.")]),
    dict(when="2026-11-04T22:01", title="Out of reach", person="A person moved the bee closer.",
         fix="The reset agent swapped its distance limit for a real reach check.",
         clips=[("reset/attempts/attempt_79", 2, 0, 13, "The bee ended up 75 cm away. The reset program only reached to 62 cm, "
                 "so it gave up.")]),
    dict(when="2026-11-04T23:29", title="The great escape", person="A person picked the bee up off the floor.",
         fix="Now every robot program ends with a gripper check, so a stuck bee stops the run before it can fall.",
         clips=[("reset/attempts/attempt_92", 11, 55, 148, "The bee got wedged between the fingers when the gripper opened..."),
                ("reset/attempts/attempt_93", 9, 0, 54, "...rode along for two more resets, then fell. Straight onto the floor.")]),
    dict(when="2026-11-05T02:43", title="Over the edge", person="In the morning a person put it back, patched up with orange tape.",
         fix="The gripper may now only open over the middle of the table.",
         clips=[("reset/attempts/attempt_119", 7, 45, 110, "Stuck to a fingertip again. This time it fell off the table on the "
                 "curtain side while the arm went home.")]),
    dict(when="2026-11-06T02:38", title="Network down", person="A person rebooted the arm the next morning.", fix=None, clips=[],
         what="The robot lost its network connection in the middle of its very first policy test."),
    dict(when="2026-11-06T13:22", title="Butterfingers", person="A person put the bee back.",
         fix="Now a bee stuck in the gripper gets set down right below, never carried.",
         clips=[("reset/attempts/attempt_209", 5, 3, 45, "A policy test ended with the bee pinched by its tip. The reset tried "
                 "to carry it over and dropped it.")]),
    dict(when="2026-11-07T12:35", title="Curtain call",
         person="A person cleared the fault, and the camera was lined back up and recalibrated.",
         fix="Now there's a keep-in box around the work area, checked for every move the arm makes.",
         clips=[("ep_000209/attempt", 7, 15, 75, "A policy under test reached way out of the work area and grabbed the "
                 "curtain. It had picked up that swing from the demos..."),
                ("reset/attempts/attempt_285", 7, 3, 58, "...then the next reset dragged the curtain across the table and the "
                 "arm faulted. The side camera ended up pointing somewhere else.")]),
    dict(when="2026-11-10T09:10", title="One last escape", person="A person put the bee back at 13:03.",
         fix="The release check now also looks at the wrist camera for a bee on the fingers.",
         clips=[("execute/attempts/attempt_407", 3.5, 0, 30, "After a policy test the bee was still hanging on the gripper. The "
                 "DAgger run dropped it, and it vanished from the table.")]),
]
for _n, _inc in enumerate(INCIDENTS, 1):
    _inc["n"] = _n
    _inc["t"] = T(_inc["when"] + ":00") - T0
    _inc["recs"] = [(rec(c[0]),) + tuple(c[1:]) for c in _inc["clips"]]
    _inc["idx"] = next((i for i, x in enumerate(IDX) if _inc["recs"] and x["dir"] == _inc["recs"][0][0]["dir"]), None)


def bee_frame(ep, frame):
    """One frame of an episode's wrist video (RGB)."""
    cap = cv2.VideoCapture(str(RUN / f"episodes/ep_{ep:06d}/wrist.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def tape_segment(dur, extra=None):
    """The bee before (episode 58) and after (episode 59) its orange-tape repair on Nov 5."""
    a = COL["human"]
    before, after = fit(bee_frame(58, 1300), 900, 506), fit(bee_frame(59, 1100), 900, 506)
    n = int(dur * FPS)
    for k in range(n):
        img = blank()
        img[170:676, 40:940] = before
        q = min(1.0, max(0.0, (k / n - 0.2) / 0.15))
        img[170:676, 980:1880] = (after * q + img[170:676, 980:1880] * (1 - q)).astype(np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        header(d, "Plot twist: the bee broke", a, "Nov 5, human rescue #5")
        text(d, (490, 700), "before · Nov 5, 02:30 · episode 58", 34, "B", fill=TXT, anchor="ma")
        if q > 0.5:
            text(d, (1430, 700), "after · Nov 5, 11:27 · episode 59", 34, "B", fill=a, anchor="ma")
        text(d, (W // 2, 800), "We patched it with orange tape. Same bee, new look.", 56, "B", fill=TXT, anchor="ma")
        text(d, (W // 2, 886), "The robot's color checks had to learn orange, too.", 36, "R", fill=DIM, anchor="ma")
        if extra:
            text(d, (W // 2, 944), extra, 32, "R", fill=DIM, anchor="ma")
        yield np.asarray(pil)


def incident_frames(inc, speed_scale=1.0, story=False, short=False):
    """One human intervention: its recordings with what happened, what the person did and what changed after."""
    a = COL["human"]
    head = f"#{inc['n']} · {inc['title']}"
    sub = f"Human rescue {inc['n']} of {len(INCIDENTS)}"
    note = None if short else [inc["person"]] + ([inc["fix"]] if inc["fix"] else [])
    for j, (rc, speed, t0, t1, cap) in enumerate(inc["recs"]):
        last = j == len(inc["recs"]) - 1
        yield from fade(dash_segment(rc, round(speed * speed_scale), head, sub, cap, a, t0, t1, note=note, story=story,
                                     badge=("human", "human to the rescue") if last else None), 6, 6)


def incident_card(inc, dur=5.0):
    lines = [(f"HUMAN RESCUE #{inc['n']}  ·  {time.strftime('%b %d, %H:%M', time.localtime(T0 + inc['t']))}", 30, "B", COL["human"]),
             (inc["title"], 72, "B", TXT), (inc["what"], 40, "R", TXT), (inc["person"], 34, "R", DIM)]
    return card(lines, dur, COL["human"])


JUMPS = [   # (result, models, what changed, note): training logs, datasets/rlpj/*/manifest.json, REPORT.md
    ("63% → 15%", COL["fail"], "full_v5 → full_v6",
     "Kept training full_v5 at the same learning rate (5e-5), adding DAgger runs mostly from an older 0% model.", None),
    ("15% → 80%", COL["policy"], "full_v6 → full_v7",
     "Restarted from full_v5 with a 5× lower learning rate (1e-5), and DAgger runs only from full_v5's own failures.",
     "Two changes at once."),
    ("80% → 55%", COL["execute"], "full_v7 → full_v8",
     "Same recipe, plus 64 more demos.", "Not significant with 20 trials each (p = 0.18)."),
]


def changes_frame():
    """Why the success rate went down, up and down again: one row per jump (one still frame)."""
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    text(d, (120, 130), "Why the success rate went up and down", 60, "B", fill=TXT)
    text(d, (120, 212), "Full fine-tunes of π0.5, 30k steps each. Success rate on the real robot.", 32, "R", fill=DIM)
    y0, row_h, x_text, w_text = 334, 240, 700, W - 700 - 120
    for k, (res, col, models, change, note) in enumerate(JUMPS):
        y = y0 + k * row_h
        d.line([120, y - 30, W - 120, y - 30], fill=(38, 44, 54), width=2)
        text(d, (120, y), res, 76, "B", fill=col)
        text(d, (124, y + 96), models, 30, "R", fill=DIM)
        lines = wrap(change, 40, w_text, "R")
        assert len(lines) <= 2, change
        for m_, line in enumerate(lines):
            text(d, (x_text, y + 8 + m_ * 54), line, 40, "R", fill=TXT)
        if note:
            assert font(32).getlength(note) <= w_text, note
            text(d, (x_text, y + 8 + len(lines) * 54 + 10), note, 32, "R", fill=DIM)
    return np.asarray(pil)


def dips_card(dur=11.0):
    """Why full_v6 dropped, full_v7 recovered and full_v8 dropped again, as a table (datasets, logs, evals 17-23)."""
    f = changes_frame()
    return fade((f for _ in range(int(dur * FPS))), 10, 10)


def end_card(lines, accent, dur=5.0):
    return card([(l, s, w, c) for l, s, w, c in lines], dur, accent)


# ------------------------------------------------------------------ videos
def v_agents(w, story=False):
    a = COL["execute"]
    w.add(card([("1  ·  THE AGENTS WRITE THE ROBOT CODE", 30, "B", a),
                ("“Fine-tune π0.5 to pick up the toy bee and put it into the bowl, with at least 70 % success.”", 46, "M", TXT),
                ("No dataset, no demos to start from. So two Claude Code agents (Claude Opus 5.5) write the programs themselves, "
                 "run them on the arm, read the logs, squint at the camera images and fix whatever breaks.", 32, "R", DIM)], 6.5, a, first=True))
    w.add(fade(code_segment(9.0, "The code the agents wrote", a, "execute.py: find the bee, plan, grab, check  ·  reset.py: undo it all and shuffle the table")))
    w.add(fade(dash_segment(FIRST_EXEC, 6, "Agent 1 · first robot run", "4 minutes after the start", "Agent 1's program, first time on the real robot. Bee in bowl. First try.", a, note=agent_note(FIRST_EXEC), badge=("ok", "✓  first try!"))))
    w.add(fade(dash_segment(RESET_FAIL, 14, "Agent 2 · first reset run", "14 minutes after the start", "Agent 2's first reset run crashes. It reads the log, squints at the camera images and patches reset.py.", COL["reset"], note=agent_note(RESET_FAIL), badge=("fail", "✗  crashed"))))
    w.add(fade(dash_segment(RESET_OK, 9, "Agent 2 · second reset run", "18 minutes after the start", "Second try: it works. Over the week Agent 2 patched reset.py 62 times, one surprise at a time.", COL["reset"], note=agent_note(RESET_OK), badge=("ok", "✓  it works"))))
    w.add(end_card([("Agent 1: nailed it on the first try, 6 minutes in", 44, "B", COL["execute"]),
                    ("Agent 2: worked on the second try, 20 minutes in", 44, "B", COL["reset"]),
                    ("878 + 1,915 lines of robot code and 159 commits, all written (and fixed) by the agents", 32, "R", DIM)], a))


def v_resets(w):
    a = COL["reset"]
    w.add(card([("2  ·  RESETS", 30, "B", a), ("The robot cleans up after itself.", 64, "B", TXT),
                ("Before every episode it fishes the bee out of the bowl and moves both to new random spots (the red ×). "
                 "Nobody's favorite job, so the robot does it. No reset, no data.", 32, "R", DIM)], 6.0, a, first=True, bee=230))
    w.add(fade(dash_segment(RESET_HERO, 8, "A reset during data collection", "bowl to a new random spot, then the bee (red × = targets)", "It carries the bowl (bee still inside) 18 cm to a new spot, then fishes the bee out and puts it where the bowl used to be.", a, badge=("ok", "✓  table reset"))))
    w.add(fade(grid_segment(RESET_GRID, 3, 2, 12, "Six resets, six random layouts", None, "Watch the bowls move: new random spots every time, and the side camera double-checks the result.", a)))
    w.add(end_card([("678 resets", 96, "B", a), ("all done by the robot", 56, "B", TXT),
                    ("544 were perfect on the first try. The rest got retried, or the agent fixed its program.", 30, "R", DIM),
                    ("When the bee escaped (off the table, stuck to a finger), a human came to the rescue. 9 times. "
                     "More in video 08.", 30, "B", COL["human"])], a, 6.0))


def v_collection(w):
    a = COL["collect"]
    w.add(card([("3  ·  THE ROBOT MAKES ITS OWN DATA", 30, "B", a), ("Zero teleop. It records its own demos.", 60, "B", TXT),
                ("Reset, do the task, record both cameras. Two separate checks look at the result, depth geometry and Claude "
                 "Opus 5.5 looking at the before/after images, and only clear wins are kept.", 32, "R", DIM)], 6.0, a, first=True, bee=230))
    w.add(fade(dash_segment(COLLECT_HERO, 7, "One demo", "perceive → plan → grasp → carry → release → verify", "Everything gets logged: the planned path (magenta), what it sees, and every decision it makes.", a, badge=("ok", "✓  keeper"))))
    w.add(fade(tape_segment(5.5, "Spot the yellow bees in the next grid: those are from before the tape.")))
    w.add(fade(grid_segment(COLLECT_GRID, 6, 4, 12, "24 of the 275 keepers", None, "recorded over three rounds, day and night", a,
                            counter=lambda s: "289 tries  ·  275 keepers (95 %)  ·  678 resets")))
    w.add(fade(wall_segment(11.0, "Every robot run, in order", "1,422 runs · 76 hours of robot time · 6 days", a)))
    w.add(end_card([("1,422 robot runs", 90, "B", a), ("289 demo runs · 678 robot resets · 267 policy tests · 117 DAgger fixes", 40, "B", TXT),
                    ("76 hours of robot time, all of it on camera. A human came to the rescue 9 times.", 32, "R", DIM)], a))


def v_training(w):
    a = COL["train"]
    w.add(card([("4  ·  TRAINING π0.5", 30, "B", a), ("Then the agents train π0.5 on it.", 60, "B", TXT),
                ("They turn the recordings into datasets, pick the training recipe, fully fine-tune π0.5 (7-8 hours a run on "
                 "one GPU) and test every model on the real robot.", 32, "R", DIM)], 6.0, a, first=True, bee=230))
    w.add(fade(chart_segment(13.0, "Watching the loss go down", a, "three full fine-tunes · smoothed · log scale", "loss")))
    w.add(fade(chart_segment(11.0, "Report card: every model on the real robot", a, "all its test runs (for full_v5, the two tests with the final 240 s limit)", "success")))
    w.add(dips_card())
    w.add(end_card([("6 full fine-tunes  +  4 LoRA runs", 60, "B", a),
                    ("Every time it failed: figure out why, fix the data or the recipe, train again.", 34, "R", TXT),
                    ("RTX 4090 and a power-capped RTX 5090 · 7-8 hours per full fine-tune", 30, "R", DIM)], a))


def v_eval_dagger(w):
    a = COL["dagger"]
    w.add(card([("5  ·  TESTING AND DAgger", 30, "B", a), ("Most of the work was watching it fail.", 60, "B", TXT),
                ("Every model gets 20 tries on the real robot. When it fails, the scripted expert goes back to where the policy "
                 "got stuck and finishes the job. That rescue becomes new training data (DAgger).", 32, "R", DIM)], 6.0, a, first=True, bee=230))
    w.add(fade(dash_segment(V1_FAIL, 3, "First full fine-tune: 0 / 20", "full_v1 · Nov 8", "It heads straight for... the curtain. The safety check blocks the move before it leaves the work area.", COL["fail"], badge=("fail", "0 / 20"))))
    w.add(card([("The problem was in the data.", 44, "B", TXT),
                ("The demos mixed two different arm poses, had long pauses, and started with a scripted camera move.", 34, "R", DIM),
                ("So: start the policy above the bowl, clean up the data, train again.  0 % → 63 %", 40, "B", COL["policy"])], 5.5, a))
    w.add(fade(dash_segment(DAG_FAIL, 16, "So close", "full_v5 (63 %), Nov 9", "It gets to the bee, but keeps missing the grab, and time runs out.", COL["fail"], badge=("fail", "✗  out of time"), note=POLICY_NOTE)))
    w.add(fade(dash_segment(DAG_FIX, 8, "DAgger to the rescue", "three minutes later, same scene", "The expert program goes back to where the policy got stuck and finishes the job. This exact clip went into full_v7's training data.", a, badge=("ok", "✓  fixed it"))))
    w.add(fade(chart_segment(10.0, "What the fixes did", a, "success rate in real-robot test runs", "success")))
    w.add(dips_card())
    w.add(end_card([("267 policy test runs · 117 DAgger fixes", 52, "B", a), ("23 rounds of testing on the real robot", 40, "B", TXT),
                    ("85 of the fixes worked. The 8 from full_v5's failures went into full_v7's training data.", 30, "R", DIM)], a))


def v_final(w):
    a = COL["policy"]
    w.add(card([("6  ·  THE MOMENT OF TRUTH", 30, "B", a), ("16 / 20  =  80 %", 96, "B", a),
                ("π0.5, fine-tuned on 113 episodes the robot recorded itself (demos plus DAgger fixes). The goal was 70 %.", 34, "R", TXT),
                (POLICY_NOTE, 28, "R", DIM)], 6.5, a, first=True, bee=200))
    w.add(fade(grid_segment(FINAL20, 5, 4, 12, "All 20 test runs of the final policy", None, "full_v7 · test round 22 · Nov 10", a, badges=True)))
    w.add(fade(dash_segment(HERO, 5, "One full run, start to finish", "full_v7 · test round 22", "From above the bowl: find the bee, grab it, carry it over, let go. Nailed it.", a, badge=("ok", "✓  nailed it"), note=[POLICY_NOTE, END_NOTE])))
    w.add(end_card([("80 % on the real robot", 80, "B", a), ("Data, resets, training, testing and retraining, all run by AI agents", 40, "B", TXT),
                    ("6 days from one sentence to the final policy. A human came to the rescue 9 times.", 32, "R", DIM)], a))


def v_story(w):
    """The whole process, ~2:15."""
    w.add(card([("WE GAVE SOME AI AGENTS A ROBOT AND ONE SENTENCE", 54, "B", TXT),
                ("“Fine-tune π0.5 to pick up the toy bee and put it into the bowl, with at least 70 % success.”", 44, "M", COL["target"]),
                ("No dataset, no teleop. Claude Code agents running Claude Opus 5.5 wrote the robot code, collected their own data, "
                 "reset the table, trained π0.5, tested it, fixed its mistakes with DAgger and trained it again. 6 days later: 80 %.", 32, "R", DIM)],
               6.0, TXT, first=True, bee=170))
    w.add(chapter_card("1 / 6", "Write your own robot code", "no demos, no dataset, no problem", COL["execute"]))
    w.add(fade(code_segment(3.0, "Agents, writing robot code", COL["execute"], "878 + 1,915 lines · 159 commits")))
    w.add(fade(dash_segment(FIRST_EXEC, 10, "Agent 1 · first robot run", "4 minutes after the start", "Agent 1's program, first time on the real robot. Bee in bowl. First try.", COL["execute"], story=True, badge=("ok", "✓  first try!"))))
    w.add(chapter_card("2 / 6", "Clean up after yourself", "a new random layout before every single episode", COL["reset"]))
    w.add(fade(dash_segment(RESET_HERO, 18, "Agent 2's reset program", "bowl and bee to new random spots (red ×)", "Carry the bowl to a new spot, fish the bee out, set it down on the table. 678 times, day and night.", COL["reset"], story=True)))
    w.add(chapter_card("3 / 6", "Make your own data", "zero teleop", COL["collect"]))
    w.add(fade(grid_segment(COLLECT_GRID, 6, 4, 22, "275 demos, zero teleop", None, "day and night, over three rounds", COL["collect"],
                            counter=lambda s: "289 tries  ·  275 keepers (95 %)  ·  678 resets")))
    w.add(fade(wall_segment(5.0, "Every robot run, in order", "1,422 runs · 76 hours of robot time", COL["collect"])))
    w.add(card([("MEANWHILE, THE BEE HAD OTHER PLANS", 34, "B", COL["human"]), ("9 times a human had to come to the rescue", 60, "B", TXT),
                ("all in the log, and most led to a new safety check", 34, "R", DIM)], 2.8, COL["human"]))
    w.add(incident_frames(INCIDENTS[3], speed_scale=2.6, story=True, short=True))
    w.add(incident_frames(INCIDENTS[7], speed_scale=2.1, story=True, short=True))
    w.add(fade(tape_segment(3.6)))
    w.add(chapter_card("4 / 6", "Train π0.5", "7-8 hours per full fine-tune", COL["train"]))
    w.add(fade(chart_segment(5.0, "Watching the loss go down", COL["train"], "three full fine-tunes, 30k steps each", "loss")))
    w.add(chapter_card("5 / 6", "Now watch it fail (a lot)", "20 real tries per model, then DAgger", COL["dagger"]))
    w.add(fade(dash_segment(V1_FAIL, 4, "First full fine-tune: 0 / 20", "Nov 8", "It heads straight for... the curtain. The safety check says no.", COL["fail"], story=True, badge=("fail", "0 / 20"))))
    w.add(fade(dash_segment(DAG_FAIL, 45, "So close", "full_v5", "It gets to the bee, keeps missing the grab, and runs out of time…", COL["fail"], story=True)))
    w.add(fade(dash_segment(DAG_FIX, 18, "DAgger to the rescue", "the expert picks up where the policy got stuck", "…so the expert program takes over and finishes the job. That clip goes straight into the training data.", COL["dagger"], story=True, badge=("ok", "✓  fixed it"))))
    w.add(fade(chart_segment(8.5, "Retrain until it works (not in a straight line)", COL["policy"], "success rate in real-robot test runs", "success")))
    w.add(chapter_card("6 / 6", "The moment of truth", "the final policy, 20 tries", COL["policy"]))
    w.add(fade(dash_segment(HERO, 14, "The final policy", "full_v7 · Nov 10", "Find the bee. Grab it. Carry it. Let go. Nailed it.", COL["policy"], t1=119.0, story=True, badge=("ok", "✓  nailed it"))))
    w.add(fade(grid_segment(FINAL20, 5, 4, 30, "All 20 test runs of the final policy", None, "goal: 70 %", COL["policy"], badges=True)))
    w.add(end_card([("80 % on the real robot", 76, "B", COL["policy"]),
                    ("6 days  ·  1,422 robot runs  ·  678 robot resets", 40, "B", TXT),
                    ("289 demo runs  ·  117 DAgger fixes  ·  10 training runs", 34, "B", TXT),
                    ("A human came to the rescue 9 times, mostly for the bee.", 34, "B", COL["human"]),
                    ("People set the goal, signed off on the big decisions and kept an eye on safety.", 28, "R", DIM),
                    ("Claude Code agents on Claude Opus 5.5  ·  π0.5 (openpi)  ·  Kinova Gen3", 28, "R", DIM)], COL["policy"], 6.0))


def v_highlights(w):
    a = COL["policy"]
    w.add(card([("THE FINAL POLICY  ·  4 RUNS", 30, "B", a), ("80 % success on the real robot", 64, "B", TXT),
                (POLICY_NOTE, 28, "R", DIM)], 4.5, a, first=True, bee=260))
    for i, rc in enumerate(HIGHLIGHTS):
        w.add(fade(dash_segment(rc, 8, f"Run {i + 1} of 4", f"full_v7 · test round 22 · episode {rc['ep']}",
                                "Just the trained π0.5 policy: find the bee, grab it, carry it, drop it in the bowl.", a, badge=("ok", "✓  nailed it"),
                                note=[POLICY_NOTE, END_NOTE])))
    w.add(end_card([("16 of 20 test runs in the bowl", 64, "B", a), ("all 20 are on video, here are 4", 34, "R", DIM)], a, 4.0))


def v_human(w):
    a = COL["human"]
    w.add(card([("08  ·  BLOOPERS", 30, "B", a), ("The bee had other plans", 72, "B", TXT),
                ("9 times a human had to come to the rescue. They're all in the operator log, and most ended with a new "
                 "safety check so it wouldn't happen again.", 34, "R", DIM)], 6.5, a, first=True, bee=250))
    for inc in INCIDENTS:
        if inc["recs"]:
            w.add(incident_frames(inc))
        else:
            w.add(incident_card(inc))
        if inc["n"] == 5:
            w.add(fade(tape_segment(6.0)))
    w.add(end_card([("9 rescues in 6.5 days", 72, "B", a),
                    ("The rest of the time it ran on its own: 1,422 robot runs, 678 resets, 23 rounds of testing.", 38, "B", TXT),
                    ("People also signed off on the big decisions, and renewed the agents' login once when it expired.", 30, "R", DIM)], a, 6.0))


VIDEOS = {"story": ("00_full_story.mp4", v_story), "agents": ("01_agents_write_programs.mp4", v_agents),
          "resets": ("02_autonomous_resets.mp4", v_resets), "collection": ("03_data_collection.mp4", v_collection),
          "training": ("04_training.mp4", v_training), "eval_dagger": ("05_evaluation_and_dagger.mp4", v_eval_dagger),
          "final": ("06_final_policy_80_percent.mp4", v_final), "highlights": ("07_final_policy_rollouts.mp4", v_highlights),
          "human": ("08_human_help.mp4", v_human)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(VIDEOS))
    ap.add_argument("--music-only", action="store_true", help="only (re)place the music under the rendered videos")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.music_only:
        for name in a.names:
            showcase_music.mux(OUT / VIDEOS[name][0], name)
            print(f"{VIDEOS[name][0]}: music", flush=True)
        return
    if {"story", "collection", "human"} & set(a.names):
        make_thumbs()
    for name in a.names:
        fname, fn = VIDEOS[name]
        t = time.time()
        w = Writer(OUT / fname)
        fn(w)
        secs = w.close()
        showcase_music.mux(OUT / fname, name)
        print(f"{fname}: {secs:.1f} s of video, rendered in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
