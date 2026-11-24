"""Showcase videos of run 20261123_1616_drawer (real-world RL: put the drawer back into a cabinet lying on its back) for X.

Every clip is a real recording (dashboard.mp4: side + wrist camera with the program's overlays and log) or the sim's own
renders; every number is computed here from the run's records (events.jsonl, rl_curve.jsonl, episode summaries,
sim/drawer_insert/results/lying), so a re-render after the run picks up the final numbers. Human help is taken from
OPERATOR_LOG.md (hand-written below, with times). 1920x1080, 30 fps, H.264 + AAC music, each video <= 2:20.
While the final evaluation is missing, every frame carries a DRAFT tag and the posts say [TBD].

    nice -n 15 taskset -c 24-31 third_party/openpi/.venv/bin/python scripts/make_drawer_videos.py [story sim curve human eval] [--site-only]
"""
import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageSequence  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import showcase_music  # noqa: E402
from showcase_common import (BG, DIM, FPS, H, PANEL, TXT, W, Video, Writer, balanced, blank, fade, fit, font,  # noqa: E402
                             mono, split_dash, text, wrap)

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20261123_1616_drawer"
SETUP = ROOT / "runs" / "drawers_setup"
SIM = ROOT / "sim" / "drawer_insert"
SIM_GIF = Path("/data/sonic/drawer_insert_runs/lying_episodes.gif")
OUT = RUN / "videos"
SITE = Path("/data/sonic/sim_replay_site")
COL = dict(calib=(80, 200, 200), reset=(255, 166, 87), rl=(88, 166, 255), learned=(88, 166, 255), scripted=(255, 166, 87),
           sim=(190, 140, 255), ok=(63, 200, 100), fail=(255, 90, 90), unsure=(255, 210, 80), human=(255, 112, 67),
           eval=(63, 200, 100), pre=(130, 140, 152))
THREADS = 2                     # libx264 threads per video


# ------------------------------------------------------------------ run data
def T(s):
    return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()] if Path(p).exists() else []


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


EVENTS = jl(RUN / "events.jsonl")
T0 = T(EVENTS[0]["t"])                                     # 2026-11-23 16:16:37, run created
RESETS = [dict(t=T(e["t"]) - T0, ok=bool(e.get("ok")), n=e["reset"]) for e in EVENTS if e["kind"] == "reset"]
NOTES = [T(e["t"]) - T0 for e in EVENTS if e["kind"] == "operator_note"]
RESTARTS = [T(e["t"]) - T0 for e in EVENTS if e["kind"] == "restart"]
ARM = {None: "pre", "learned": "learned", "scripted": "scripted", "sim": "sim", "learned_eval": "learned_eval"}


def fixed_return(final, ins):
    """scripts/rl_plots.py fixed_return: +10 success, -1 per force stop, -0.1 per N of peak above 15 N, -0.05 per step."""
    return (10.0 * (final == "success") - 1.0 * (ins.get("contacts") or 0)
            - 0.1 * max(0.0, (ins.get("peak") or 0) - 15.0) - 0.05 * (ins.get("steps") or 0))


def load_episodes():
    """Every RL episode (training 'rl' and evaluation 'rl_eval') with its verdict, arm and insertion numbers."""
    eps = []
    for e in EVENTS:
        if e["kind"] != "episode":
            continue
        i = int(e["episode"])
        s = jload(RUN / f"episodes/ep_{i:06d}/summary.json", {})
        ins = s.get("insert") or {}
        cm = ins.get("contact_map")
        if ins.get("contacts") is None:
            rim = None
        else:                       # rim touch = force stop with the drawer's bottom above 80 mm (docs/report/make_tables.py)
            rim = int(ins["contacts"]) if cm is None else sum(c[3] > 80.0 for c in cm)
        eps.append(dict(ep=i, mode=e.get("mode"), final=e.get("final"), t=T(e["t"]) - T0, dur=e.get("duration") or 0,
                        dir=Path(e["attempt"]), arm=ARM.get(s.get("arm"), s.get("arm")), rim=rim, steps=ins.get("steps"),
                        fixed=fixed_return(e.get("final"), ins)))
    return eps


EPS = load_episodes()
TRAIN = [e for e in EPS if e["mode"] == "rl"]
EVAL_EPS = [e for e in EPS if e["mode"] == "rl_eval"]
EVAL = next((r for r in jl(RUN / "rl_curve.jsonl") if r.get("kind") == "eval"), None)
WIN = 17                         # repaired learner and one reward for every arm from episode 17 (report, Section 5.6)


def stats():
    """The numbers the videos and posts quote."""
    s = dict(n_train=len(TRAIN), ok=sum(e["final"] == "success" for e in TRAIN))
    best, cur = (0, None), (0, None)                    # longest run of counted successes
    for e in TRAIN:
        cur = (cur[0] + 1, cur[1] if cur[0] else e["ep"]) if e["final"] == "success" else (0, None)
        best = max(best, cur, key=lambda r: r[0])
    s["streak"], s["streak_from"] = best
    since = [e for e in TRAIN if e["ep"] >= 9]
    s["since9"] = (sum(e["final"] == "success" for e in since), len(since))
    for arm in ("learned", "scripted", "sim"):
        g = [e for e in TRAIN if e["arm"] == arm and e["ep"] >= WIN and e["rim"] is not None]
        s[arm] = dict(n=len(g), ok=sum(e["final"] == "success" for e in g), clean=sum(e["rim"] == 0 for e in g),
                      clean_eps=[e["ep"] for e in g if e["rim"] == 0], eps=[e["ep"] for e in g],
                      rim=np.mean([e["rim"] for e in g]) if g else 0, steps=np.mean([e["steps"] for e in g]) if g else 0,
                      fixed=np.mean([e["fixed"] for e in g]) if g else 0)
    s["sim_eps"] = [e["ep"] for e in TRAIN if e["arm"] == "sim"]
    s["sim_ok"] = sum(e["final"] == "success" for e in TRAIN if e["arm"] == "sim")
    s["resets"], s["resets_ok"] = len(RESETS), sum(r["ok"] for r in RESETS)
    s["notes"], s["restarts"] = len(NOTES), len(RESTARTS)
    s["eval"] = None if EVAL is None else (int(EVAL["success"]), int(EVAL["n"]))
    s["fisher"] = fisher_two_sided(s["learned"]["clean"], s["learned"]["n"], s["scripted"]["clean"], s["scripted"]["n"])
    return s


def fisher_two_sided(a, n1, b, n2):
    """Fisher's exact test (two-sided) for a/n1 vs b/n2."""
    if not n1 or not n2:
        return None
    k, n = a + b, n1 + n2
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9))


S = stats()
DRAFT = S["eval"] is None


def clock(t):
    """Header clock: (time, date) at run time t (seconds since T0)."""
    lt = time.localtime(T0 + t)
    return time.strftime("%H:%M", lt), time.strftime("%a %b %d", lt)


def hm(t):
    return time.strftime("%H:%M", time.localtime(T0 + t))


def at(hhmm, day="2026-11-23"):
    return T(f"{day}T{hhmm}:00") - T0


def rec(path):
    """A recording: its directory, start (s since T0) and length."""
    d = Path(path)
    if not d.is_absolute():
        d = RUN / "agents" / d
    a = jload(d / "attempt.json", {})
    return dict(dir=d, t=float(a.get("started", T0)) - T0, what=str(a.get("what") or ""))


def ep_rec(i):
    return rec(next(e["dir"] for e in EPS if e["ep"] == i))


# ------------------------------------------------------------------ mascot: a small cartoon drawer
@lru_cache(None)
def drawer_sprite(h, wink=False):
    """The mascot (RGBA), h pixels tall: a wooden drawer with a brass label frame for a mouth and the knob for a nose."""
    w0, h0 = 300, 250
    im = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (34, 24, 18, 255)
    d.polygon([(34, 78), (92, 30), (276, 30), (218, 78)], fill=(222, 150, 92, 255), outline=ink)        # top
    d.polygon([(218, 78), (276, 30), (276, 176), (218, 224)], fill=(140, 72, 34, 255), outline=ink)    # side
    d.rounded_rectangle([34, 78, 218, 224], 10, fill=(188, 104, 52, 255), outline=ink, width=6)        # front
    d.line([(34, 78), (92, 30), (276, 30), (276, 176), (218, 224)], fill=ink, width=6)
    d.line([(218, 78), (276, 30)], fill=ink, width=5)
    for x in (60, 120, 180):                                                                         # wood grain
        d.arc([x - 30, 150, x + 30, 214], 200, 330, fill=(160, 86, 40, 255), width=3)
    for cx in (88, 164):                                                                             # eyes
        if wink and cx == 164:
            d.arc([cx - 18, 100, cx + 18, 130], 200, 340, fill=ink, width=6)
            continue
        d.ellipse([cx - 19, 94, cx + 19, 136], fill=(255, 255, 255, 255), outline=ink, width=5)
        d.ellipse([cx - 7, 108, cx + 9, 128], fill=ink)
        d.ellipse([cx - 2, 111, cx + 4, 117], fill=(255, 255, 255, 255))
    d.ellipse([52, 140, 76, 156], fill=(235, 120, 100, 160))                                         # cheeks
    d.ellipse([176, 140, 200, 156], fill=(235, 120, 100, 160))
    d.rectangle([92, 150, 160, 186], fill=(120, 60, 28, 255), outline=(196, 196, 204, 255), width=7)  # label frame
    d.arc([104, 156, 148, 182], 15, 165, fill=(250, 236, 220, 255), width=5)                         # smile
    d.ellipse([114, 190, 138, 214], fill=(205, 205, 212, 255), outline=ink, width=4)                 # knob
    d.ellipse([120, 194, 128, 202], fill=(255, 255, 255, 255))
    return im.resize((int(w0 * h / h0), h), Image.LANCZOS)


def paste_drawer(pil, cx, cy, h, angle=0.0, squash=1.0, wink=False):
    spr = drawer_sprite(h, wink)
    if squash != 1.0:
        spr = spr.resize((int(spr.width / squash), max(4, int(spr.height * squash))), Image.LANCZOS)
    if angle:
        spr = spr.rotate(angle, resample=Image.BICUBIC, expand=True)
    pil.paste(spr, (int(cx - spr.width / 2), int(cy - spr.height)), spr)


def hop_drawer(pil, k, n, y0=300, h=96, hops=4):
    """The mascot hops across the frame, leaving little dust dots where it lands."""
    def pos(kk):
        u = kk / max(1, n - 1)
        ph = (u * hops) % 1.0
        return -170 + (W + 340) * u, y0 - 70 * math.sin(math.pi * ph), ph
    d = ImageDraw.Draw(pil)
    for j in range(1, hops * 2):
        uj = j / (hops * 2) if j % 2 == 0 else None
        if uj is None:
            continue
        kk = uj * (n - 1)
        if kk < k:
            x = -170 + (W + 340) * uj
            for dx in (-34, -18, 18, 34):
                d.ellipse([x + dx - 4, y0 + 4, x + dx + 4, y0 + 12], fill=(70, 78, 90))
    x, y, ph = pos(k)
    sq = 0.82 if ph < 0.08 or ph > 0.92 else 1.0
    paste_drawer(pil, x, y, h, angle=-10 * math.cos(math.pi * ph), squash=sq, wink=0.4 < ph < 0.6)


# ------------------------------------------------------------------ layouts
def header(d, chapter, accent, sub=None, t=None, speed=None):
    d.rectangle([0, 0, W, 124], fill=PANEL)
    d.rectangle([0, 0, 10, 124], fill=accent)
    text(d, (40, 14), chapter, 54, "B", fill=TXT)
    if sub:
        text(d, (40, 80), sub, 30, "R", fill=DIM)
    if t is not None:
        hhmm, day = clock(t)
        text(d, (W - 40, 12), hhmm, 54, "B", fill=accent, anchor="ra")
        text(d, (W - 40, 80), day, 30, "R", fill=DIM, anchor="ra")
    if speed:
        x = W - 230 if t is not None else W - 40
        d.rounded_rectangle([x - 180, 36, x - 10, 92], 14, fill=accent)
        text(d, (x - 95, 64), f"{speed:g}x speed", 30, "B", fill=BG, anchor="mm")


def counters(d, t, y=1018, accent=TXT):
    items = [("training episodes", sum(e["t"] <= t for e in TRAIN), accent),
             ("drawer back in", sum(e["t"] <= t and e["final"] == "success" for e in EPS), COL["ok"]),
             ("robot resets", sum(r["t"] <= t for r in RESETS), accent),
             ("notes from us", sum(x <= t for x in NOTES), COL["human"]),
             ("hands on the scene", int(t >= at("16:45")), COL["human"])]
    d.rectangle([0, y - 14, W, H], fill=PANEL)
    for i, (k, v, col) in enumerate(items):
        x = 40 + i * 376
        text(d, (x, y), f"{v:,}", 40, "B", fill=col)
        text(d, (x + font(40, "B").getlength(f"{v:,}") + 12, y + 10), k, 28, "R", fill=DIM)


def timeline_bar(d, t, pil=None, y=990):
    """The run from 16:16 to its last episode: hour ticks, a green tick per counted success, orange per human note."""
    total = max(e["t"] for e in EPS) + 600
    x0, x1 = 40, W - 40
    X = lambda tt: x0 + (x1 - x0) * min(max(tt, 0), total) / total
    d.rectangle([x0, y, x1, y + 6], fill=(45, 52, 62))
    d.rectangle([x0, y, X(t), y + 6], fill=COL["rl"])
    for hh in range(17, 28):
        xx = X(at(f"{hh % 24:02d}:00", "2026-11-24" if hh >= 24 else "2026-11-23"))
        if xx < x1:
            d.line([xx, y - 6, xx, y + 12], fill=(80, 88, 100), width=2)
    for e in EPS:
        if e["t"] <= t and e["final"] == "success":
            d.line([X(e["t"]), y - 2, X(e["t"]), y + 8], fill=COL["ok"], width=3)
    for tn in NOTES + [at("16:39"), at("16:42"), at("16:45")]:
        if tn <= t:
            d.rectangle([X(tn) - 3, y - 9, X(tn) + 3, y + 15], fill=COL["human"])
    if pil is not None:
        paste_drawer(pil, X(t), y + 4, 30)


def badge_at(d, f, cx, cy, kind, label, w=290):
    """A badge that pops in (f frames since it appeared)."""
    sc = 0.5 + 0.7 * f / 5 if f < 5 else 1.2 - 0.2 * min(1.0, (f - 5) / 5)
    col = {"ok": COL["ok"], "fail": COL["fail"], "human": COL["human"], "unsure": COL["unsure"], "sim": COL["sim"]}[kind]
    d.rounded_rectangle([cx - w * sc, cy - 74 * sc, cx + w * sc, cy + 74 * sc], int(24 * sc), fill=col)
    text(d, (cx, cy), label, max(12, int(50 * sc)), "B", fill=BG, anchor="mm")


def crop(img, box):
    """box (x0, y0, x1, y1) in the 640x360 camera image."""
    if not box:
        return img
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


SIDE_ZOOM = (110, 70, 510, 295)       # the cabinet and the drawer's table spot in the side camera (16:9)


def dash_segment(rc, speed, chapter, sub, caption, accent, t0=0.0, t1=None, note=None, story=False, badge=None, zoom=None,
                 badge_at_s=None):
    """One recording, full layout: header with clock, side + wrist cameras, the program's log, caption."""
    v = Video(rc["dir"] / "dashboard.mp4")
    t0 = max(t0, 0.25)
    span = (min(t1, v.dur) if t1 else v.dur) - t0
    k_badge = None
    for k, fr in enumerate(v.frames(speed, t0, t1)):
        side, wrist, log = split_dash(fr)
        img = blank()
        img[130:670, 0:960] = fit(crop(side, zoom), 960, 540)
        img[130:670, 960:1920] = fit(wrist, 960, 540)
        img[676:856, 0:1920] = fit(log, 1920, 180)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        tt = k * speed / FPS
        t_run = rc["t"] + t0 + tt
        header(d, chapter, accent, sub, t_run, speed)
        text(d, (16, 140), "side camera" + (" (zoomed in)" if zoom else ""), 26, "M", fill=TXT, shadow=True)
        text(d, (976, 140), "wrist camera", 26, "M", fill=TXT, shadow=True)
        if badge and tt > (span * 0.8 if badge_at_s is None else badge_at_s):
            k_badge = k if k_badge is None else k_badge
            badge_at(d, k - k_badge, 1610, 766, badge[0], badge[1])
        y = 864
        cap = wrap(caption, 44, W - 80, "M")[:2]
        for line in cap:
            text(d, (40, y), line, 44, "M", fill=TXT)
            y += 54
        if note and not story:
            nl = wrap(note, 30, W - 80)
            assert len(nl) <= (3 if len(cap) == 1 else 1), (caption, note)
            for line in nl:
                text(d, (40, y + 6), line, 30, "R", fill=DIM)
                y += 38
        if story:
            counters(d, t_run, accent=accent)
            timeline_bar(d, t_run, pil)
        yield np.asarray(pil)


def grid_segment(items, cols, rows, speed, chapter, sub, head, caption, accent, zoom=SIDE_ZOOM, dur=None, label=None):
    """Several recordings at once (side camera), each tile freezing on its last frame with a pass/fail badge.
    items: (rec, t0, t1, final, tag) per tile."""
    tw, th = W // cols, int((W // cols) * 9 / 16)
    gy = 130
    gens = [Video(r["dir"] / "dashboard.mp4").frames(speed, t0, t1) for r, t0, t1, *_ in items]
    last, done, done_k = [None] * len(items), [False] * len(items), [None] * len(items)
    n_frames = int(dur * FPS) if dur else int(max(t1 - t0 for _, t0, t1, *_ in items) / speed * FPS) + 45
    for k in range(n_frames):
        img = blank()
        for i, g in enumerate(gens):
            if not done[i]:
                try:
                    last[i] = next(g)
                except StopIteration:
                    done[i], done_k[i] = True, k
            if last[i] is None:
                continue
            r, c = divmod(i, cols)
            x, y = c * tw, gy + r * th
            img[y:y + th - 4, x:x + tw - 4] = fit(crop(split_dash(last[i])[0], zoom), tw - 4, th - 4)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub, None, speed)
        n_ok = 0
        for i, (rc, t0, t1, final, tag) in enumerate(items):
            r, c = divmod(i, cols)
            x, y = c * tw, gy + r * th
            if tag:
                tcol = COL.get(tag[1], TXT)
                text(d, (x + 14, y + 10), tag[0], 26, "B", fill=tcol, shadow=True)
            if done[i]:
                ok = final == "success"
                n_ok += ok
                f = k - done_k[i]
                sc = 0.4 + 0.9 * f / 5 if f < 5 else 1.3 - 0.3 * min(1.0, (f - 5) / 5)
                cx, cy = x + tw - 42, y + 38
                d.rounded_rectangle([cx - 28 * sc, cy - 28 * sc, cx + 28 * sc, cy + 28 * sc], int(12 * sc), fill=verdict_col(final))
                text(d, (cx, cy), "✓" if ok else ("?" if final == "unsure" else "✗"), max(10, int(40 * sc)), "B", fill=BG, anchor="mm")
        y_cap = gy + rows * th + 14
        hd = head(n_ok, sum(done)) if callable(head) else head
        text(d, (40, y_cap), hd, 44, "B", fill=accent)
        for j, line in enumerate(wrap(caption, 32, W - 80)[:2]):
            text(d, (40, y_cap + 60 + j * 42), line, 32, "R", fill=DIM)
        yield np.asarray(pil)


def card(lines, dur, accent=TXT, first=False, mascot=None):
    """Title card: lines of (text, size, weight, color), centred; text slides up into place. first: no fade-in (X shows
    the first frame before playing). mascot: y of the hopping drawer."""
    rows = []
    for s, size, weight, col in lines:
        size = size if size >= 40 else size + 6
        rows += [(line, size, weight, col) for line in balanced(s, size, W - 240, weight)]
    n = int(dur * FPS)

    def render(k):
        pil = Image.fromarray(blank())
        d = ImageDraw.Draw(pil)
        off = int(46 * (1 - min(1.0, k / 10)) ** 2)
        y = (H - sum(size + 24 for _, size, _, _ in rows)) // 2 + (40 if mascot else 0)
        for line, size, weight, col in rows:
            text(d, (W // 2, y + size // 2 + off), line, size, weight, fill=col, anchor="mm")
            y += size + 24
        d.rectangle([W // 2 - 80, H - 120, W // 2 + 80, H - 114], fill=accent)
        if mascot is not None:
            hop_drawer(pil, k, n, y0=mascot)
        return np.asarray(pil)

    def frames():
        still = None
        for k in range(n):
            if k <= 10 or mascot is not None:
                yield render(k)
            else:
                still = render(10) if still is None else still
                yield still
    return fade(frames(), 0 if first else 10, 10)


def chapter_card(num, title, sub, accent, dur=2.0):
    return card([(f"{num}", 34, "B", accent), (title, 72, "B", TXT), (sub, 34, "R", DIM)], dur, accent, mascot=330)


def still(frame, dur, n_in=10, n_out=10):
    return fade((frame for _ in range(int(dur * FPS))), n_in, n_out)


def code_segment(dur, chapter, accent, sub):
    """The agents' two main programs, scrolling side by side."""
    files = [(RUN / "agents/reset/programs/reset.py", "Reset agent  ·  reset.py", COL["reset"]),
             (RUN / "agents/execute/programs/execute.py", "RL agent  ·  execute.py", COL["rl"])]
    srcs = [p.read_text().splitlines() for p, *_ in files]
    n, lh, rows = int(dur * FPS), 24, 33
    for k in range(n):
        pil = Image.fromarray(blank())
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub)
        for c, ((p, title, col), lines) in enumerate(zip(files, srcs)):
            x = 40 + c * 940
            d.rounded_rectangle([x, 136, x + 900, 1050], 16, fill=PANEL)
            text(d, (x + 24, 152), title, 34, "B", fill=col)
            text(d, (x + 876, 160), f"{len(lines):,} lines", 26, "R", fill=DIM, anchor="ra")
            start = int((len(lines) - rows) * k / n)
            for j in range(rows):
                i = start + j
                if i >= len(lines):
                    break
                d.text((x + 24, 214 + j * lh), f"{i + 1:5d}  {lines[i][:66]}", font=mono(20),
                       fill=col if lines[i].lstrip().startswith(("def ", "class ")) else (200, 208, 218))
        yield np.asarray(pil)


# ------------------------------------------------------------------ charts (drawn from the run's records)
def arm_col(arm):
    return COL.get({"learned_eval": "learned"}.get(arm, arm), COL["pre"])


def verdict_col(final):
    return {"success": COL["ok"], "unsure": COL["unsure"]}.get(final, COL["fail"])


def runs_of(eps, pred):
    """(first ep, last ep) of each run of consecutive episodes for which pred holds."""
    out, cur = [], None
    for e in eps:
        if pred(e):
            cur = [e["ep"], e["ep"]] if cur is None else [cur[0], e["ep"]]
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    return out + ([tuple(cur)] if cur else [])


def strip_notes():
    """Labels over the episode strip, from the verdicts: (first, last, text, colour, level)."""
    notes = []
    for a, b in runs_of(TRAIN, lambda e: e["final"] == "fail"):
        rim = all((e["rim"] or 0) > 0 for e in TRAIN if a <= e["ep"] <= b)
        notes.append((a, b, "on the rim" if rim else ("pull not found" if a == b == 8 else "not in"), COL["fail"], 0))
    for j, (a, b) in enumerate(runs_of(TRAIN, lambda e: e["final"] == "unsure")):
        notes.append((a, b, "in, but our check said “unsure”" if j == 0 else "“unsure” again (our bug)", COL["unsure"], 1))
    best = max(runs_of(TRAIN, lambda e: e["final"] == "success"), key=lambda r: r[1] - r[0], default=None)
    if best and best[1] - best[0] >= 4:
        notes.append((best[0], best[1], f"{best[1] - best[0] + 1} in a row", COL["ok"], 0))
    return notes


def longest_streak():
    best = max(runs_of(TRAIN, lambda e: e["final"] == "success"), key=lambda r: r[1] - r[0], default=(0, -1))
    return best[1] - best[0] + 1, best[0]


def strip_frame(p, chapter="Every training episode", accent=COL["ok"]):
    """All training episodes as tiles (green in, red not, yellow 'unsure'), the arm under each; p in [0, 1] = reveal."""
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    header(d, chapter, accent, "green: drawer in (both checks agree)  ·  red: not in  ·  yellow: in, but our check said “unsure”")
    n = max(50, TRAIN[-1]["ep"] + 1 if TRAIN else 50)
    pitch = (W - 160) // n
    sz, th = pitch - 4, 70
    x0, y0 = (W - n * pitch) // 2, 430
    shown = p * len(TRAIN)
    for i, e in enumerate(TRAIN):
        if i >= shown:
            break
        x = x0 + e["ep"] * pitch
        f = min(1.0, (shown - i) / 2.0)
        cx, cy, hw, hh = x + sz / 2, y0 + th / 2, sz / 2 * (0.6 + 0.4 * f), th / 2 * (0.6 + 0.4 * f)
        d.rounded_rectangle([cx - hw, cy - hh, cx + hw, cy + hh], 5, fill=verdict_col(e["final"]))
        d.rectangle([x, y0 + th + 6, x + sz, y0 + th + 16], fill=arm_col(e["arm"]))
    for k in range(0, n, 5):
        text(d, (x0 + k * pitch + sz / 2, y0 + th + 28), str(k), 26, "R", fill=DIM, anchor="ma")
    text(d, (x0, y0 + th + 70), "training episode", 28, "R", fill=DIM)
    for a, b, label, col, lvl in strip_notes():
        if shown < next((i + 1 for i, e in enumerate(TRAIN) if e["ep"] == b), 1e9):
            continue
        xa, xb = x0 + a * pitch, x0 + b * pitch + sz
        yb = y0 - 22 - lvl * 64
        d.line([xa, yb, xb, yb], fill=col, width=4)
        d.line([xa, yb, xa, yb + 12], fill=col, width=4)
        d.line([xb, yb, xb, yb + 12], fill=col, width=4)
        big = b - a >= 10
        right = not big and xa + font(32, "B").getlength(label) > W - 30
        text(d, ((xa + xb) / 2 if big else (xb if right else xa), yb - 10), label, 46 if big else 32, "B", fill=col,
             anchor="mb" if big else ("rb" if right else "lb"))
    lx, ly = x0, 640
    for name, key in (("before the arms", "pre"), ("learned", "learned"), ("scripted baseline", "scripted"),
                      ("sim policy, zero-shot", "sim")):
        d.rectangle([lx, ly + 12, lx + 30, ly + 22], fill=COL[key])
        text(d, (lx + 42, ly), name, 30, "R", fill=DIM)
        lx += 90 + font(30).getlength(name)
    if p >= 1.0 and TRAIN:
        fo = next((e for e in TRAIN if e["final"] == "success"), None)
        un = [e["ep"] for e in TRAIN if e["final"] == "unsure"]
        lines = [f"{S['ok']} of {S['n_train']} training episodes counted as a success."]
        if fo:
            lines.append(f"The first one at {hm(fo['t'])}, {(fo['t'] - TRAIN[0]['t'] + TRAIN[0]['dur']) / 3600:.1f} hours "
                         f"after the first episode started. No manual resets.")
        if un:
            lines.append(f"Episodes {', '.join(map(str, un[:-1]))} and {un[-1]} put the drawer in too; bugs in our check "
                         f"called them “unsure”, and they stay that way." if len(un) > 1 else
                         f"Episode {un[0]} put the drawer in too; a bug in our check called it “unsure”.")
        for j, line in enumerate(lines):
            text(d, (x0, 760 + j * 58), line, 40 if j == 0 else 34, "B" if j == 0 else "R", fill=TXT if j == 0 else DIM)
    return pil


def strip_segment(dur, chapter="Every training episode", accent=COL["ok"]):
    n = int(dur * FPS)
    for k in range(n):
        yield np.asarray(strip_frame(min(1.0, (k + 1) / (n * 0.65)), chapter, accent))


def return_frame(p):
    """Fixed-formula return per training episode, by arm, with the rolling mean of 5 per arm (learned, scripted)."""
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    header(d, "Return per episode", COL["rl"], "same formula for every episode: +10 drawer in, -1 per force stop, -0.1 per N above 15 N, -0.05 per step")
    x0, y0, x1, y1 = 180, 190, W - 120, 860
    n = max(50, len(TRAIN))
    lo, hi = -10.0, 10.0
    X = lambda ep: x0 + (ep + 0.5) / n * (x1 - x0)
    Y = lambda v: y1 - (min(max(v, lo), hi) - lo) / (hi - lo) * (y1 - y0)
    for v in (-10, -5, 0, 5, 10):
        d.line([x0, Y(v), x1, Y(v)], fill=(40, 46, 56), width=1)
        text(d, (x0 - 16, Y(v)), f"{v:+d}" if v else "0", 30, "R", fill=DIM, anchor="rm")
    d.line([x0, y1, x1, y1], fill=(80, 88, 100), width=3)
    for k in range(0, n + 1, 10):
        text(d, (X(k) - (x1 - x0) / n / 2, y1 + 32), str(k), 30, "R", fill=DIM, anchor="mm")
    text(d, ((x0 + x1) // 2, y1 + 78), "training episode", 30, "R", fill=DIM, anchor="mm")
    for ep, label in ((5, "new program: measure first, then feel for the opening"), (WIN, "learner fixed, one reward for all arms")):
        xx = X(ep) - (x1 - x0) / n / 2
        for yy in range(int(y0), int(y1), 18):
            d.line([xx, yy, xx, yy + 9], fill=(110, 118, 130), width=2)
        text(d, (xx + 10, y0 + (0 if ep == 5 else 40)), label, 26, "R", fill=DIM)
    shown = [e for e in TRAIN if e["ep"] <= p * n]
    for arm in ("learned", "scripted", "sim"):
        g = [e for e in shown if e["arm"] == arm]
        pts = []
        for j in range(len(g)):
            w_ = g[max(0, j - 4):j + 1]
            pts.append((X(g[j]["ep"]), Y(sum(e["fixed"] for e in w_) / len(w_))))
        if len(pts) > 1 and arm != "sim":
            d.line(pts, fill=COL[arm], width=5)
    for e in shown:
        x, y = X(e["ep"]), Y(e["fixed"])
        if e["arm"] == "learned_eval":
            d.line([x - 9, y - 9, x + 9, y + 9], fill=DIM, width=4)
            d.line([x - 9, y + 9, x + 9, y - 9], fill=DIM, width=4)
        else:
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=arm_col(e["arm"]))
    lx = x1 - 560
    for j, (name, key) in enumerate((("learned (line: mean of last 5)", "learned"), ("scripted baseline (same)", "scripted"),
                                     ("sim policy, zero-shot", "sim"), ("before the arms", "pre"))):
        d.ellipse([lx, y1 - 220 + j * 46, lx + 22, y1 - 198 + j * 46], fill=COL[key])
        text(d, (lx + 36, y1 - 222 + j * 46), name, 28, "R", fill=DIM)
    text(d, (x0 - 130, (y0 + y1) // 2), "return", 30, "R", fill=DIM, anchor="mm")
    return pil


def return_segment(dur):
    n = int(dur * FPS)
    for k in range(n):
        yield np.asarray(return_frame(min(1.0, (k + 1) / (n * 0.7))))


def clean_frame(p):
    """Since episode WIN: did the drawer go straight in (no rim touch)? One dot per episode and arm."""
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    header(d, "Did it go straight in?", COL["learned"], f"training episodes {WIN} and later: learner fixed, one reward and the same program for every arm")
    rows = [("learned", "learned"), ("scripted baseline", "scripted"), ("sim policy, zero-shot", "sim")]
    y = 250
    for name, key in rows:
        st = S[key]
        text(d, (80, y), name, 46, "B", fill=COL[key])
        q = min(1.0, max(0.0, p * 3 - rows.index((name, key)) * 0.6))
        eps = [e for e in TRAIN if e["arm"] == key and e["ep"] >= WIN and e["rim"] is not None]
        for j, e in enumerate(eps[:int(round(q * len(eps)))]):
            cx, cy = 680 + j * 62, y + 26
            if e["rim"] == 0:
                d.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], fill=COL["ok"])
                text(d, (cx, cy), "✓", 30, "B", fill=BG, anchor="mm")
            else:
                d.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], outline=COL["fail"], width=4)
                text(d, (cx, cy), str(e["rim"]), 28, "B", fill=COL["fail"], anchor="mm")
        if q >= 1.0:
            text(d, (80, y + 64), f"{st['clean']} of {st['n']} went straight in  ·  {st['steps']:.1f} steps on average", 32, "R", fill=TXT)
        y += 200
    if p >= 1.0:
        text(d, (80, 860), "✓ = no touch on the rim  ·  number = rim touches before it dropped in. Every episode here ended with the "
             "drawer in.", 30, "R", fill=DIM)
        fp = S["fisher"]
        if fp is not None:
            text(d, (80, 910), f"Only {S['learned']['n'] + S['scripted']['n']} episodes, so this could still be luck "
                 f"(learned vs scripted, Fisher's exact test p = {fp:.2g}).", 30, "R", fill=DIM)
    return pil


def clean_segment(dur):
    n = int(dur * FPS)
    for k in range(n):
        yield np.asarray(clean_frame(min(1.0, (k + 1) / (n * 0.6))))


def sim_eval():
    d = jload(SIM / "results/lying/eval_carry.json", {})
    return [(name, 100 * d[key]["summary"]["success"], d[key]["summary"]["n"]) for key, name in
            (("straight", "straight down"), ("spiral", "spiral search"), ("guided", "force-guided script"), ("policy", "PPO policy"))]


def sim_bars_frame(p):
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    rows = sim_eval()
    n_ep = rows[0][2] if rows else 0
    header(d, "Success in simulation", COL["sim"], f"{n_ep} episodes each, the same random starts for every controller (MuJoCo)")
    x0, y1, top = 260, 860, 240
    bw = (W - 2 * x0) / len(rows)
    for i, (name, v, _) in enumerate(rows):
        q = min(1.0, max(0.0, p * len(rows) - i))
        hh = v / 100 * q * (y1 - top)
        xa, xb = x0 + i * bw + bw * 0.2, x0 + (i + 1) * bw - bw * 0.2
        col = COL["sim"] if name == "PPO policy" else (110, 118, 130)
        d.rectangle([xa, y1 - hh, xb, y1], fill=col)
        if q > 0.95:
            text(d, ((xa + xb) / 2, y1 - hh - 14), f"{v:.1f} %", 52, "B", fill=TXT, anchor="mb")
        text(d, ((xa + xb) / 2, y1 + 22), name, 36, "B" if name == "PPO policy" else "R", fill=col if name == "PPO policy" else DIM, anchor="mt")
    d.line([x0, y1, W - x0, y1], fill=(80, 88, 100), width=3)
    if p >= 1:
        text(d, (W // 2, 975), "The force-guided script was written after the policy showed the trick: step toward the sideways push of the rim.",
             30, "R", fill=DIM, anchor="mm")
    return pil


def sim_curve_frame(p):
    """Sim success while training: deterministic test on 4,096 starts every 10 iterations, runs ly_n1..ly_n4 chained."""
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    header(d, "Training in simulation", COL["sim"], "PPO on 4,096 parallel worlds (NVIDIA Newton on one RTX 4090), four runs, each one starting from the last")
    pts, off, bounds = [], 0.0, []
    for r in ("ly_n1", "ly_n2", "ly_n3", "ly_n4"):
        rows = [x.split(",") for x in (SIM / f"results/lying/curves_{r}.csv").read_text().split("\n")[1:] if x.strip()]
        for x in rows:
            if x[10]:
                pts.append((off + float(x[1]) / 1e6, 100 * float(x[10])))
        off += float(rows[-1][1]) / 1e6
        bounds.append(off)
    x0, y0, x1, y1 = 200, 200, W - 160, 860
    X = lambda s: x0 + s / off * (x1 - x0)
    Y = lambda v: y1 - v / 100 * (y1 - y0)
    for v in (0, 25, 50, 75, 100):
        d.line([x0, Y(v), x1, Y(v)], fill=(40, 46, 56), width=1)
        text(d, (x0 - 16, Y(v)), f"{v} %", 30, "R", fill=DIM, anchor="rm")
    for b in bounds[:-1]:
        d.line([X(b), y0, X(b), y1], fill=(60, 66, 78), width=2)
    for s in range(0, int(off) + 1, 10):
        text(d, (X(s), y1 + 30), f"{s}M", 28, "R", fill=DIM, anchor="mm")
    text(d, ((x0 + x1) // 2, y1 + 76), "simulated steps (millions)", 30, "R", fill=DIM, anchor="mm")
    m = max(2, int(len(pts) * p))
    d.line([(X(s), Y(v)) for s, v in pts[:m]], fill=COL["sim"], width=6)
    for s, v in pts[:m]:
        d.ellipse([X(s) - 6, Y(v) - 6, X(s) + 6, Y(v) + 6], fill=COL["sim"])
    if p >= 1:
        s, v = pts[-1]
        text(d, (X(s) - 10, Y(v) - 20), f"{v:.1f} %", 44, "B", fill=TXT, anchor="rb")
        text(d, (W // 2, 990), "Checked every 10 iterations on 4,096 test starts in Newton, no exploration. The bar chart uses a "
             "separate check: 500 episodes in MuJoCo.", 30, "R", fill=DIM, anchor="mm")
    return pil


def chart_segment(fn, dur, grow=0.6):
    n = int(dur * FPS)
    for k in range(n):
        yield np.asarray(fn(min(1.0, (k + 1) / (n * grow))))


@lru_cache(None)
def gif_frames():
    """The sim's own render of four episodes (61 frames each, 0.12 s apart): policy, straight, policy, straight."""
    return [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(Image.open(SIM_GIF))]


def sim_compare_segment(dur, chapter, sub, caption):
    """Same start, two controllers: the PPO policy drops in, the straight descent stays on the rim."""
    fr = gif_frames()
    eps = [(fr[0:61], "PPO policy", COL["sim"], ("ok", "✓  in")), (fr[61:122], "straight down", (150, 160, 172), ("fail", "✗  stuck"))]
    n = int(dur * FPS)
    sp = max(1.0, 61 * 0.12 / (dur - 1.5))
    k_done = None
    for k in range(n):
        j = min(60, int(k / FPS * sp / 0.12))
        k_done = k if (j >= 60 and k_done is None) else k_done
        pil = Image.fromarray(blank())
        d = ImageDraw.Draw(pil)
        header(d, chapter, COL["sim"], sub)
        for c, (seq, name, col, bd) in enumerate(eps):
            x = 500 + c * 620
            pil.paste(Image.fromarray(fit(seq[j], 456, 760)), (x - 12, 140))
            text(d, (x + 216, 966), name, 44, "B", fill=col, anchor="mb")
            if k_done is not None:
                badge_at(d, k - k_done, x + 216, 820, bd[0], bd[1], w=170)
        for i, line in enumerate(wrap(caption, 32, 330)[:8]):
            text(d, (40, 300 + i * 44), line, 32, "R", fill=DIM)
        yield np.asarray(pil)


def sim_real_segment(rc, speed, t0, t1, chapter, sub, caption, story=False):
    """Left: the policy in simulation (sim render). Right: the same policy, unchanged, on the real robot."""
    seq = gif_frames()[122:183]
    v = Video(rc["dir"] / "dashboard.mp4")
    span = t1 - t0
    k_badge = None
    for k, fr in enumerate(v.frames(speed, t0, t1)):
        side, wrist, _ = split_dash(fr)
        img = blank()
        j = min(60, int(k / FPS / 0.12))
        img[150:870, 40:472] = fit(seq[j], 432, 720)
        img[150:860, 520:1880] = fit(crop(side, SIDE_ZOOM), 1360, 710)[:710]
        img[560:860, 1347:1880] = fit(wrist, 533, 300)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        t_run = rc["t"] + t0 + k * speed / FPS
        header(d, chapter, COL["sim"], sub, t_run, speed)
        text(d, (52, 160), "in simulation", 28, "B", fill=TXT, shadow=True)
        text(d, (536, 160), "real robot, side camera (zoomed in)", 28, "B", fill=TXT, shadow=True)
        text(d, (1360, 570), "wrist camera", 24, "B", fill=TXT, shadow=True)
        d.rectangle([1347, 560, 1880, 860], outline=PANEL, width=4)
        if k * speed / FPS > span * 0.78:
            k_badge = k if k_badge is None else k_badge
            badge_at(d, k - k_badge, 900, 790, "sim", "✓  zero-shot", w=260)
        for i, line in enumerate(wrap(caption, 40, W - 80, "M")[:2]):
            text(d, (40, 884 + i * 50), line, 40, "M", fill=TXT)
        if story:
            counters(d, t_run, accent=COL["sim"])
            timeline_bar(d, t_run, pil)
        yield np.asarray(pil)


def stills_segment(paths, labels, dur, chapter, sub, caption, accent):
    """Two camera stills side by side (before / after), the second fading in."""
    ims = [fit(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), 900, 506) for p in paths]
    n = int(dur * FPS)
    for k in range(n):
        img = blank()
        img[170:676, 40:940] = ims[0]
        q = min(1.0, max(0.0, (k / n - 0.2) / 0.15))
        img[170:676, 980:1880] = (ims[1] * q + img[170:676, 980:1880] * (1 - q)).astype(np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub)
        text(d, (490, 700), labels[0], 34, "B", fill=TXT, anchor="ma")
        if q > 0.5:
            text(d, (1430, 700), labels[1], 34, "B", fill=accent, anchor="ma")
        for i, line in enumerate(wrap(caption, 44, W - 160, "M")[:3]):
            text(d, (W // 2, 790 + i * 56), line, 44, "M", fill=TXT, anchor="ma")
        yield np.asarray(pil)


def pair_segment(items, speed, chapter, sub, caption, accent):
    """Two episodes side by side (side camera, zoomed in), each with its own badge at the end.
    items: (rec, t0, t1, label, label colour, badge)."""
    gens = [Video(r["dir"] / "dashboard.mp4").frames(speed, t0, t1) for r, t0, t1, *_ in items]
    last, done_k = [None, None], [None, None]
    n = int(max(t1 - t0 for _, t0, t1, *_ in items) / speed * FPS) + 40
    for k in range(n):
        img = blank()
        for i, g in enumerate(gens):
            if done_k[i] is None:
                try:
                    last[i] = next(g)
                except StopIteration:
                    done_k[i] = k
            if last[i] is None:
                continue
            side, wrist, _ = split_dash(last[i])
            x = 20 + i * 950
            img[200:720, x:x + 924] = fit(crop(side, SIDE_ZOOM), 924, 520)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        header(d, chapter, accent, sub, None, speed)
        for i, (rc, t0, t1, label, lcol, bd) in enumerate(items):
            x = 20 + i * 950
            text(d, (x + 462, 140), label, 40, "B", fill=lcol, anchor="ma")
            if done_k[i] is not None:
                badge_at(d, k - done_k[i], x + 462, 800, bd[0], bd[1], w=250)
        for j, line in enumerate(wrap(caption, 36, W - 80, "M")[:2]):
            text(d, (40, 910 + j * 46), line, 36, "M", fill=TXT)
        yield np.asarray(pil)


def max_force(ep):
    """Largest contact force (N) in an episode's insertion segments (docs/report/make_tables.py max_force)."""
    import ast
    p = RUN / f"episodes/ep_{ep:06d}/log.txt"
    best = None
    for line in (p.read_text(errors="replace") if p.exists() else "").splitlines():
        m = re.search(r"exec \d+_(\w+): (\{.*\})\s*$", line)
        if not m or not m.group(1).startswith(("ins", "unload", "sim", "desc", "touch")):
            continue
        try:
            d = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            continue
        r = re.search(r"contact force ([\d.]+) N", d.get("reason", ""))
        for v in (d.get("max_force_n"), float(r.group(1)) if r else None):
            if v is not None:
                best = v if best is None else max(best, v)
    return best


def force_by_arm():
    out = {}
    for arm in ("learned", "scripted", "sim"):
        v = [max_force(e["ep"]) for e in TRAIN if e["arm"] == arm and e["ep"] >= WIN]
        v = [x for x in v if x is not None]
        out[arm] = (max(v) if v else 0, float(np.median(v)) if v else 0)
    return out


# ------------------------------------------------------------------ the recordings used (all real, this run)
CALIB = rec(SETUP / "sidecal/auto_20261123_155304")     # 15:53, automated side-camera calibration (camera moved 131 mm)
MEASURE = rec(SETUP / "lying/measure_20261123_160851")  # 16:08, cabinet measurement
RESET_OUT = rec("reset/attempts/attempt_36")            # 18:46, first time the reset took the drawer out of the cabinet
EDGE = rec("reset/attempts/attempt_9")                  # 16:40, reset test at the table edge, stopped by a human
MERGED = rec("reset/attempts/attempt_45")               # 19:30, reset 22: placed fine, rejected (drawer and cabinet one blob)
FIRST_IN = rec("execute/attempts/attempt_30")           # 18:37, first insertion (a test of the new program)
EP0 = ep_rec(0)                                         # 17:33, episode 0: on the rim
EP5 = ep_rec(5)                                         # 18:50, in, but scored 'unsure' (check bug)
TIP = rec("reset/attempts/attempt_101")                 # 01:10, recovery near the robot base: the drawer tips over when set down
RIGHT = rec("reset/attempts/attempt_105")               # 01:35, the reset agent stands the drawer back up (no human touch)
SIM31 = ep_rec(31)                                      # 22:20, sim policy zero-shot, in without a force stop
CALIB_LOG = (SETUP / "sidecal/auto_calib_20261123.log").read_text(errors="replace")
_m = re.search(r"poses (\d+) \(dropped (\d+)\) \| spread ([\d.]+) mm.*moved (\d+) mm, ([\d.]+) deg", CALIB_LOG)
CAL = dict(kept=int(_m.group(1)), dropped=int(_m.group(2)), spread=float(_m.group(3)), moved=int(_m.group(4)), deg=float(_m.group(5)))


def py_lines(role):
    return sum(len(p.read_text(errors="replace").splitlines()) for p in (RUN / "agents" / role / "programs").glob("*.py"))


def eval_text():
    return f"{S['eval'][0]} / {S['eval'][1]}" if S["eval"] else "[TBD] / 10"


# Every time a person stepped in during the run (OPERATOR_LOG.md; times as logged there).
HELP = [
    ("16:41", "Too close to the edge",
     "The first reset spot we gave the agent sat right on the table's edge. We moved it, but the agent's next test still "
     "used the old spot, so our watcher (the Claude Code session supervising the run) stopped tests 8 and 9 with Ctrl+C."),
    ("16:45", "Hands on", "The user moved the drawer away from the edge by hand. The only time anyone touched the scene during the run."),
    ("17:58", "Hints", "Episodes 0-4 all ended on the rim. Two notes from us: the second one said measure the opening before the "
     "grasp and feel for it with light touches. The agent built both and found the start offset itself."),
    ("18:57", "Our bugs", "Three real insertions were scored “unsure” by our own success check: episodes 5 and 6 (a full "
     "compartment shifts the cabinet's blob 3.3 cm) and 46 (it took the drawer for the cabinet). We fixed it twice; all three stay “unsure”."),
    ("19:45", "One big blob", "Two resets were rejected: drawer and cabinet merged into one blob for the camera. We moved the reset spot 2 cm."),
]
CALLS = [("19:10", "Add a scripted baseline: every 3rd episode runs with the learned part switched off (the user's call)"),
         ("19:49", "Freeze the scripted part, so the curves measure learning and not program edits"),
         ("20:19", "Run to 50 training episodes plus one 10-episode test, and try the sim policy on 5 episodes (the user's call)"),
         ("20:21", "One return formula for every episode, so the curve can show learning"),
         ("20:45", "Fix our cabinet guard (it misread the cabinet when the drawer stood next to it)"),
         ("21:27", "A hint for the learner: relabel the first descent with where the drawer actually went in"),
         ("23:58", "Fix our success check again (episode 46) and restart before the final test")]
LOG_CHECKED = "2026-11-24T03:10"     # HELP and CALLS cover OPERATOR_LOG.md up to here
NAP = ("00:45-02:22: resets failed with the drawer next to the robot base, and a recovery tipped it over. We sent notes and "
       "the user said keep trying. The reset agent stood it back up at 01:35 (attempt 105); nobody touched it.")
SETUP_HELP = ("Before the run, people set up the scene (cabinet on its back, checkerboard on the table) and pushed the drawer "
              "in by hand once to test the success check. The side camera had moved 13 cm since its last calibration.")


# ------------------------------------------------------------------ videos
def v_story(w):
    """The whole run, about 2 minutes."""
    w.add(card([("TWO AI AGENTS, A ROBOT ARM AND A DRAWER", 54, "B", TXT),
                ("Real-world RL: put the drawer back into the cabinet. It has 1-2 mm to spare on each side.", 44, "M", COL["unsure"]),
                (f"Claude Code agents running Claude Opus 5.5 wrote the resets, the episode program, the reward and the learner, "
                 f"and ran {S['n_train']} training episodes in one night. We helped a few times, and it's all marked.", 32, "R", DIM)],
               6.5, TXT, first=True, mascot=230))
    w.add(chapter_card("1 / 6", "Calibrate yourself", "nobody waving a checkerboard around", COL["calib"]))
    w.add(fade(dash_segment(CALIB, 48, "The arm calibrates the side camera", "before the run, started by us, then on its own",
                            f"The wrist camera finds the checkerboard on the table, then the arm stops at 12 poses where both cameras "
                            f"see it. {CAL['kept']} kept, {CAL['spread']:.1f} mm spread.", COL["calib"], t0=372, t1=664, story=True,
                            badge=("ok", "✓  calibrated"))))
    w.add(fade(dash_segment(MEASURE, 24, "Then it measures the cabinet", "a look straight down into the compartments",
                            "The cabinet lies on its back. The drawer goes into the compartment nearest the robot.", COL["calib"],
                            t0=160, t1=290, story=True)))
    w.add(chapter_card("2 / 6", "Write your own robot code", "and your own resets", COL["reset"]))
    w.add(fade(code_segment(3.0, "Two agents writing robot code", COL["reset"], f"{py_lines('reset'):,} + {py_lines('execute'):,} lines of Python")))
    w.add(fade(dash_segment(RESET_OUT, 16, "The reset agent's program", "pull the drawer out, stand it next to the cabinet",
                            f"Pinch the pull, lift the drawer out, stand it on the table. {S['resets']} resets so far, none by hand.",
                            COL["reset"], t0=8, t1=118, story=True, badge=("ok", "✓  reset"))))
    w.add(chapter_card("3 / 6", "Put it back", "with 1-2 mm to spare", COL["rl"]))
    w.add(fade(dash_segment(EP0, 7, "Episode 0", "the RL agent's first program", "Episodes 0 to 4 all end the same way: "
                            "the drawer sits on the rim.", COL["fail"], t0=50, t1=84, zoom=SIDE_ZOOM, story=True,
                            badge=("fail", "✗  on the rim"), badge_at_s=21)))
    w.add(card([("A HINT FROM US", 34, "B", COL["human"]),
                ("Our note: measure the opening before you pick up the drawer, then feel for it with light touches.", 48, "M", TXT),
                ("The agent built both, and worked out the right starting offset on its own.", 34, "R", DIM)], 5.0, COL["human"]))
    w.add(fade(dash_segment(FIRST_IN, 5, "First time in!", "a test of the new program, 18:37",
                            "It measures the opening first and starts 3 mm off center, where its probing had touched lowest. Straight in.", COL["ok"], t0=56, t1=92,
                            zoom=SIDE_ZOOM, story=True, badge=("ok", "✓  it went in!"))))
    w.add(fade(strip_segment(8.5)))
    w.add(chapter_card("4 / 6", "Did the learning help?", "a scripted baseline runs alongside", COL["learned"]))
    w.add(fade(clean_segment(9.0)))
    w.add(chapter_card("5 / 6", "Practice in simulation", "trained with PPO + SGS-style resets (CoRL 2026), then straight onto the robot", COL["sim"]))
    w.add(fade(sim_compare_segment(6.5, "In simulation", "a third agent built it: MuJoCo + NVIDIA Newton, PPO with an SGS-style curriculum",
                                   "Same start, 14 mm off. The policy steps toward the push of the rim and drops in.")))
    w.add(fade(chart_segment(sim_bars_frame, 5.0, 0.5)))
    w.add(fade(sim_real_segment(SIM31, 4, 54, 92, "Zero-shot on the real robot", "the sim policy, unchanged",
                                f"Not one real episode in its training. On the robot: {S['sim_ok']} / {len(S['sim_eps'])}.", story=True)))
    w.add(card([("A FEW TIMES, WE STEPPED IN", 34, "B", COL["human"]), ("All of it is in the log", 64, "B", TXT),
                (f"1 hand on the scene · 2 tests stopped by our watcher · {S['notes']} notes to the agents · {S['restarts']} restarts "
                 "to load our changes", 34, "R", DIM)], 3.0, COL["human"]))
    w.add(fade(dash_segment(EDGE, 7, "Too close to the edge", "human help, 16:42", "The agent's test still used an old spot "
                            "right at the table's edge, so we hit stop.", COL["human"], t0=100, t1=150, story=True,
                            badge=("human", "we hit stop"))))
    w.add(fade(stills_segment([EDGE["dir"] / "scene_after.jpg", rec("reset/attempts/attempt_12")["dir"] / "scene_before.jpg"],
                              ["16:43, right at the edge", "16:52, moved in by hand"], 3.5, "Hands on", "human help, 16:45",
                              "The user moved the drawer inward by hand. The only touch during the run.", COL["human"])))
    w.add(fade(dash_segment(TIP, 12, "The drawer took a nap", "during the final test, 01:10", "A reset near the robot base sets "
                            "the drawer down, and it tips over.", COL["fail"], t0=268, t1=330, story=True, badge=("fail", "✗  tipped over"))))
    w.add(fade(dash_segment(RIGHT, 24, "...and got back up", "01:35, no human touch", "The reset agent wrote a righting move and "
                            "stood it back up on its own.", COL["ok"], t0=5, t1=150, story=True, badge=("ok", "✓  back on its feet"))))
    w.add(chapter_card("6 / 6", "The final test", "10 episodes, learned policy, no exploration noise", COL["eval"]))
    w.add(fade(eval_part(8.0, story=True)))
    w.add(end_card())


def end_card(dur=6.5):
    lines = [(f"{S['ok']} of {S['n_train']} training episodes in  ·  {S['streak']} in a row", 60, "B", COL["ok"]),
             (f"Final test: {eval_text()}", 44, "B", TXT),
             (f"Sim policy, zero-shot on the real robot: {S['sim_ok']} / {len(S['sim_eps'])}", 40, "B", COL["sim"]),
             (f"{S['resets']} robot resets, none by hand. A human touched the scene once.", 34, "B", COL["human"]),
             ("People set the goal, made the big calls and kept an eye on safety.", 28, "R", DIM),
             ("Claude Code agents on Claude Opus 5.5  ·  Kinova Gen3  ·  MuJoCo + NVIDIA Newton", 28, "R", DIM)]
    return card(lines, dur, COL["ok"])


def eval_part(dur, story=False):
    """The final evaluation: a grid of its episodes when they exist, else a [TBD] card."""
    if S["eval"] is None and len(EVAL_EPS) < 10:
        done = f" ({len(EVAL_EPS)} of 10 done)" if EVAL_EPS else ""
        return card([("FINAL TEST", 34, "B", COL["eval"]), (f"[TBD] / 10{done}", 96, "B", TXT),
                     ("Runs after training episode 49: the learned policy, no exploration noise, 10 episodes.", 34, "R", DIM)], dur, COL["eval"])
    items = [(rec(e["dir"]), 45.0, 100.0, e["final"], (f"episode {e['ep']}", "eval")) for e in EVAL_EPS[:10]]
    return grid_segment(items, 4, 3, 13 if story else 7, "The final test", "learned policy, no exploration noise",
                        lambda k, n: f"{k} / {n} in", "Every tile is one real test episode (side camera, zoomed in). "
                        "✗ = not in, ? = our check said “unsure”.", COL["eval"],
                        dur=dur if story else None)


def v_sim(w):
    a = COL["sim"]
    f = force_by_arm()
    w.add(card([("SIM TO REAL", 30, "B", a), ("Trained only in simulation, then straight onto the real robot", 60, "B", TXT),
                ("While the robot ran, a third Claude Code agent built a simulation of this drawer and cabinet (MuJoCo, with an "
                 "NVIDIA Newton twin on the GPU) and trained a policy with PPO and an SGS-style reset curriculum (SGS-RL, CoRL 2026).",
                 32, "R", DIM)], 7.0, a, first=True, mascot=230))
    w.add(fade(chart_segment(sim_curve_frame, 8.0, 0.7)))
    w.add(fade(sim_compare_segment(9.0, "Same start, two controllers", "the drawer starts 14 mm off, tilted 5.7° in the grip",
                                   "The policy learned to step toward the sideways push of the rim. Going straight down stays stuck.")))
    w.add(fade(chart_segment(sim_bars_frame, 7.0, 0.5)))
    w.add(card([("THEN THE REAL ROBOT", 30, "B", a), ("Same weights, no fine-tuning", 64, "B", TXT),
                ("The real program does the grasp, the carry and the starting point, then the sim policy takes over the descent. "
                 "Its force stop was 35 N (25 N for the other arms).", 32, "R", DIM)], 5.5, a))
    w.add(fade(sim_real_segment(SIM31, 3, 52, 94, "Zero-shot on the real robot", f"episode 31 · the first of {len(S['sim_eps'])} sim-policy episodes",
                                "No force stop on the way in. Seated, flush, both checks say yes.")))
    items = [(rec(e["dir"]), 52.0, 96.0, e["final"], (f"episode {e['ep']}" + (" · straight in" if e["rim"] == 0 else f" · {e['rim']} rim touch" + ("es" if e["rim"] > 1 else "")), "sim"))
             for e in TRAIN if e["arm"] == "sim"]
    w.add(fade(grid_segment(items, 3, 2, 8, "All sim-policy episodes on the robot", None,
                            lambda k, n: f"{k} / {n} in", "Episodes 31, 34, 37, 40 and 43, every third episode of the second half of training.", a)))
    sim_clean = sum(e["rim"] == 0 for e in TRAIN if e["arm"] == "sim")
    w.add(card([(f"{S['sim_ok']} / {len(S['sim_eps'])} on the real robot, zero-shot", 64, "B", a),
                (f"{sim_clean} of {len(S['sim_eps'])} went straight in without touching the rim.", 40, "B", TXT),
                (f"Five tries is a small test, and it pressed harder than the real program: up to {f['sim'][0]:.0f} N, "
                 f"against {max(f['learned'][0], f['scripted'][0]):.0f} N for the other arms.", 30, "R", DIM),
                ("Sim: SGS-RL, Zhang et al., CoRL 2026 (reset curriculum) on top of PPO; MuJoCo and NVIDIA Newton.", 26, "R", DIM)], 6.5, a))


def v_curve(w):
    a = COL["learned"]
    w.add(card([("THE NUMBERS", 30, "B", a), ("Did the learning actually help?", 64, "B", TXT),
                ("To find out, every third training episode ran the same program with the learned part switched off: a "
                 "scripted baseline. The agent's learner is AWAC on the real robot's own episodes.", 32, "R", DIM)], 6.5, a, first=True, mascot=230))
    w.add(fade(strip_segment(9.0)))
    w.add(fade(return_segment(10.0)))
    w.add(card([("WHERE THE JUMP CAME FROM", 30, "B", a),
                ("Most of it came from the agent rebuilding its program after episode 4, with our hint. The learner came later.", 46, "M", TXT),
                ("What's left for the learner: getting in without touching the rim first.", 34, "R", DIM)], 5.0, a))
    w.add(fade(clean_segment(9.0)))
    L = [e for e in TRAIN if e["arm"] == "learned" and e["ep"] >= WIN and e["rim"] == 0 and e["final"] == "success"]
    Sc = [e for e in TRAIN if e["arm"] == "scripted" and e["ep"] >= WIN and e["rim"] and e["rim"] >= 2]
    if L and Sc:
        el, es = L[-1], Sc[-1]
        w.add(fade(pair_segment([(rec(el["dir"]), 58, 92, f"learned · episode {el['ep']}", COL["learned"], ("ok", "✓  straight in")),
                                 (rec(es["dir"]), 58, 92, f"scripted · episode {es['ep']}", COL["scripted"], ("unsure", f"{es['rim']} rim touches"))],
                                3, "Side by side", "side camera, zoomed in, carry and insertion",
                                "Both end with the drawer in. The learned one went straight in; the scripted one touched the rim twice first.", a)))
    st_l, st_s = S["learned"], S["scripted"]
    w.add(card([(f"Straight in: learned {st_l['clean']} of {st_l['n']}, scripted {st_s['clean']} of {st_s['n']}", 54, "B", a),
                (f"Steps per episode: {st_l['steps']:.1f} vs {st_s['steps']:.1f}. Counted successes: {st_l['ok']}/{st_l['n']} vs "
                 f"{st_s['ok']}/{st_s['n']}{unsure_note()}.", 34, "B", TXT),
                (f"Training episodes {WIN} and later. Small numbers (Fisher's exact p = {S['fisher']:.2g}), and the final test "
                 f"({eval_text()}) is the learned policy without exploration noise.", 30, "R", DIM)], 6.5, a))


def v_human(w):
    a = COL["human"]
    w.add(card([("BLOOPERS", 30, "B", a), ("We had to step in", 72, "B", TXT),
                ("Not often, and every time is in the operator log. “We” is the user plus the Claude Code session that watched the run.",
                 34, "R", DIM)], 5.0, a, first=True, mascot=250))
    w.add(card([("BEFORE THE RUN", 30, "B", a), (SETUP_HELP, 40, "M", TXT)], 5.0, a))
    t, title, txt = HELP[0]
    w.add(fade(dash_segment(EDGE, 6, f"#1 · {title}", f"human help, {t}", "The agent's reset test is about to set the drawer down "
                            "at the very edge of the table...", a, t0=95, t1=152, note=txt, badge=("human", "we hit stop"))))
    t, title, txt = HELP[1]
    w.add(fade(stills_segment([EDGE["dir"] / "scene_after.jpg", rec("reset/attempts/attempt_12")["dir"] / "scene_before.jpg"],
                              ["16:43, right at the edge", "16:52, moved in by hand"], 5.5, f"#2 · {title}", f"human help, {t}", txt, a)))
    t, title, txt = HELP[2]
    w.add(fade(dash_segment(EP0, 7, f"#3 · {title}", f"human help, {t} and 18:30", "Episodes 0-4: the drawer lands on the rim, "
                            "every time.", COL["fail"], t0=50, t1=84, zoom=SIDE_ZOOM, note=txt, badge=("fail", "✗  on the rim"), badge_at_s=21)))
    w.add(fade(dash_segment(FIRST_IN, 5, f"#3 · {title}", "18:37, the agent's new program", "Measure first, start where the probing touched lowest. "
                            "Straight in, no touch at all.", COL["ok"], t0=56, t1=92, zoom=SIDE_ZOOM, note=txt, badge=("ok", "✓  it went in!"))))
    t, title, txt = HELP[3]
    w.add(fade(dash_segment(EP5, 6, f"#4 · {title}", f"human help, {t}", "Episode 5: the drawer is in, flush. Our check said "
                            "“unsure”.", a, t0=50, t1=95, zoom=SIDE_ZOOM, note=txt, badge=("unsure", "“unsure”"))))
    t, title, txt = HELP[4]
    w.add(fade(dash_segment(MERGED, 10, f"#5 · {title}", f"human help, {t}", "Reset 22 puts the drawer down 4 mm from its target. "
                            "The camera can't tell it from the cabinet.", a, t0=50, t1=145, note=txt, badge=("human", "moved the spot 2 cm"))))
    w.add(fade(dash_segment(TIP, 8, "#6 · The drawer took a nap", "during the final test, 01:10", "A reset tries to get the "
                            "drawer away from the robot base, sets it down... and it tips over.", COL["fail"], t0=255, t1=330,
                            note=NAP, badge=("fail", "✗  tipped over"))))
    w.add(fade(dash_segment(RIGHT, 12, "#6 · ...and got back up", "01:35, no human touch", "Lift, turn, set down. Back on its "
                            "feet, no hands needed.", COL["ok"], t0=2, t1=150, note=NAP, badge=("ok", "✓  back on its feet"))))
    w.add(list_card("The calls we made", "decisions and notes from our side, as logged", CALLS, 9.0))
    w.add(card([("1 hand on the scene  ·  2 tests stopped", 60, "B", a),
                (f"{S['notes']} notes to the agents  ·  {S['restarts']} restarts to load our changes  ·  0 manual resets", 40, "B", TXT),
                (f"The rest ran on its own: {S['resets']} robot resets and {S['n_train']} training episodes.", 32, "R", DIM)], 6.0, a))


def list_card(title, sub, rows, dur):
    pil = Image.fromarray(blank())
    d = ImageDraw.Draw(pil)
    header(d, title, COL["human"], sub)
    y = 190
    for t, s in rows:
        text(d, (80, y), t, 38, "B", fill=COL["human"])
        for j, line in enumerate(wrap(s, 38, W - 340)[:2]):
            text(d, (260, y + j * 48), line, 38, "R", fill=TXT)
        y += 48 * max(1, len(wrap(s, 38, W - 340)[:2])) + 34
    return still(np.asarray(pil), dur)


def v_eval(w):
    a = COL["eval"]
    w.add(card([("THE FINAL TEST", 30, "B", a), (eval_text(), 110, "B", TXT),
                (f"After {S['n_train']} training episodes: the learned policy without exploration noise, 10 episodes in a row, "
                 "same resets and the same two independent checks.", 32, "R", DIM)], 6.0, a, first=True, mascot=200))
    w.add(fade(eval_part(0)))
    fails = [e for e in EVAL_EPS if e["final"] != "success"]
    if fails:
        fe = fails[0]
        w.add(fade(dash_segment(rec(fe["dir"]), 5, f"A miss: episode {fe['ep']}", f"{len(fails)} of {len(EVAL_EPS)} test episodes "
                                f"failed ({', '.join(str(e['ep']) for e in fails)}), all the same way",
                                "It drops 4-6 mm into the opening and jams, tilted, at the first touch. The program has no "
                                "lift-and-try-again move, so it gives up and puts the drawer back on the table.", COL["fail"],
                                t0=52, t1=92, zoom=SIDE_ZOOM, badge=("fail", "✗  jammed"), badge_at_s=17)))
    if EVAL_EPS:
        hero = next((e for e in EVAL_EPS if e["final"] == "success"), EVAL_EPS[0])
        w.add(fade(dash_segment(rec(hero["dir"]), 5, "One test episode, start to finish", f"episode {hero['ep']}",
                                "Look, grab the pull, carry, feel for the opening, let go.", a, t0=5, t1=110, zoom=SIDE_ZOOM,
                                badge=("ok", "✓  in") if hero["final"] == "success" else ("fail", "✗  not in"))))
    w.add(end_card())


VIDEOS = {"story": ("00_full_story.mp4", v_story), "sim": ("01_sim_to_real.mp4", v_sim),
          "curve": ("02_learning_curve_and_arms.mp4", v_curve), "human": ("03_human_help.mp4", v_human),
          "eval": ("04_final_evaluation.mp4", v_eval)}
showcase_music.STYLES.update({   # synthesized music per video (scripts/showcase_music.py), keyed by the video names
    "d_story": dict(prog="minor", bpm=90, key=-2, seed=11, build=True), "d_sim": dict(prog="major", bpm=94, key=3, seed=12),
    "d_curve": dict(prog="minor", bpm=86, key=1, seed=13), "d_human": dict(prog="major", bpm=104, key=-1, seed=14, playful=True),
    "d_eval": dict(prog="major", bpm=92, key=0, seed=15, build=True)})


class DraftWriter(Writer):
    """Writer that stamps 'DRAFT' on every frame while the final numbers are missing."""

    def __init__(self, path):
        super().__init__(path, threads=THREADS)
        self.tag = None
        if DRAFT:
            pil = Image.new("RGBA", (330, 54), (0, 0, 0, 0))
            d = ImageDraw.Draw(pil)
            d.rounded_rectangle([0, 0, 329, 53], 12, fill=(200, 40, 40, 220))
            text(d, (165, 27), "DRAFT · numbers pending", 26, "B", fill=(255, 255, 255), anchor="mm")
            self.tag = np.asarray(pil).astype(np.float32)

    def add(self, frames):
        for f in frames:
            if self.tag is not None:
                f = np.array(f, np.uint8)
                y, x = 132, W - 350
                a = self.tag[..., 3:] / 255.0
                reg = f[y:y + 54, x:x + 330].astype(np.float32)
                f[y:y + 54, x:x + 330] = (self.tag[..., :3] * a + reg * (1 - a)).astype(np.uint8)
            self.p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
            self.n += 1


def render(name):
    fname, fn = VIDEOS[name]
    t = time.time()
    w = DraftWriter(OUT / fname)
    fn(w)
    secs = w.close()
    showcase_music.mux(OUT / fname, "d_" + name)
    return f"{fname}: {secs:.1f} s, {(OUT / fname).stat().st_size / 1e6:.0f} MB, rendered in {time.time() - t:.0f} s"


# ------------------------------------------------------------------ X thread texts, stills and the web page
def xlen(s):
    """Length as X counts it: most Latin/Greek characters 1, everything else (arrows, emoji, CJK) 2."""
    ok = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))
    return sum(1 if any(a <= ord(c) <= b for a, b in ok) else 2 for c in s)


SGS = ("SGS-RL (Success Guided Sampling): Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots, \"A Balanced Data Diet: "
       "Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control\", CoRL 2026")


def fail_note():
    fails = [e["ep"] for e in EVAL_EPS if e["final"] != "success"]
    return "" if not fails or S["eval"] is None else (f" (misses: episodes {', '.join(map(str, fails))}, all the same way: "
                                                      "jammed tilted 4-6 mm into the opening, no lift-and-retry in the program)")


def unsure_note():
    un = [e["ep"] for e in TRAIN if e["ep"] >= WIN and e["final"] == "unsure" and e["arm"] in ("learned", "scripted")]
    return "" if not un else (f" (episode{'s' if len(un) > 1 else ''} {', '.join(map(str, un))}: in, but scored \"unsure\" "
                              f"by a bug in our check)")


def thread():
    """The 4-post thread: (title, attach, short text <= 280, long text)."""
    L, Sc = S["learned"], S["scripted"]
    ev = f"{S['eval'][0]}/{S['eval'][1]}" if S["eval"] else "[TBD]/10"
    sims = sim_eval()
    simtxt = ", ".join(f"{v:.1f}% {n}" for n, v, _ in sims[:3])
    p1s = (f"Round 2: Claude Code agents (Opus 5.5) vs. a drawer. Real-world RL on our arm, cabinet on its back, "
           f"1-2 mm of clearance per side.\n\nThey wrote the resets, the episode code, the reward and the learner. First 5 "
           f"episodes: stuck on the rim. Later {S['streak']} in a row, final test {ev} 🗄️")
    p1l = (p1s + f"\n\nA third agent built a sim of the drawer (MuJoCo + NVIDIA Newton) and trained a PPO policy. We put it on "
           f"the real robot without changing a thing: {S['sim_ok']}/{len(S['sim_eps'])}.\n\nWe stepped in a few times. Every "
           f"time is marked in the video.")
    p2s = (f"Numbers ({S['n_train']} training episodes, one night, 0 manual resets):\n"
           f"- {S['ok']}/{S['n_train']} in, {S['streak']} in a row\n"
           f"- ep ≥{WIN}, straight in w/o rim touch: learned {L['clean']}/{L['n']} vs scripted baseline {Sc['clean']}/{Sc['n']} "
           f"(p={S['fisher']:.2g})\n"
           f"- final test, deterministic: {ev}\n"
           f"- sim PPO policy, zero-shot: {S['sim_ok']}/{len(S['sim_eps'])}")
    p2l = (f"The numbers. One night (16:16 to 02:42), {S['n_train']} training episodes, {S['resets']} robot resets, none by hand.\n\n"
           f"Success = two independent checks agree: side-camera depth geometry and a Claude Opus 5.5 judge on before/after images.\n\n"
           f"- Episodes 0-4: 0/5, drawer on the rim. The agent rebuilt its program (measure the opening before the grasp, then a "
           f"touch search; both ideas came from a note we sent, the start offset it found itself). Since episode 9: "
           f"{S['since9'][0]}/{S['since9'][1]} counted, {S['streak']} of them in a row; "
           f"{', '.join(str(e['ep']) for e in TRAIN if e['final'] == 'unsure' and e['ep'] >= 9) or 'none'} put the drawer in "
           f"but our check said \"unsure\" (a bug).\n"
           f"- Learned (AWAC on the robot's own episodes) vs a scripted baseline (same program, learned part off, every 3rd "
           f"episode), episodes ≥{WIN}: straight in without touching the rim {L['clean']}/{L['n']} vs {Sc['clean']}/{Sc['n']} "
           f"(Fisher's exact p = {S['fisher']:.2g}), {L['steps']:.1f} vs {Sc['steps']:.1f} steps, counted successes "
           f"{L['ok']}/{L['n']} vs {Sc['ok']}/{Sc['n']}{unsure_note()}.\n"
           f"- Final test after {S['n_train']} training episodes: deterministic policy, 10 episodes: {ev}"
           f"{fail_note()}.\n"
           f"- Sim: PPO with an SGS-style reset curriculum, 4,096 worlds in NVIDIA Newton on one RTX 4090 (about 1.5 h). "
           f"500 MuJoCo episodes, same starts: PPO {sims[3][1]:.1f}% vs {simtxt}. On the real robot, unchanged: "
           f"{S['sim_ok']}/{len(S['sim_eps'])}.\n\nSGS: Zhang et al., \"A Balanced Data Diet\", CoRL 2026.")
    n_un = len([e for e in TRAIN if e["final"] == "unsure"])
    p3s = ("Bloopers. We put the first reset spot right on the table's edge (our bad): we hit Ctrl+C twice, and I moved the "
           f"drawer back by hand once. Bugs in our own check scored {n_un} real insertions \"unsure\".\n\nMid-test the drawer "
           "tipped over. The reset agent stood it back up by itself 😅")
    p3l = (p3s + "\n\nAnd the fix for the rim? Two ideas from a note we sent. \"We\" = me plus a Claude Code session watching "
           f"the run.\n\nAll in all: 1 hand on the scene, 2 tests stopped, {S['notes']} notes to the agents, {S['restarts']} "
           f"restarts to load our changes. The big calls were ours too: a scripted baseline, the freeze, one return formula, "
           f"the 10-episode test, trying the sim policy.")
    p4s = ("Takeaway: agents can take over the chores of real-world RL: resets, rewards, logging, plots, even a sim.\n\nA "
           "scripted baseline kept us honest. Most of the gain came from the agent rewriting its program; the learner only cut "
           "rim touches. And humans still made the key calls.")
    p4l = p4s + "\n\nP.S. Opus 5.5 made these videos too."
    return [("The main video", ["00_full_story.mp4"], p1s, p1l),
            ("The numbers", ["thread/episodes.png", "thread/learned_vs_scripted.png", "thread/sim_results.png", "thread/return.png"], p2s, p2l),
            ("Bloopers", ["03_human_help.mp4"], p3s, p3l),
            ("Takeaway", [], p4s, p4l)]


EXTRAS = [("01_sim_to_real.mp4", "Sim to real: trained in simulation, 5 episodes on the real robot"),
          ("02_learning_curve_and_arms.mp4", "The numbers: learned vs scripted baseline"),
          ("04_final_evaluation.mp4", "The final test")]


def posts_md():
    out = ["# X thread: real-world RL with Claude Code agents (drawer into a lying cabinet)", "",
           ("**DRAFT**: the final evaluation has not run yet; numbers marked [TBD] are filled in by the re-render.\n" if DRAFT else ""),
           "Counts are X's (arrows, ≥ and emoji count 2). Short versions fit 280; long versions need a Premium account.",
           f"Sim credit: {SGS}.", ""]
    for i, (title, media, s, l) in enumerate(thread(), 1):
        out += [f"## {i}/4 · {title}", "", "Attach: " + (", ".join(f"`{m}`" for m in media) or "nothing (text only)") +
                (" (X: up to 4 images per post)" if media and media[0].endswith(".png") else ""), "",
                f"Short ({xlen(s)}/280):", "", "```", s, "```", "", f"Long ({xlen(l)}):", "", "```", l, "```", ""]
    out += ["Extra videos (replies, a project page): " + ", ".join(f"`{f}`" for f, _ in EXTRAS), ""]
    return "\n".join(out)


def thread_images():
    (OUT / "thread").mkdir(parents=True, exist_ok=True)
    strip_frame(1.0).save(OUT / "thread/episodes.png")
    clean_frame(1.0).save(OUT / "thread/learned_vs_scripted.png")
    sim_bars_frame(1.0).save(OUT / "thread/sim_results.png")
    return_frame(1.0).save(OUT / "thread/return.png")


def probe(f):
    o = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(f)]))
    return float(o["format"]["duration"]), int(o["format"]["size"])


def build_site():
    """videos/POSTS.md, thread stills, posters; SITE/drawer_videos.html (files linked from SITE/drawer_videos/), linked from index."""
    thread_images()
    (OUT / "POSTS.md").write_text(posts_md())
    for title, media, s, l in thread():
        assert xlen(s) <= 280, (title, xlen(s))
    vids = [f for f, _ in VIDEOS.values() if (OUT / f).exists()]
    (OUT / "posters").mkdir(exist_ok=True)
    for f in vids:
        dur, _ = probe(OUT / f)
        assert dur <= 140.5, (f, dur)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(min(dur - 1, 0.2 if f.startswith("00") else 1.0)), "-i", str(OUT / f),
                        "-frames:v", "1", "-vf", "scale=960:-1", "-q:v", "4", str(OUT / "posters" / (f[:-4] + ".jpg"))], check=True)
    link = SITE / "drawer_videos"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(OUT)
    ver = lambda rel: f"drawer_videos/{rel}?v={int((OUT / rel).stat().st_mtime)}"

    def media(rel):
        if rel.endswith(".mp4"):
            dur, size = probe(OUT / rel)
            return (f'<video controls playsinline preload="metadata" poster="{ver("posters/" + rel[:-4] + ".jpg")}" src="{ver(rel)}"></video>'
                    f'<div class="bar"><span>{rel} · {int(dur // 60)}:{int(dur % 60):02d} · {size / 1e6:.0f} MB</span>'
                    f'<a class="btn" href="drawer_videos/{rel}" download>Download</a></div>')
        return (f'<img src="{ver(rel)}" alt="{html.escape(Path(rel).stem)}"><div class="bar"><span>{Path(rel).name}</span>'
                f'<a class="btn" href="drawer_videos/{rel}" download>Download</a></div>')
    posts = []
    for i, (title, med, s, l) in enumerate(thread()):
        att = "".join(media(m) for m in med if (OUT / m).exists()) or '<p class="none">Text only, nothing to attach.</p>'
        posts.append(f"""
<section>
  <div class="tag">Post {i + 1} of 4</div><h2>{html.escape(title)}</h2>
  <div class="post"><div class="ph"><span>short · {xlen(s)}/280</span><button onclick="cp('s{i}')">Copy</button></div><pre id="s{i}">{html.escape(s)}</pre></div>
  <details><summary>Long version ({xlen(l)} characters, Premium)</summary>
  <div class="post"><div class="ph"><span>long · {xlen(l)}</span><button onclick="cp('l{i}')">Copy</button></div><pre id="l{i}">{html.escape(l)}</pre></div></details>
  <div class="att">{att}</div>
</section>""")
    extras = "".join(f'<section><div class="tag">Extra</div><h2>{html.escape(t)}</h2>{media(f)}</section>' for f, t in EXTRAS if (OUT / f).exists())
    draft = ('<p class="draft">DRAFT: the run\'s final evaluation has not happened yet. Numbers marked [TBD] and the red tag in '
             'the videos go away in the final render.</p>' if DRAFT else "")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drawer RL videos</title>
<style>
  :root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #66645e; --card: #ffffff; --line: #e4e2dc; --accent: #2f6fb0; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #161615; --fg: #ecebe6; --muted: #a3a19a; --card: #1f1f1d; --line: #33322e; --accent: #7fb0e3; }} }}
  body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 28px 16px 60px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; }} h2 {{ font-size: 19px; margin: 0 0 8px; }}
  p {{ margin: 0 0 10px; color: var(--muted); }}
  section {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 18px 0; }}
  video, img {{ width: 100%; border-radius: 6px; background: #000; display: block; margin-top: 12px; }}
  .tag {{ font-size: 13px; color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
  .post {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin: 8px 0; }}
  .ph {{ display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 13px; }}
  .ph button {{ border: 1px solid var(--line); background: var(--bg); color: var(--fg); border-radius: 6px; padding: 3px 10px; cursor: pointer; }}
  pre {{ white-space: pre-wrap; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; margin: 6px 0 0; }}
  summary {{ cursor: pointer; color: var(--accent); margin: 6px 0; }}
  .bar {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; margin-top: 6px; }}
  .btn {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
  .none {{ margin-top: 10px; }} .draft {{ color: #c62828; font-weight: 600; }}
</style></head><body><main>
<h1>Drawer RL: videos and the X thread</h1>
<p>Run 20261123_1616_drawer. Four posts with their text (short and long version) and the files to attach, then the extra
videos. Every clip is a real recording of the run or the simulation's own render; every human step-in is marked.
<a href="index.html">All simulation replays</a> · <a href="drawer_videos/POSTS.md">POSTS.md</a></p>
{draft}
{''.join(posts)}
{extras}
</main><script>function cp(id){{navigator.clipboard.writeText(document.getElementById(id).innerText)}}</script></body></html>"""
    (SITE / "drawer_videos.html").write_text(page)
    idx = (SITE / "index.html").read_text()
    block = ('<!-- drawer-videos -->\n  <section>\n    <div class="tag">Drawer run, showcase</div>\n    <h2><a href="drawer_videos.html">'
             'Videos and X thread drafts for the drawer RL run</a></h2>\n    <p>The full story, sim to real, the numbers, the human '
             'help and the final test.</p>\n  </section>\n<!-- /drawer-videos -->')
    idx = re.sub(r"<!-- drawer-videos -->.*?<!-- /drawer-videos -->", block, idx, flags=re.S) if "<!-- drawer-videos -->" in idx \
        else idx.replace("\n  <section>", "\n  " + block + "\n\n  <section>", 1)
    (SITE / "index.html").write_text(idx)
    url = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", (SITE / "cloudflared.log").read_text(errors="replace"))
    print(f"page: {url[-1] if url else 'http://localhost:8096'}/drawer_videos.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(VIDEOS))
    ap.add_argument("--jobs", type=int, default=3, help="videos rendered in parallel (each: 1 Python + %d x264 threads)" % THREADS)
    ap.add_argument("--site-only", action="store_true", help="only POSTS.md, the thread stills and the web page")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'DRAFT: ' if DRAFT else ''}{S['n_train']} training episodes, {S['ok']} in, streak {S['streak']}, "
          f"eval {S['eval']}, sim {S['sim_ok']}/{len(S['sim_eps'])}, notes {S['notes']}, restarts {S['restarts']}", flush=True)
    late = [e for e in EVENTS if e["kind"] in ("operator_note", "restart") and e["t"] > LOG_CHECKED]
    if late:
        print(f"check HELP/CALLS against OPERATOR_LOG.md: {len(late)} operator notes/restarts after {LOG_CHECKED}", flush=True)
    if not a.site_only:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for msg in ex.map(render, a.names):
                print(msg, flush=True)
    build_site()


if __name__ == "__main__":
    main()
