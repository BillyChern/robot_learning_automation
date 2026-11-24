"""Drawing and video helpers shared by the showcase video generators (make_showcase_videos.py, make_drawer_videos.py).

1920x1080 at 30 fps; text in Ubuntu (DejaVu for check marks and arrows); recordings read with OpenCV; H.264 out via ffmpeg.
"""
import subprocess
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

W, H, FPS = 1920, 1080, 30
BG = (12, 16, 22)
PANEL = (22, 27, 34)
TXT = (235, 240, 245)
DIM = (150, 160, 172)
FONTS = "/usr/share/fonts/truetype/ubuntu/"
FALLBACK = set("✓✗→")


@lru_cache(None)
def font(size, weight="R"):
    return ImageFont.truetype(FONTS + f"Ubuntu-{weight}.ttf", size)


@lru_cache(None)
def mono(size):
    return ImageFont.truetype(FONTS + "UbuntuMono-R.ttf", size)


@lru_cache(None)
def dejavu(size, weight):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight in "BM" else ""), size)


def runs(s, size, weight):
    out = []
    for ch in s:
        f = dejavu(size, weight) if ch in FALLBACK else font(size, weight)
        if out and out[-1][1] is f:
            out[-1][0] += ch
        else:
            out.append([ch, f])
    return out


def text(d, xy, s, size, weight="R", fill=TXT, anchor="la", shadow=False):
    """Draw text; characters missing from Ubuntu (check marks, arrows) come from DejaVu Sans."""
    if not FALLBACK & set(s):
        f = font(size, weight)
        if shadow:
            d.text((xy[0] + 2, xy[1] + 2), s, font=f, fill=(0, 0, 0), anchor=anchor)
        d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return
    rs = runs(s, size, weight)
    total = sum(f.getlength(t) for t, f in rs)
    x = xy[0] - {"l": 0, "m": total / 2, "r": total}[anchor[0]]
    for t, f in rs:
        if shadow:
            d.text((x + 2, xy[1] + 2), t, font=f, fill=(0, 0, 0), anchor="l" + anchor[1])
        d.text((x, xy[1]), t, font=f, fill=fill, anchor="l" + anchor[1])
        x += f.getlength(t)


def wrap(s, size, width, weight="R"):
    f, words, lines, cur = font(size, weight), s.replace(" %", " %").split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def balanced(s, size, width, weight):
    """Wrap into as many lines as needed, but of similar length (no one-word last line)."""
    lines = wrap(s, size, width, weight)
    if len(lines) < 2:
        return lines
    target = font(size, weight).getlength(s) / len(lines) * 1.06
    for w_ in range(int(target), int(width) + 1, 20):
        trial = wrap(s, size, w_, weight)
        if len(trial) == len(lines):
            return trial
    return lines


def fit(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_CUBIC)


def blank():
    return np.full((H, W, 3), BG, np.uint8)


def fade(gen, n_in=8, n_out=8):
    """Fade from and to black, streaming (only the last n_out frames are held)."""
    buf = deque()
    for i, f in enumerate(gen):
        if i < n_in:
            f = (f.astype(np.float32) * (i + 1) / (n_in + 1)).astype(np.uint8)
        buf.append(f)
        if len(buf) > n_out:
            yield buf.popleft()
    m = len(buf)
    for j, f in enumerate(buf):
        yield (f.astype(np.float32) * (m - j) / (m + 1)).astype(np.uint8)


class Video:
    """Sequential reader of a recording with speed-up; yields RGB frames at the output rate."""

    def __init__(self, path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dur = self.n / self.fps
        cap.release()

    def frames(self, speed, t0=0.0, t1=None):
        cap = cv2.VideoCapture(self.path)
        start = int(t0 * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        cur, last, k = start - 1, None, 0
        t1 = self.dur if t1 is None else min(t1, self.dur)
        while True:
            t = t0 + k * speed / FPS
            want = int(t * self.fps)
            if t > t1 or want >= self.n:
                break
            while cur < want:
                if cur < want - 1:
                    ok = cap.grab()
                else:
                    ok, fr = cap.read()
                    if ok:
                        last = fr
                if not ok:
                    cap.release()
                    return
                cur += 1
            if last is None:
                break
            yield cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
            k += 1
        cap.release()


def split_dash(fr):
    """Dashboard frame (480x1280) -> side, wrist (360x640 each), log strip (120x1280)."""
    return fr[:360, :640], fr[:360, 640:1280], fr[360:480, :1280]


class Writer:
    """Raw RGB frames -> H.264 (CRF 19) with a silent AAC track (the music is muxed in afterwards).
    threads: libx264 thread count (None = ffmpeg's default, one per core)."""

    def __init__(self, path, threads=None):
        self.path = Path(path)
        self.n = 0
        thr = [] if threads is None else ["-threads", str(threads)]
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest", "-c:v", "libx264", "-preset", "medium", "-crf", "19"]
            + thr + ["-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
                     str(self.path)], stdin=subprocess.PIPE)

    def add(self, frames):
        for f in frames:
            self.p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
            self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        return self.n / FPS
