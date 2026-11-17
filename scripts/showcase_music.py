"""Calm background music for the showcase videos, synthesized here (pads, plucked arpeggio, soft bass and drums),
so there is nothing to license. mux() puts a track of the right length under a rendered video.

    third_party/openpi/.venv/bin/python scripts/showcase_music.py <video.mp4> [style]
"""
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
PROGS = {   # (bass midi, pad notes) per chord
    "minor": [(45, (57, 60, 64, 67, 71)), (41, (53, 57, 60, 64, 67)), (48, (55, 60, 64, 67, 71)), (43, (55, 59, 62, 64, 69))],
    "major": [(48, (55, 60, 64, 67, 72)), (43, (55, 59, 62, 67, 71)), (45, (57, 60, 64, 69, 72)), (41, (53, 57, 60, 65, 69))],
}
STYLES = {
    "story": dict(prog="minor", bpm=88, key=0, seed=1, build=True),
    "agents": dict(prog="minor", bpm=92, key=2, seed=2),
    "resets": dict(prog="major", bpm=96, key=-3, seed=3),
    "collection": dict(prog="minor", bpm=86, key=-2, seed=4),
    "training": dict(prog="minor", bpm=84, key=3, seed=5),
    "eval_dagger": dict(prog="minor", bpm=90, key=0, seed=6),
    "final": dict(prog="major", bpm=92, key=0, seed=7, build=True),
    "highlights": dict(prog="major", bpm=88, key=2, seed=8),
    "human": dict(prog="major", bpm=104, key=0, seed=9, playful=True),
}


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def filt(x, fc, kind="low", order=2):
    return sosfilt(butter(order, fc, kind, fs=SR, output="sos"), x)


def add(buf, start, x):
    end = min(len(buf), start + len(x))
    if 0 <= start < end:
        buf[start:end] += x[:end - start]


def shift(x, d):
    return np.concatenate([np.zeros(d), x[:-d]])


def music(dur, prog="minor", bpm=88, key=0, seed=0, build=False, playful=False):
    """Stereo track (2, n) of dur seconds: pads from the start, arpeggio and bass after the intro, drums in the middle."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    per = 1 if playful else 2                      # bars per chord
    n = int((dur + 0.2) * SR)
    pad = np.zeros((2, n))
    arp = np.zeros((2, n))
    bass = np.zeros(n)
    drums = np.zeros(n)
    chords = PROGS[prog]
    arp_in = min(2 * bar, dur * 0.12)
    drum_in, drum_out = dur * (0.15 if playful else (0.3 if build else 0.22)), dur - 2 * bar
    pattern = [0, 2, 4, 1, 3, 4, 2, 1] if seed % 2 else [0, 1, 2, 4, 3, 2, 4, 1]

    for c in range(int(np.ceil(dur / (bar * per))) + 1):
        t0 = c * bar * per
        if t0 >= dur:
            break
        root, notes = chords[c % len(chords)]
        span = bar * per
        L = int((span + 1.6) * SR)
        t = np.arange(L) / SR
        e = np.clip(t / 0.7, 0, 1) * np.where(t < span, 1.0, np.clip(1 - (t - span) / 1.6, 0, 1) ** 2)
        e *= 1 + 0.08 * np.sin(2 * np.pi * 0.23 * t)
        for m in notes:
            f = hz(m + key)
            for det, gl, gr in ((-6, 0.8, 0.35), (0, 0.6, 0.6), (6, 0.35, 0.8)):
                ff = f * 2 ** (det / 1200)
                ph = rng.uniform(0, 2 * np.pi)
                w = (np.sin(2 * np.pi * ff * t + ph) + 0.22 * np.sin(4 * np.pi * ff * t + ph) + 0.08 * np.sin(6 * np.pi * ff * t)) * e * 0.035
                add(pad[0], int(t0 * SR), w * gl)
                add(pad[1], int(t0 * SR), w * gr)
            sh = np.sin(2 * np.pi * 2 * f * t + rng.uniform(0, 6.3)) * e * 0.006   # shimmer an octave up
            add(pad[0], int(t0 * SR), sh)
            add(pad[1], int(t0 * SR), sh)
        for b in range(per):                       # bass: half notes on the root
            for k, ln in ((0, 2.0), (2, 1.5)) if not playful else ((0, 1.0), (1.5, 0.5), (2, 1.0), (3.5, 0.5)):
                ts = t0 + b * bar + k * beat
                if ts < arp_in or ts >= dur - bar:
                    continue
                Lb = int((ln * beat + 0.15) * SR)
                tb = np.arange(Lb) / SR
                eb = np.clip(tb / 0.01, 0, 1) * np.clip(1 - (tb - ln * beat) / 0.15, 0, 1) * np.exp(-tb / 1.2)
                f = hz(root + key)
                add(bass, int(ts * SR), np.tanh(1.4 * (np.sin(2 * np.pi * f * tb) + 0.35 * np.sin(4 * np.pi * f * tb))) * eb * 0.085)
        tones = sorted(m + 12 for m in notes)
        for b in range(per):                       # plucked 8th-note arpeggio
            for k in range(8):
                ts = t0 + b * bar + k * beat / 2 + rng.normal(0, 0.004)
                if ts < arp_in or ts >= dur - 0.6 or (k and rng.random() < 0.1):
                    continue
                f = hz(tones[pattern[k] % len(tones)] + key)
                tau = 0.18 if playful else 0.38
                Lp = int(tau * 5 * SR)
                tp = np.arange(Lp) / SR
                ep = np.clip(tp / 0.004, 0, 1) * np.exp(-tp / tau)
                w = (np.sin(2 * np.pi * f * tp) + 0.45 * np.sin(4 * np.pi * f * tp) * np.exp(-tp / 0.15)
                     + 0.2 * np.sin(6 * np.pi * f * tp) * np.exp(-tp / 0.08) + 0.08 * np.sin(8 * np.pi * f * tp) * np.exp(-tp / 0.05)) * ep
                if k == 0:                         # soft bell an octave up on each downbeat
                    w = w + 0.35 * np.sin(4 * np.pi * f * tp) * np.exp(-tp / (tau * 2.5))
                v = 0.055 * (1.15 if k % 4 == 0 else 1.0) * rng.uniform(0.85, 1.1) * min(1.0, (ts - arp_in) / (2 * bar) + 0.3)
                pan = 0.65 if k % 2 else 0.35
                add(arp[0], int(ts * SR), w * v * (1 - pan) * 2)
                add(arp[1], int(ts * SR), w * v * pan * 2)
        for b in range(per):                       # soft kick on 1 and 3, hats on the off-beats
            for k in range(8):
                ts = t0 + b * bar + k * beat / 2
                if not drum_in <= ts < drum_out:
                    continue
                if k in (0, 4) or (playful and k == 7):
                    Lk = int(0.35 * SR)
                    tk = np.arange(Lk) / SR
                    fk = 45 + 75 * np.exp(-tk / 0.035)
                    add(drums, int(ts * SR), np.sin(2 * np.pi * np.cumsum(fk) / SR) * np.exp(-tk / 0.16) * 0.15)
                if k % 2 == 1 and (not build or ts > dur * 0.45):
                    Lh = int(0.08 * SR)
                    h = filt(rng.standard_normal(Lh), 7000, "high") * np.exp(-np.arange(Lh) / SR / 0.022)
                    add(drums, int(ts * SR), h * (0.04 if k != 3 else 0.055))
    for ch in range(2):                            # ping-pong delay on the arpeggio
        d = int(0.75 * beat * SR)
        other = arp[1 - ch].copy()
        arp[ch] += 0.28 * shift(other, d) + 0.1 * shift(arp[ch], 2 * d)
    mix = np.clip(np.arange(n) / SR / dur, 0, 1)
    bright = 0.25 + 0.75 * mix if build else np.full(n, 0.6)
    out = np.zeros((2, n))
    for ch in range(2):
        p = filt(pad[ch], 1000) * (1 - bright) + filt(pad[ch], 4500) * bright
        a = filt(arp[ch], 8000)
        ir_t = np.arange(int(3.0 * SR)) / SR
        ir = filt(rng.standard_normal(len(ir_t)), 5000) * np.exp(-ir_t / 0.45)
        ir[: int(0.02 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        wet = fftconvolve(p + a, ir)[:n]
        out[ch] = p + a + 0.3 * wet + bass + drums
    out = filt(out, 35, "high")
    t = np.arange(n) / SR
    out *= np.clip(t / 1.5, 0, 1) * np.clip((dur - t) / 3.5, 0, 1)
    return out / (np.max(np.abs(out)) + 1e-9) * 0.89


def write_wav(path, x):
    y = (np.clip(x.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


def duration(video):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]))


def mux(video, style):
    """Replace the video's audio with a synthesized track (video stream copied; -20 LUFS, AAC 160k)."""
    video = Path(video)
    wav = video.with_suffix(".music.wav")
    tmp = video.with_suffix(".tmp.mp4")
    write_wav(wav, music(duration(video), **STYLES[style]))
    tmp.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-af", "loudnorm=I=-20:TP=-2:LRA=11,aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo",
                    "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", str(tmp)], check=True)
    os.replace(tmp, video)
    wav.unlink()


if __name__ == "__main__":
    mux(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "story")
