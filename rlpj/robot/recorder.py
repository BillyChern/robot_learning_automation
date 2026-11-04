"""Episode recorder (DESIGN.md section 4): samples the session at recording.fps in its own thread and writes
side.mp4, wrist.mp4, steps.npz, phases.json, meta.json, before/after jpgs, summary.json and log.txt copies.
It never raises into the control program and never blocks it (cheap UDP feedback read, no waits)."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

import cv2
import numpy as np

from rlpj import config

KEYS = ("t", "q", "dq", "tool_pose", "gripper", "q_cmd", "gripper_cmd", "phase")


def update_meta(ep_dir, **fields):
    """Merge fields into <ep_dir>/meta.json (dict values such as outcome are merged one level deep)."""
    p = os.path.join(ep_dir, "meta.json")
    meta = json.load(open(p)) if os.path.exists(p) else {}
    for k, v in fields.items():
        if isinstance(v, dict) and isinstance(meta.get(k), dict):
            meta[k] = {**meta[k], **v}
        else:
            meta[k] = v
    with open(p, "w") as f:
        json.dump(meta, f, indent=1, default=_js)
    return meta


def _js(o):
    return o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating) else str(o)


def _run_dirs(ep_dir):
    """(run_id, episode) from runs/<run_id>/episodes/<episode>; '' when the layout differs."""
    ep = os.path.abspath(ep_dir)
    parent = os.path.dirname(ep)
    run_id = os.path.basename(os.path.dirname(parent)) if os.path.basename(parent) == "episodes" else ""
    return run_id, os.path.basename(ep)


class EpisodeRecorder:
    """Reads session.feedback(), session.executor (running, last['q_ref']), session.robot.gripper_cmd,
    session.phase_idx / session.phases, session.side / session.wrist (latest frames), session.outcome."""

    def __init__(self, session, ep_dir, fps=None, size=None, info=None, h264=True):
        rec = config.load().get("recording", {})
        self.s, self.dir = session, str(ep_dir)
        self.fps = float(fps or rec.get("fps", 15))
        self.size = tuple(size or (rec.get("width", 640), rec.get("height", 360)))
        self.info, self.h264 = dict(info or {}), h264
        os.makedirs(self.dir, exist_ok=True)
        self.rows = {k: [] for k in KEYS}
        self.lock = threading.Lock()
        self.writers = {}
        self.running, self.closed = False, False
        self.t_start = self.t_end = None
        self.n, self.errors = 0, 0
        self.th = None

    # ---------- lifecycle ----------
    def start(self):
        self.t_start = time.time()
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True, name="recorder")
        self.th.start()
        return self

    def snapshot(self, prefix):
        """<prefix>_side.jpg and <prefix>_wrist.jpg at 640 px from the raw camera frames."""
        for name, cam in (("side", getattr(self.s, "side", None)), ("wrist", getattr(self.s, "wrist", None))):
            try:
                im = cam.latest()[0] if cam is not None else None
                if im is not None:
                    h = int(round(640 * im.shape[0] / im.shape[1]))
                    cv2.imwrite(os.path.join(self.dir, f"{prefix}_{name}.jpg"), cv2.resize(im, (640, h), interpolation=cv2.INTER_AREA))
            except Exception:
                pass

    def close(self, **meta):
        """Stop sampling and write every file; extra meta fields are merged into meta.json."""
        if self.closed:
            return
        self.closed = True
        self.running = False
        if self.th is not None:
            self.th.join(max(2.0, 3.0 / self.fps))
        self.t_end = time.time()
        for w in self.writers.values():
            try:
                w.release()
            except Exception:
                pass
        try:
            self._write(meta)
        except Exception:
            traceback.print_exc()
        if self.h264:
            for name in ("side", "wrist"):
                self._reencode(os.path.join(self.dir, f"{name}.mp4"))

    # ---------- sampling ----------
    def _loop(self):
        period = 1.0 / self.fps
        nxt = time.time()
        while self.running:
            try:
                self._tick()
            except Exception:
                self.errors += 1
                if self.errors <= 3:
                    traceback.print_exc()
            nxt += period
            now = time.time()
            if nxt < now - period:   # fell behind: resynchronise instead of bursting
                nxt = now + period
            time.sleep(max(0.0, nxt - now))

    def _tick(self):
        s = self.s
        fb = s.feedback()
        t = time.time()
        q = np.asarray(fb["q"], float)
        ex = getattr(s, "executor", None)
        q_cmd = q
        if ex is not None and getattr(ex, "running", False):
            qr = ex.last.get("q_ref") if isinstance(ex.last, dict) else None
            if qr is not None:
                q_cmd = np.asarray(qr, float)
        g_cmd = float(getattr(getattr(s, "robot", None), "gripper_cmd", float("nan")))
        if not np.isfinite(g_cmd):
            g_cmd = float(fb["gripper"]) / 100.0 if np.isfinite(fb["gripper"]) else float("nan")
        frames = {}
        for name, cam in (("side", getattr(s, "side", None)), ("wrist", getattr(s, "wrist", None))):
            im = cam.latest()[0] if cam is not None else None
            frames[name] = im
        with self.lock:
            self.rows["t"].append(t - self.t_start)
            self.rows["q"].append(q)
            self.rows["dq"].append(np.asarray(fb["dq"], float))
            self.rows["tool_pose"].append(np.asarray(fb["pose"], float))
            self.rows["gripper"].append(float(fb["gripper"]))
            self.rows["q_cmd"].append(q_cmd)
            self.rows["gripper_cmd"].append(g_cmd)
            self.rows["phase"].append(int(getattr(s, "phase_idx", 0)))
            for name, im in frames.items():
                self._write_frame(name, im)
            self.n += 1

    def _write_frame(self, name, im):
        w = self.writers.get(name)
        if w is None:
            w = cv2.VideoWriter(os.path.join(self.dir, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            self.writers[name] = w
        if im is None:
            fr = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        else:
            fr = cv2.resize(im, self.size, interpolation=cv2.INTER_AREA) if im.shape[1::-1] != self.size else im
        w.write(fr)

    # ---------- output ----------
    def phases(self):
        """[{"name", "i0", "i1"}] from the per-tick phase indices and the session's phase names."""
        names = list(getattr(self.s, "phases", []) or [])
        with self.lock:
            ph = np.asarray(self.rows["phase"], int)
        out = []
        for k, name in enumerate(names):
            idx = np.where(ph == k)[0]
            if len(idx):
                out.append(dict(name=name, i0=int(idx[0]), i1=int(idx[-1] + 1)))
            else:   # phase shorter than one tick: zero-length range at the next tick
                nxt = np.where(ph > k)[0]
                i = int(nxt[0]) if len(nxt) else len(ph)
                out.append(dict(name=name, i0=i, i1=i))
        return out

    def _write(self, extra):
        with self.lock:
            rows = {k: np.asarray(v) for k, v in self.rows.items()}
        n = len(rows["t"])
        arrays = dict(t=rows["t"].astype(np.float64), q=rows["q"].reshape(n, 6), dq=rows["dq"].reshape(n, 6),
                      tool_pose=rows["tool_pose"].reshape(n, 6), gripper=rows["gripper"].astype(np.float64),
                      q_cmd=rows["q_cmd"].reshape(n, 6), gripper_cmd=rows["gripper_cmd"].astype(np.float64),
                      phase=rows["phase"].astype(np.int64))
        np.savez_compressed(os.path.join(self.dir, "steps.npz"), **arrays)
        with open(os.path.join(self.dir, "phases.json"), "w") as f:
            json.dump(self.phases(), f, indent=1)
        s = self.s
        rob = {}
        task = {}
        try:
            rob = config.robot()
            task = config.load().get("task", {})
        except Exception:
            pass
        hashes = {}
        try:
            from . import calib
            hashes = calib.file_hashes()
        except Exception:
            pass
        run_id, episode = _run_dirs(self.dir)
        meta = dict(episode=episode, run_id=run_id, task=task.get("name", ""), instruction=task.get("instruction", ""),
                    mode=getattr(s, "mode", "collect"),
                    program=dict(name=getattr(s, "program", ""), commit=getattr(s, "commit", ""), command=" ".join(sys.argv)),
                    robot=dict(name=rob.get("name", ""), ip=rob.get("ip", "")), fps=self.fps,
                    layout_target=None, layout_before=None, layout_after=None,
                    t_start=self.t_start, t_end=self.t_end, n_steps=n,
                    outcome=dict(claim=getattr(s, "outcome", None), perception=None, vlm=None, operator=None, final=None, reason=None),
                    calibration_hashes=hashes)
        meta.update(self.info)
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(meta.get(k), dict):
                meta[k] = {**meta[k], **v}
            else:
                meta[k] = v
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=_js)
        A = getattr(s, "A", None)
        if A:
            for name in ("summary.json", "log.txt"):
                p = os.path.join(A, name)
                if os.path.exists(p):
                    shutil.copy(p, os.path.join(self.dir, name))

    def _reencode(self, path):
        """Re-encode an mp4v file to H.264 with ffmpeg when available; keep the original on any problem."""
        if not os.path.exists(path) or shutil.which("ffmpeg") is None:
            return
        tmp = path[:-4] + "_h264.mp4"
        try:
            n0 = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "18", "-pix_fmt", "yuv420p", tmp], check=True, timeout=180, capture_output=True)
            n1 = int(cv2.VideoCapture(tmp).get(cv2.CAP_PROP_FRAME_COUNT))
            if n1 == n0 and n0 > 0:
                os.replace(tmp, path)
            else:
                os.remove(tmp)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
