"""Camera readers. Each camera can be opened by one process only: open it once and share frames via latest()."""
import os
import threading
import time
import warnings

import cv2
import numpy as np

from . import calib


class SideCam:
    """RealSense side_left: colour 1280x720 (BGR) + depth aligned to colour (metres, depth gain applied)."""

    def __init__(self, serial=None):
        import pyrealsense2 as rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial or calib.side_serial())
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale() * calib.side_depth_gain()
        self.align = rs.align(rs.stream.color)
        self.lock = threading.Lock()
        self.color = self.depth = None
        self.t, self.n = 0.0, 0
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while self.running:
            try:
                fs = self.align.process(self.pipe.wait_for_frames(500))
                c = np.asanyarray(fs.get_color_frame().get_data()).copy()
                d = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.scale
                with self.lock:
                    self.color, self.depth, self.t = c, d, time.time()
                    self.n += 1
            except Exception:
                time.sleep(0.05)

    def latest(self):
        with self.lock:
            return self.color, self.depth, self.t

    def depth_median(self, n=5, timeout=3.0):
        """Median of n fresh depth frames (less noise)."""
        ds, last, t0 = [], -1, time.time()
        while len(ds) < n and time.time() - t0 < timeout:
            with self.lock:
                if self.n != last and self.depth is not None:
                    ds.append(self.depth)
                    last = self.n
            time.sleep(0.01)
        D = np.stack(ds)
        D[D <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nan_to_num(np.nanmedian(D, 0), nan=0.0)

    def close(self):
        self.running = False
        try:
            self.th.join(2.0)
            self.pipe.stop()
        except Exception:
            pass


class WristCam:
    """Kinova vision module colour over RTSP (1280x720, BGR). The stream serves at most 2 clients; a crashed client
    keeps its slot ~40 s, a normal close frees it at once."""

    def __init__(self, ip=None):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.url = f"rtsp://{ip or calib.ip()}/color"
        self.lock = threading.Lock()
        self.color, self.t, self.n = None, 0.0, 0
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        cap = None
        while self.running:
            try:
                if cap is None:
                    cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, f = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue
                with self.lock:
                    self.color, self.t = f, time.time()
                    self.n += 1
            except Exception:
                cap = None
                time.sleep(0.5)
        if cap is not None:
            cap.release()

    def latest(self):
        with self.lock:
            return self.color, self.t

    def close(self):
        self.running = False
        self.th.join(3.0)
