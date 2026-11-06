"""Live web view (port 8090): both cameras at native 1280x720 (MJPEG, JPEG q=90) with overlays of the scene the
program perceives (objects, targets, planned path, gripper from the measured pose), an HTML status panel and a
recording to <attempt>/dashboard.mp4. Own threads at ~8 fps; every failure inside is caught, never raised."""
import collections
import json
import os
import socket
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from . import calib

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>rlpj live</title>
<style>body{font-family:monospace;background:#111;color:#ddd;margin:8px} img{width:49%;border:1px solid #444}
#st{white-space:pre-wrap;font-size:15px;background:#222;padding:8px} .big{font-size:22px;color:#ffd060}</style></head>
<body><div><img src="/side.mjpg"><img src="/wrist.mjpg"></div><div id="hdr" class="big"></div><div id="st"></div>
<script>
async function poll(){try{const r=await fetch('/status.json');const s=await r.json();
document.getElementById('hdr').textContent=`${s.program} | PHASE ${s.phase} | segment ${s.segment} | progress ${(100*s.progress).toFixed(0)}% | dev ${s.dev_deg.toFixed(2)} deg / ${s.dev_mm.toFixed(1)} mm | gripper ${s.gripper.toFixed(1)}`;
document.getElementById('st').textContent='decisions:\\n'+s.decisions.join('\\n')+'\\n\\nlog:\\n'+s.log.join('\\n')+'\\n\\njoints: '+s.q+'\\ntool: '+s.pose;
}catch(e){} setTimeout(poll,300);} poll();
</script></body></html>"""

TIPS = np.array([[0.0425, 0, 0.027], [-0.0425, 0, 0.027], [0, 0, 0]])   # open fingertips and tool point, tool frame


def addresses():
    """(lan_ip, tailscale_ip) of this machine, '?' when unknown; never raises."""
    lan = ts = "?"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); lan = s.getsockname()[0]; s.close()
    except Exception:
        pass
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show", "dev", "tailscale0"], capture_output=True, text=True, timeout=2).stdout
        ts = out.split("inet ")[1].split("/")[0] if "inet " in out else "?"
    except Exception:
        pass
    return lan, ts


class LiveView:
    def __init__(self, attempt_dir, side=None, wrist=None, kin=None, port=8090, fps=8.0, program=""):
        self.A, self.side, self.wrist, self.kin, self.port, self.fps = attempt_dir, side, wrist, kin, port, fps
        self.lock = threading.Lock()
        self.st = dict(program=program, phase="init", segment="-", progress=0.0, dev_deg=0.0, dev_mm=0.0,
                       gripper=float("nan"), q="", pose="", decisions=[], log=[])
        self.logq = collections.deque(maxlen=18)
        self.decq = collections.deque(maxlen=10)
        self.objects, self.targets, self.path, self.T_base_tool = [], [], None, None
        self.jpeg = {"side": None, "wrist": None}
        self.running = True
        self.errors = 0
        self.writer = self.srv = self.loop_th = None
        self.url = None
        self.table_z = calib.table_z()
        try:
            self.T_side_base = np.linalg.inv(calib.T_base_side())
            self.Ks, self.ds = calib.side_intrinsics()
            self.Kw, self.dw = calib.wrist_intrinsics()
            self.T_tool_cam = calib.T_tool_cam()
            self.proj_ok = True
        except Exception as e:
            print("live view: calibration not loaded, overlays off:", e)
            self.proj_ok = False

    # ---------- API used by the program (cheap, non-blocking) ----------
    def set(self, **kw):
        with self.lock:
            self.st.update(kw)

    def log(self, msg):
        with self.lock:
            self.logq.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def decide(self, msg):
        with self.lock:
            self.decq.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def set_scene(self, objects=None, targets=None, path=None):
        """objects: [{xy, z_top, r, label}] (base frame, m); targets: same dicts or [(name, xyz)]; path: Nx3 tool positions."""
        with self.lock:
            if objects is not None:
                self.objects = list(objects)
            if targets is not None:
                self.targets = list(targets)
            if path is not None:
                self.path = np.asarray(path, float).reshape(-1, 3)

    def set_robot(self, q, pose, gripper):
        """q: measured joints; pose: measured tool_pose (Kortex)."""
        with self.lock:
            try:
                self.T_base_tool = calib.pose_to_T(np.asarray(pose, float))
            except Exception:
                pass
            self.st["q"] = str(np.round(q, 1).tolist()); self.st["pose"] = str(np.round(pose, 3).tolist())
            self.st["gripper"] = float(gripper)

    # ---------- threads ----------
    def start(self):
        try:
            self._serve()
        except Exception as e:
            print("live view server failed (run continues):", e)
        self.loop_th = threading.Thread(target=self._loop, daemon=True); self.loop_th.start()
        return self

    def _serve(self):
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                self.send_response(200); self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                try:
                    if self.path in ("/", "/index.html"):
                        self._send(PAGE.encode(), "text/html")
                    elif self.path == "/status.json":
                        with view.lock:
                            s = dict(view.st); s["log"] = list(view.logq); s["decisions"] = list(view.decq)
                        self._send(json.dumps(s, default=str).encode(), "application/json")
                    elif self.path in ("/side.mjpg", "/wrist.mjpg"):
                        key = self.path[1:-5]
                        self.send_response(200)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
                        last = None
                        while view.running:
                            j = view.jpeg[key]
                            if j is not None and j is not last:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                                last = j
                            time.sleep(0.05)
                    else:
                        self.send_response(404); self.end_headers()
                except Exception:
                    pass

        ThreadingHTTPServer.daemon_threads = True
        ThreadingHTTPServer.allow_reuse_address = True
        self.srv = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        lan, ts = addresses()
        self.url = f"http://localhost:{self.port}"
        print(f"LIVE VIEW: {self.url}  http://{lan}:{self.port}  http://{ts}:{self.port}", flush=True)

    def _proj_side(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        Pc = calib.apply(self.T_side_base, P)
        uv = calib.project(Pc, self.Ks, self.ds)
        uv[Pc[:, 2] < 0.05] = np.nan
        return uv

    def _proj_wrist(self, P):
        if self.T_base_tool is None:
            return None
        T_cam_base = np.linalg.inv(self.T_base_tool @ self.T_tool_cam)
        Pc = calib.apply(T_cam_base, np.asarray(P, float).reshape(-1, 3))
        uv = calib.project(Pc, self.Kw, self.dw)
        uv[Pc[:, 2] < 0.03] = np.nan
        return uv

    def _draw(self, img, proj):
        if not self.proj_ok:
            return img
        with self.lock:
            objects, targets, path, T = list(self.objects), list(self.targets), self.path, self.T_base_tool

        def pt(u):
            return None if u is None or not np.all(np.isfinite(u)) or abs(u[0]) > 5000 or abs(u[1]) > 5000 else (int(u[0]), int(u[1]))

        a = np.linspace(0, 2 * np.pi, 32)
        for k, o in enumerate(objects):   # circle at the object top and at the table, with label
            col = [(0, 200, 0), (255, 120, 0), (200, 200, 0), (255, 0, 200), (0, 200, 255)][k % 5]
            xy, r = o["xy"], float(o.get("r", 0.03))
            for z in (float(o.get("z_top", self.table_z + 0.03)), self.table_z):
                P = np.stack([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), np.full(32, z)], 1)
                uv = proj(P)
                if uv is not None and np.all(np.isfinite(uv)):
                    cv2.polylines(img, [uv.astype(np.int32)], True, col, 2)
            u = proj([[xy[0], xy[1], float(o.get("z_top", self.table_z)) + 0.02]])
            p = pt(u[0]) if u is not None else None
            if p:
                cv2.putText(img, str(o.get("label", f"obj{k}")), p, 0, 0.9, col, 2)
        if path is not None and len(path) > 1:
            uv = proj(path)
            if uv is not None:
                ok = np.all(np.isfinite(uv), 1)
                if ok.sum() > 1:
                    cv2.polylines(img, [uv[ok].astype(np.int32)], False, (255, 0, 255), 2)
        for t in targets:        # dicts like objects (xy, z_top, label) or (name, xyz) pairs
            name, xyz = ((t.get("label", "target"), [t["xy"][0], t["xy"][1], float(t.get("z_top", self.table_z))])
                         if isinstance(t, dict) else t)
            u = proj([xyz]); p = pt(u[0]) if u is not None else None
            if p:
                cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                cv2.putText(img, str(name), (p[0] + 8, p[1] - 8), 0, 0.6, (0, 0, 255), 2)
        if T is not None:   # gripper: tool point and fingertips from the measured pose
            u = proj(calib.apply(T, TIPS))
            if u is not None:
                for k in range(3):
                    p = pt(u[k])
                    if p:
                        cv2.circle(img, p, 7 if k < 2 else 10, (0, 255, 255), 2)
        return img

    def _loop(self):
        period = 1.0 / self.fps
        while self.running:
            t0 = time.time()
            try:
                frames = {}
                if self.side is not None:
                    c = self.side.latest()[0]
                    if c is not None:
                        frames["side"] = self._draw(c.copy(), self._proj_side)
                if self.wrist is not None:
                    c = self.wrist.latest()[0]
                    if c is not None:
                        frames["wrist"] = self._draw(c.copy(), self._proj_wrist)
                for k, im in frames.items():
                    ok, j = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        self.jpeg[k] = j.tobytes()
                self._record(frames)
            except Exception:
                self.errors += 1
                if self.errors < 5:
                    traceback.print_exc()
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _record(self, frames):
        tiles = []
        for k in ("side", "wrist"):
            im = frames.get(k)
            tiles.append(cv2.resize(im, (640, 360)) if im is not None else np.zeros((360, 640, 3), np.uint8))
        bar = np.zeros((120, 1280, 3), np.uint8)
        with self.lock:
            s = self.st
            lines = [f"{s['phase']} | {s['segment']} {100 * s['progress']:.0f}% | dev {s['dev_deg']:.2f}deg {s['dev_mm']:.1f}mm | gripper {s['gripper']:.1f}"]
            lines += list(self.logq)[-3:]
        for i, l in enumerate(lines):
            cv2.putText(bar, l[:130], (8, 25 + 30 * i), 0, 0.6, (230, 230, 230), 1)
        fr = np.vstack([np.hstack(tiles), bar])
        if self.writer is None:
            self.writer = cv2.VideoWriter(os.path.join(self.A, "dashboard.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (1280, 480))
        self.writer.write(fr)
        self._publish(fr)

    def _publish(self, fr, every_s=0.1):
        """Latest dashboard frame for the read-only web viewer (rlpj.orchestrator.webview): runs/live/latest.jpg + .json."""
        now = time.time()
        if now - getattr(self, "_pub_t", 0.0) < every_s:
            return
        self._pub_t = now
        from rlpj import config
        d = config.RUNS / "live"
        d.mkdir(parents=True, exist_ok=True)
        ok, j = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            (d / "latest.jpg.tmp").write_bytes(j.tobytes())
            os.replace(d / "latest.jpg.tmp", d / "latest.jpg")
            (d / "latest.json.tmp").write_text(json.dumps(dict(t=now, attempt=str(self.A), program=getattr(self, "program", ""),
                                                                 phase=self.st.get("phase"))))
            os.replace(d / "latest.json.tmp", d / "latest.json")

    def snapshot(self, prefix):
        """Save the current overlaid frames as <prefix>_side.jpg / <prefix>_wrist.jpg (640x360)."""
        for k, j in self.jpeg.items():
            if j is not None:
                im = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(self.A, f"{prefix}_{k}.jpg"), cv2.resize(im, (640, 360)))

    def close(self):
        self.running = False
        try:
            self.loop_th and self.loop_th.join(3.0)
        except Exception:
            pass
        try:
            if self.writer is not None and not (self.loop_th and self.loop_th.is_alive()):
                self.writer.release()
        except Exception:
            pass
        try:
            self.srv and self.srv.shutdown()
        except Exception:
            pass
