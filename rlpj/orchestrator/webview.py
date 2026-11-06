"""Read-only web page for watching a run from elsewhere (e.g. over Tailscale): the latest dashboard frame (side + wrist
cameras with overlays, written by the live view while a robot program runs), the run status, cost/time totals and the
latest events. Reads files only: no camera, no robot, no state changes.

    python -m rlpj.orchestrator.webview <run> [--host 0.0.0.0] [--port 8091]"""
import argparse
import html
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rlpj import config
from . import state as S

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>rlpj {run}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{background:#111;color:#ddd;font:14px monospace;margin:12px}} img{{width:100%;max-width:1280px}}
pre{{white-space:pre-wrap;background:#1b1b1b;padding:8px}} a{{color:#8cf}}</style></head><body>
<div id="live">loading...</div><img id="frame" src="/stream.mjpg">
<p>full-rate live view while a robot program runs: <a id="lv" href="#">port 8090</a></p>
<pre id="status"></pre>
<script>
document.getElementById("lv").href = location.protocol + "//" + location.hostname + ":8090/";
async function status() {{
  const r = await fetch("/status"); const j = await r.json();
  document.getElementById("live").textContent = j.live;
  document.getElementById("status").textContent = j.text;
}}
setInterval(status, 5000); status();
</script></body></html>"""


def status(run):
    """{'live': one line about the camera frame, 'text': status table + the last events} for the page."""
    run_dir = config.RUNS / run
    meta = config.RUNS / "live" / "latest.json"
    try:
        m = json.loads(meta.read_text())
        age = time.time() - float(m["t"])
        where = str(m.get("attempt", "")).split("/agents/")[-1].split("/episodes/")[-1]
        live = (f"LIVE: {where} phase {m.get('phase')}" if age < 5 else
                f"no robot program running for {age / 60:.0f} min (agents thinking, training, or evaluating); last frame: {where}")
    except (OSError, ValueError, KeyError):
        live = "no frame yet"
    try:
        st = S.RunState.open(run)
        lines = S.status_lines(st, config.load(st.task))
    except Exception as e:
        lines = [f"status unavailable: {e!r}"]
    ev = S.read_jsonl(run_dir / "events.jsonl")[-15:]
    lines += ["", "latest events:"] + [
        "  " + " ".join(str(x) for x in (e.get("t", "")[11:], e.get("kind"), e.get("role") or "", e.get("action") or "",
                                         e.get("final") or e.get("outcome") or "", (e.get("what") or e.get("reason") or "")[:110]) if x)
        for e in reversed(ev)]
    return dict(live=live, text="\n".join(lines))


def handler(run):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.format(run=html.escape(run)).encode(), "text/html; charset=utf-8")
            elif path == "/latest.jpg":
                f = config.RUNS / "live" / "latest.jpg"
                self._send(200, f.read_bytes(), "image/jpeg") if f.exists() else self._send(404, b"no frame yet", "text/plain")
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/status":
                self._send(200, json.dumps(status(run)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        def _stream(self, fps=10.0, max_s=4 * 3600):
            """MJPEG of runs/live/latest.jpg: a part whenever the file changes (the live view writes ~8 frames/s) and
            the same frame again every second while it does not (browsers draw a part only once the next one starts)."""
            f = config.RUNS / "live" / "latest.jpg"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last, sent, t0 = None, 0.0, time.time()
            try:
                while time.time() - t0 < max_s:
                    try:
                        m = f.stat().st_mtime_ns
                    except OSError:
                        m = None
                    if m is not None and (m != last or time.time() - sent > 1.0):
                        last, sent, body = m, time.time(), f.read_bytes()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(body)).encode()
                                         + b"\r\n\r\n" + body + b"\r\n")
                        self.wfile.flush()
                    time.sleep(1.0 / fps)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="run id (runs/<run>)")
    p.add_argument("--host", default="0.0.0.0", help="address to listen on (e.g. the Tailscale address only)")
    p.add_argument("--port", type=int, default=8091)
    a = p.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), handler(a.run))
    print(f"webview: http://{a.host}:{a.port}/ for run {a.run}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
