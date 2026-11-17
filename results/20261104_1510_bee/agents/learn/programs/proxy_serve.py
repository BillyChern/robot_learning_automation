"""Policy 'server' for rlpj eval that forwards TCP 0.0.0.0:<port> on the robot machine to the openpi server already
running on the 5090 (started by programs/serve_5090.sh for the SAME checkpoint). Workaround for the local 4090 server
not coming up (evals 01, 02). Arguments as required by serve.sh: --checkpoint --port --config. Stdlib only."""
import argparse
import socket
import sys
import threading
import time

UPSTREAM = ("100.122.29.80", 8000)


def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default="")
    a = p.parse_args()
    print(f"[proxy] checkpoint {a.checkpoint} (served upstream at {UPSTREAM}), config {a.config}", flush=True)
    for i in range(60):                               # upstream must answer before we listen
        try:
            socket.create_connection(UPSTREAM, timeout=3).close()
            break
        except OSError as e:
            print(f"[proxy] upstream not reachable ({e}), retry {i}", flush=True)
            time.sleep(5)
    else:
        sys.exit(1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(8)
    print(f"[proxy] listening on {a.port}", flush=True)
    while True:
        c, addr = srv.accept()
        try:
            u = socket.create_connection(UPSTREAM, timeout=10)
            u.settimeout(None)
        except OSError as e:
            print(f"[proxy] upstream connect failed: {e}", flush=True)
            c.close()
            continue
        for s in (c, u):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[proxy] client {addr}", flush=True)
        threading.Thread(target=pipe, args=(c, u), daemon=True).start()
        threading.Thread(target=pipe, args=(u, c), daemon=True).start()


if __name__ == "__main__":
    main()
