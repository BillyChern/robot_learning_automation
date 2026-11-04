"""Client for openpi's websocket policy server (msgpack-numpy protocol, own implementation: works with numpy 2).
Protocol: on connect the server sends packed metadata; each infer sends one packed obs dict and receives one packed
dict (a text frame instead of bytes is the server's traceback). Obs for KinovaInputs: image, wrist_image uint8
224x224x3 RGB, state float32 (7,), prompt str.  CLI smoke test: python -m rlpj.vla.policy_client --host H --port P"""
import argparse
import functools
import logging
import time

import msgpack
import numpy as np

from .images import SIZE, bgr_to_model


def pack_array(obj):
    """msgpack default hook: ndarray / numpy scalar -> tagged dict (same layout as openpi_client.msgpack_numpy)."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    """msgpack object_hook: tagged dict -> ndarray / numpy scalar."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def make_obs(side_bgr, wrist_bgr, state, prompt, size=SIZE):
    """Camera BGR frames + state (7,) + instruction -> obs dict (images letterboxed to size x size RGB)."""
    return {"image": bgr_to_model(side_bgr, size), "wrist_image": bgr_to_model(wrist_bgr, size),
            "state": np.asarray(state, np.float32).reshape(7), "prompt": str(prompt)}


class PolicyClient:
    def __init__(self, host, port=8000, timeout=60.0, retry_s=2.0, log=logging.getLogger(__name__).info):
        """Connect (retrying until timeout) and read the server metadata."""
        self.uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self.timeout, self.log = timeout, log
        self._packer = Packer()
        self.ws = None
        self.metadata = {}
        self.last_ms = float("nan")
        t0 = time.time()
        while True:
            try:
                self._connect()
                break
            except (ConnectionRefusedError, OSError, TimeoutError) as e:
                if time.time() - t0 > timeout:
                    raise ConnectionError(f"policy server {self.uri} not reachable: {e}") from e
                log(f"waiting for policy server {self.uri} ({e})")
                time.sleep(retry_s)

    def _connect(self):
        import websockets.sync.client
        self.ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None, open_timeout=10)
        self.metadata = unpackb(self.ws.recv())
        self.log(f"policy server {self.uri}: metadata {self.metadata}")

    def infer(self, obs: dict) -> dict:
        """Send one obs, return the server's dict (actions (H, 7) float32, server_timing, policy_timing)."""
        t0 = time.perf_counter()
        self.ws.send(self._packer.pack(obs))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"policy server error:\n{resp}")
        out = unpackb(resp)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--prompt", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from rlpj import config
    cfg = config.load()
    host = a.host or cfg["hosts"]["policy_server"]
    port = a.port or cfg["hosts"]["policy_port"]
    prompt = a.prompt or cfg["task"]["instruction"]
    rng = np.random.default_rng(0)
    side = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    state = np.array([0, 0.45, -1.66, 0, -1.0, 0, 0], np.float32)
    with PolicyClient(host, port) as c:
        for i in range(a.n):
            out = c.infer(make_obs(side, wrist, state, prompt))
            acts = np.asarray(out["actions"])
            print(f"call {i}: actions {acts.shape} {acts.dtype} round-trip {c.last_ms:.0f} ms "
                  f"server infer {out.get('server_timing', {}).get('infer_ms', float('nan')):.0f} ms")
        print("first chunk (rad, gripper):\n", np.round(acts, 3))
    return 0


if __name__ == "__main__":
    main()
