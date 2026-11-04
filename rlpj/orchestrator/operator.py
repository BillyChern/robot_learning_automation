"""The operator in the loop: gate before robot motion, questions, manual resets, review queue and labels."""
import json
import select
import sys
from pathlib import Path

from .state import now_iso, update_meta


class Operator:
    """Plain input() interface. hands_free: the countdown starts without a keypress; yes: no prompt for dry steps."""

    def __init__(self, hands_free=False, yes=False, countdown_s=5):
        self.hands_free, self.yes, self.countdown_s = hands_free, yes or hands_free, countdown_s
        self.tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self.last = None     # go | refused | skipped (result of the last gate)

    def notify(self, text):
        print(f"== {text}", flush=True)

    def gate(self, what, moves_robot, command=None):
        """Approve one command. Robot motion: Enter arms a countdown (Enter = go now, n = refuse, s = skip)."""
        print(f"== {'ROBOT' if moves_robot else 'dry'}: {what}" + (f"\n   $ {command}" if command else ""), flush=True)
        if not moves_robot:
            if self.yes or not self.tty:
                self.last = "go"
                return True
            return self._answer(input("   run? [Enter=yes, n=no, s=skip] "))
        if not self.tty:
            print("   refused: robot motion needs an operator at a terminal", flush=True)
            self.last = "refused"
            return False
        if not self.hands_free and not self._answer(input("   Enter=arm the countdown, n=refuse, s=skip: ")):
            return False
        return self._countdown()

    def _answer(self, s):
        s = (s or "").strip().lower()
        self.last = "refused" if s.startswith("n") else "skipped" if s.startswith("s") else "go"
        return self.last == "go"

    def _countdown(self):
        for i in range(int(self.countdown_s), 0, -1):
            print(f"\r   starting in {i} s (Enter=now, n=refuse, s=skip) ", end="", flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                print()
                return self._answer(sys.stdin.readline())
        print("\r   go" + " " * 44, flush=True)
        self.last = "go"
        return True

    def ask(self, text, choices=None):
        """Question for the operator; with choices returns one of them (the first is the default), else free text."""
        if choices and (self.hands_free or not self.tty):
            print(f"== {text} -> {choices[0]} (auto)", flush=True)
            return choices[0]
        if not self.tty:
            return ""
        while True:
            a = input(f"== {text}" + (f" [{'/'.join(choices)}] " if choices else ": ")).strip()
            if not choices:
                return a
            if not a:
                return choices[0]
            hits = [c for c in choices if c.startswith(a.lower())]
            if len(hits) == 1:
                return hits[0]

    def wait_manual_reset(self, targets):
        """Ask the operator to place the objects by hand; True when done, False when impossible or stopped."""
        print("== manual reset: place the objects at (base frame, m)", flush=True)
        for k, xy in (targets or {}).items():
            print(f"   {k:<12} x={xy[0]:+.3f}  y={xy[1]:+.3f}", flush=True)
        if not self.tty:
            print("   no terminal: cannot wait for a manual reset", flush=True)
            return False
        return input("   Enter when done (q = stop): ").strip().lower() != "q"

    def queue_review(self, ep_dir, verdict):
        """Append an episode to runs/<run>/operator_queue.jsonl for `rlpj review`."""
        ep_dir = Path(ep_dir)
        rec = dict(t=now_iso(), episode=str(ep_dir), verdict={k: verdict.get(k) for k in ("final", "reason")})
        with open(ep_dir.parent.parent / "operator_queue.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"== queued for review: {ep_dir.name} ({rec['verdict']['final']})", flush=True)

    def label(self, ep_dir, label, note=""):
        """Operator verdict for an episode: operator_labels.jsonl and meta.json outcome; returns the previous final."""
        ep_dir = Path(ep_dir)
        rec = dict(t=now_iso(), episode=str(ep_dir), label=label, note=note)
        with open(ep_dir.parent.parent / "operator_labels.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        meta = ep_dir / "meta.json"
        prev = ((json.loads(meta.read_text()) if meta.exists() else {}).get("outcome") or {}).get("final")
        update_meta(ep_dir, outcome=dict(operator=label, final=label, reason=f"operator: {note}" if note else "operator label"))
        return prev
