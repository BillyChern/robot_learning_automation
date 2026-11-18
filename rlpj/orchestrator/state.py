"""Run state: runs/<run_id>/state.json (atomic writes), an append-only events.jsonl, status and report text."""
import json
import os
import threading
import time
from pathlib import Path

from rlpj import config


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_run_id(task_name):
    """20261104_1130_bee: date, time and the first word of the task name."""
    return time.strftime("%Y%m%d_%H%M") + "_" + str(task_name).split("_")[0]


def _counts():
    return dict(total=0, success=0, fail=0, unsure=0)


class RunState:
    """What the orchestrator knows about one run; the attributes in FIELDS are the keys of state.json."""

    FIELDS = ("run_id", "task", "condition", "created", "phase", "counters", "agents", "datasets", "training_runs",
              "evals", "deployable_checkpoint", "zero_shot", "collection", "next_episode", "next_reset", "vlm",
              "operator_notes", "rl")

    def __init__(self, run_id, task=None, condition=None):
        self.run_id, self.task, self.condition = run_id, task, condition
        self.created, self.phase = now_iso(), "new"
        self.counters = dict(episodes=_counts(), evals=_counts(), resets=dict(total=0, ok=0, fail=0, manual=0),
                             consecutive_failures=dict(execute=0, reset=0))
        self.agents, self.datasets, self.training_runs, self.evals = {}, {}, {}, {}
        self.deployable_checkpoint, self.zero_shot = None, None
        self.collection = dict(round=0, round_successes=0, round_target=None)
        self.next_episode, self.next_reset = 0, 0
        self.vlm = dict(calls=0, cost_usd=0.0)
        self.operator_notes = []
        self.rl = dict(episodes=0, evals=[], since_review=0, learner_fails=0)   # RL tasks (docs/RL_DESIGN.md)
        self.lock = threading.RLock()       # agent reflections run in threads and save too

    @property
    def dir(self) -> Path:
        return config.RUNS / self.run_id

    @property
    def file(self) -> Path:
        return self.dir / "state.json"

    @classmethod
    def create(cls, run_id, task, condition=None):
        st = cls(run_id, task, condition)
        if st.file.exists():
            raise FileExistsError(st.file)
        st.dir.mkdir(parents=True, exist_ok=True)
        st.save()
        st.event("created", task=task, condition=condition)
        return st

    @classmethod
    def open(cls, run_id):
        st = cls(run_id)
        if not st.file.exists():
            raise FileNotFoundError(f"no run {run_id} ({st.file})")
        d = json.loads(st.file.read_text())
        for k in cls.FIELDS:
            if k in d:
                setattr(st, k, d[k])
        return st

    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    def save(self):
        with self.lock:
            for _ in range(5):                          # another thread may add a key while we serialise: retry
                try:
                    text = json.dumps(self.to_dict(), indent=1, default=str)
                    break
                except RuntimeError:
                    time.sleep(0.01)
            tmp = self.file.with_name("state.json.tmp")
            tmp.write_text(text)
            os.replace(tmp, self.file)

    def event(self, kind, **data):
        with self.lock, open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(dict(t=now_iso(), kind=kind, **data), default=str) + "\n")

    def ledger(self, kind, **data):
        """Append one cost/time record (agent turn, VLM call, GPU job, evaluation) with the run totals to usage.jsonl."""
        with self.lock, open(self.dir / "usage.jsonl", "a") as f:
            f.write(json.dumps(dict(t=now_iso(), kind=kind, **data, totals=totals(self)), default=str) + "\n")

    def note(self, text):
        self.operator_notes.append(dict(t=now_iso(), text=text))
        self.save()

    def agent(self, role):
        """Record of one agent session (created on first use)."""
        return self.agents.setdefault(role, dict(
            status="new", session_id=None, model=None, started=None, ended=None, attempts=0, robot_attempts=0,
            turns=0, cost_usd=0.0, input_tokens=0, output_tokens=0, cache_read=0, cache_creation=0, duration_s=0.0,
            first_success_attempt=None, first_success_after_attempts=None, time_to_first_success_s=None,
            repairs=0, rounds=0, program_command=None, last_report=None))


def update_meta(ep_dir, **fields):
    """rlpj.robot.recorder.update_meta (dicts merged one level deep); same merge here if the recorder is not importable."""
    try:
        from rlpj.robot.recorder import update_meta as f
        return f(ep_dir, **fields)
    except ImportError:
        p = Path(ep_dir) / "meta.json"
        meta = json.loads(p.read_text()) if p.exists() else {}
        for k, v in fields.items():
            meta[k] = {**meta[k], **v} if isinstance(v, dict) and isinstance(meta.get(k), dict) else v
        p.write_text(json.dumps(meta, indent=1, default=str))
        return meta


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pending_reviews(st):
    """Queued episodes without an operator label."""
    done = {r["episode"] for r in read_jsonl(st.dir / "operator_labels.jsonl")}
    return [r for r in read_jsonl(st.dir / "operator_queue.jsonl") if r["episode"] not in done]


def _secs(iso):
    return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


def totals(st):
    """Run totals: Claude cost (agents + VLM judge) and tokens, agent time, robot time, GPU job time, wall clock."""
    ag = list(st.agents.values())
    agents_usd = sum(a.get("cost_usd", 0) for a in ag)
    robot_s = sum(float(e.get("duration") or 0) for e in read_jsonl(st.dir / "events.jsonl")
                  if e.get("kind") in ("reset", "episode") or (e.get("kind") == "attempt" and e.get("moves_robot")))
    gpu_s = sum(float(v.get("duration_s") or 0) for v in list(st.datasets.values()) + list(st.training_runs.values()))
    return dict(cost_usd=round(agents_usd + st.vlm["cost_usd"], 4), agents_usd=round(agents_usd, 4),
                vlm_usd=round(st.vlm["cost_usd"], 4), vlm_calls=st.vlm["calls"],
                tokens_in=sum(a.get("input_tokens", 0) for a in ag), tokens_out=sum(a.get("output_tokens", 0) for a in ag),
                cache_read=sum(a.get("cache_read", 0) for a in ag), cache_creation=sum(a.get("cache_creation", 0) for a in ag),
                agent_turns=sum(a.get("turns", 0) for a in ag), agent_min=round(sum(a.get("duration_s", 0) for a in ag) / 60, 1),
                robot_min=round(robot_s / 60, 1), gpu_min=round(gpu_s / 60, 1),
                wall_min=round((time.time() - _secs(st.created)) / 60, 1))


def totals_line(st):
    t = totals(st)
    return (f"cost ${t['cost_usd']:.2f} (agents ${t['agents_usd']:.2f}, judge ${t['vlm_usd']:.2f} / {t['vlm_calls']} calls)  "
            f"tokens {t['tokens_in'] / 1e3:.0f}k in / {t['tokens_out'] / 1e3:.0f}k out (+{t['cache_read'] / 1e6:.1f}M cache read, "
            f"+{t['cache_creation'] / 1e3:.0f}k cache write)  agent {t['agent_min']:.0f} min ({t['agent_turns']} turns)  "
            f"robot {t['robot_min']:.0f} min  gpu {t['gpu_min']:.0f} min  wall {t['wall_min'] / 60:.1f} h")


def agent_summary(role, a):
    first = (f"first success at robot attempt {a['first_success_attempt']} after {a['time_to_first_success_s']:.0f} s"
             if a.get("first_success_attempt") else "no verified success yet")
    return (f"{role:<8} {a.get('status', '-'):<13} runs {a.get('attempts', 0)} (robot {a.get('robot_attempts', 0)}), {first}, "
            f"turns {a.get('turns', 0)}, tokens {a.get('input_tokens', 0) / 1e3:.0f}k in / {a.get('output_tokens', 0) / 1e3:.0f}k out "
            f"(+{a.get('cache_read', 0) / 1e3:.0f}k cached), ${a.get('cost_usd', 0):.2f}, {a.get('duration_s', 0) / 60:.0f} min, "
            f"repairs {a.get('repairs', 0)}")


def status_lines(st, cfg=None):
    """Short text table of a run."""
    e, r, c = st.counters["episodes"], st.counters["resets"], st.collection
    rate = e["success"] / e["total"] if e["total"] else 0.0
    target = (cfg or {}).get("evaluation", {}).get("target_success_rate")
    lines = [f"run {st.run_id}  task {st.task}  condition {st.condition}  phase {st.phase}  created {st.created}",
             f"episodes {e['total']} (success {e['success']}, fail {e['fail']}, unsure {e['unsure']}; {rate:.0%}; "
             f"unsure deleted and redone {c.get('discarded', 0)}; dagger corrections "
             f"{st.counters.get('dagger', {}).get('success', 0)}/{st.counters.get('dagger', {}).get('total', 0)} success)  "
             f"eval episodes {st.counters['evals']['total']}  resets {r['total']} (ok {r['ok']}, fail {r['fail']}, manual {r['manual']})",
             f"collection round {c.get('round', 0)}: {c.get('round_successes', 0)}/{c.get('round_target') or '-'} successes  "
             f"vlm judge {st.vlm['calls']} calls ${st.vlm['cost_usd']:.2f}  operator queue {len(pending_reviews(st))} pending",
             totals_line(st),
             "agents:" if st.agents else "agents: none"]
    lines += [f"  {agent_summary(k, v)}" for k, v in st.agents.items()]
    lines.append("datasets: " + (", ".join(f"{k} (exit {v.get('exit_code')})" for k, v in st.datasets.items()) or "none"))
    lines.append("training runs: " + (", ".join(f"{k} (exit {v.get('exit_code')})" for k, v in st.training_runs.items()) or "none"))
    lines.append("evals:" if st.evals else "evals: none")
    for k, v in st.evals.items():
        lines.append(f"  {k}: {v.get('success', 0)}/{v.get('n', 0)} success ({v.get('success_rate', 0):.0%})"
                     + (f", target {target:.0%}" if target else "") + (" MET" if v.get("met") else "")
                     + (" [zero-shot]" if v.get("zero_shot") else "") + (f" error: {v['error']}" if v.get("error") else ""))
    lines.append(f"deployable checkpoint: {st.deployable_checkpoint or '-'}")
    return lines


def report_md(st, cfg=None):
    """Markdown summary of a run (REPORT.md)."""
    e, r = st.counters["episodes"], st.counters["resets"]
    rate = e["success"] / e["total"] if e["total"] else 0.0
    out = [f"# Run {st.run_id}", "", f"task `{st.task}`, condition `{st.condition}`, phase `{st.phase}`, created {st.created}", "",
           "## Cost and time", "", totals_line(st), "", "Per call: `usage.jsonl` (agent turns, VLM judge calls, GPU jobs, "
           "evaluations, each with the running totals).", "", "## Agents", "", "| role | status | runs | robot runs | first success (robot attempt) | time to first success | turns | "
           "tokens in / out / cached | cost | agent time | repairs |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in st.agents.items():
        tfs = a.get("time_to_first_success_s")
        out.append(f"| {k} | {a.get('status')} | {a.get('attempts', 0)} | {a.get('robot_attempts', 0)} | {a.get('first_success_attempt') or '-'} | "
                   f"{f'{tfs / 60:.1f} min' if tfs else '-'} | {a.get('turns', 0)} | {a.get('input_tokens', 0)} / {a.get('output_tokens', 0)} / "
                   f"{a.get('cache_read', 0)} | ${a.get('cost_usd', 0):.2f} | {a.get('duration_s', 0) / 60:.0f} min | {a.get('repairs', 0)} |")
    out += ["", "## Collection", "", f"- episodes: {e['total']} (success {e['success']}, fail {e['fail']}, unsure {e['unsure']}; {rate:.0%})",
            f"- unsure episodes deleted and redone: {st.collection.get('discarded', 0)}",
            f"- resets: {r['total']} (ok {r['ok']}, fail {r['fail']}, manual {r['manual']})",
            f"- rounds: {st.collection.get('round', 0)}", f"- VLM judge: {st.vlm['calls']} calls, ${st.vlm['cost_usd']:.2f}",
            f"- operator review queue: {len(pending_reviews(st))} pending", "", "## Datasets and training", ""]
    out += [f"- dataset `{k}`: exit {v.get('exit_code')}, episodes {v.get('episodes')}" for k, v in st.datasets.items()] or ["- no datasets"]
    out += [f"- training run `{k}`: dataset {v.get('dataset')}, exit {v.get('exit_code')}, {v.get('duration_s', 0) / 60:.0f} min, `{v.get('command')}`"
            for k, v in st.training_runs.items()] or ["- no training runs"]
    out += ["", "## Evaluations", "", "| eval | checkpoint | episodes | success | rate | met | DAgger corrections (success/total) |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {k} | `{v.get('checkpoint')}` | {v.get('n', 0)} | {v.get('success', 0)} | {v.get('success_rate', 0):.0%} | "
            f"{'yes' if v.get('met') else 'no'}{' (zero-shot)' if v.get('zero_shot') else ''} | "
            f"{sum(d['final'] == 'success' for d in v.get('dagger', []))}/{len(v.get('dagger', []))} |" for k, v in st.evals.items()]
    d = st.counters.get("dagger")
    if d:
        out.append(f"\nDAgger corrections: {d['total']} (success {d['success']}, fail {d['fail']}, unsure {d['unsure']})")
    out += ["", f"deployable checkpoint: `{st.deployable_checkpoint or '-'}`"]
    if st.zero_shot:
        out.append(f"\nzero-shot: {st.zero_shot}")
    if st.operator_notes:
        out += ["", "## Operator notes", ""] + [f"- {n['t']}: {n['text']}" for n in st.operator_notes]
    return "\n".join(out) + "\n"
