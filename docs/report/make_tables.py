"""Tables and key numbers for docs/report/drawer_lying_report.md from a run's records (reads runs/, writes only docs/report/).

    python3 docs/report/make_tables.py [run_id] [--from 17] [--until N] [--update] [--figures]

Prints the Markdown blocks to stdout (--until N: only episodes up to index N, for a consistent snapshot of a live run;
resets, cost and the figures are not cut). --update replaces each block between <!-- BEGIN:name --> and <!-- END:name --> in the
report; --figures copies the run's plots and a few stills to docs/report/figures/ (runs/ is not in git).
Definitions: rim touches = force stops with the drawer bottom above 80 mm (insert.contact_map; episodes 0-4 have no map
and all their stops were on the rim); floor stop = a stop below 30 mm (the drawer seated on the compartment floor);
peak = insert.peak (the last insertion step's peak, 25 = a force stop); max force = the largest contact force over the
insertion segments in the episode log (the stop value when a stop fired); fixed return = scripts/rl_plots.py fixed_return.
"""
import ast
import json
import math
import re
import shutil
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/report/drawer_lying_report.md"
FIG = ROOT / "docs/report/figures"
SIM = ROOT / "sim/drawer_insert/results/lying"
RIM_TOUCH_MM, FLOOR_MM = 80.0, 30.0
STILLS = {"jam_ep0_side.jpg": "agents/execute/attempts/attempt_12/side_inserted.jpg",
          "hover_view_attempt26.jpg": "agents/execute/attempts/attempt_26/wrist_align2.jpg",
          "opening_lookdown_attempt30.jpg": "agents/execute/attempts/attempt_30/wrist_opening.jpg",
          "inserted_attempt30_side.jpg": "agents/execute/attempts/attempt_30/side_inserted.jpg"}
PLOTS = ("return_fixed.png", "return.png", "success.png", "timing.png", "timeline.png")
ARM = {None: "pre-arm", "learned": "learned", "scripted": "scripted", "sim": "sim", "learned_eval": "learned (eval mode)"}


def jl(p):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def jload(p, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def wilson(k, n, z=1.96):
    """95 % Wilson score interval of k successes in n trials."""
    if not n:
        return None, None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def rate(k, n):
    lo, hi = wilson(k, n)
    return "-" if not n else f"{k}/{n} = {100 * k / n:.0f} % [{100 * lo:.0f}, {100 * hi:.0f}]"


def fixed_return(final, s):
    """scripts/rl_plots.py fixed_return: +10 success, -1 per force stop, -0.1 per N of peak above 15 N, -0.05 per step."""
    ins = s.get("insert") or {}
    return (10.0 * (final == "success") - 1.0 * (ins.get("contacts") or 0)
            - 0.1 * max(0.0, (ins.get("peak") or 0) - 15.0) - 0.05 * (ins.get("steps") or 0))


def max_force(log):
    """Largest contact force (N) over the insertion segments of an episode log (stop value when a stop fired)."""
    best = None
    for line in log.splitlines():
        m = re.search(r"exec \d+_(\w+): (\{.*\})\s*$", line)
        if not m or not m.group(1).startswith(("ins", "unload", "sim", "desc", "touch")):
            continue
        try:
            d = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            continue
        r = re.search(r"contact force ([\d.]+) N", d.get("reason", ""))
        for v in (d.get("max_force_n"), float(r.group(1)) if r else None):
            if v is not None:
                best = v if best is None else max(best, v)
    return best


def touches(s):
    """(rim touches, floor stops) of an episode, None when it never reached the insertion."""
    ins = s.get("insert") or {}
    if ins.get("contacts") is None:
        return None, None
    cm = ins.get("contact_map")
    if cm is None:
        return int(ins["contacts"]), 0
    return sum(c[3] > RIM_TOUCH_MM for c in cm), sum(c[3] < FLOOR_MM for c in cm)


def entry(s):
    """Offset (a, b) in mm, relative to the planned opening centre, at which the drawer passed into the opening."""
    e = s.get("entry") or {}
    if e.get("c"):
        return [1000 * e["c"][0], 1000 * e["c"][1]]
    for t in s.get("decisions") or []:
        m = re.search(r"at (?:offset )?a ([-\d.]+) mm, b ([-\d.]+) mm", t)
        if m:
            return [float(m.group(1)), float(m.group(2))]
    return None


def votes(reason):
    m = re.search(r"perception=(\w+), vlm=(\w+)", reason or "")
    sign = {"True": "+", "False": "-"}
    return "" if not m else f" (P{sign.get(m.group(1), '?')} V{sign.get(m.group(2), '?')})"


def load(run, until=None):
    d = ROOT / "runs" / run
    ev = jl(d / "events.jsonl")
    curve = {r["episode"]: r for r in jl(d / "rl_curve.jsonl") if r.get("kind") == "train"}
    eps = []
    for e in ev:
        if e.get("kind") != "episode":
            continue
        i = int(e["episode"])
        if until is not None and i > until:
            continue
        ed = d / "episodes" / f"ep_{i:06d}"
        s, meta = jload(ed / "summary.json", {}), jload(ed / "meta.json", {})
        log = (ed / "log.txt").read_text(errors="replace") if (ed / "log.txt").exists() else ""
        rim, floor = touches(s)
        ins = s.get("insert") or {}
        t0 = meta.get("t_start")
        eps.append(dict(ep=i, mode=e.get("mode"), final=e.get("final"), reason=e.get("reason"), dur=e.get("duration"),
                        start=time.strftime("%H:%M", time.localtime(t0)) if t0 else e["t"][11:16], end=e["t"],
                        arm=ARM.get(s.get("arm"), s.get("arm")), rev=s.get("scripted_rev"), rim=rim, floor=floor,
                        stops=ins.get("contacts"), peak=ins.get("peak"), steps=ins.get("steps"), seat=ins.get("end"),
                        maxf=max_force(log), entry=entry(s), ret=(curve.get(i) or {}).get("reward", s.get("return")),
                        fixed=fixed_return(e.get("final"), s), cmap=ins.get("contact_map") or []))
    return d, ev, eps


def block_episodes(eps):
    rows = []
    for e in eps:
        en = "-" if not e["entry"] else f"{e['entry'][0]:+.1f}, {e['entry'][1]:+.1f}"
        mode = "" if e["mode"] == "rl" else " [eval]"
        rows.append([e["ep"], e["start"], e["arm"] + mode, e["final"] + votes(e["reason"]),
                     "-" if e["rim"] is None else f"{e['rim']} + {e['floor']}", f(e["peak"]), f(e["maxf"]),
                     "-" if e["steps"] is None else e["steps"], en, f(e["dur"], 0), f(e["ret"], 2), f(e["fixed"], 2)])
    return table(["Ep", "Start", "Arm", "Verdict (perception, VLM)", "Rim touches + floor stop", "Peak N",
                  "Max N", "Steps", "Entry a, b (mm)", "Duration (s)", "Agent return", "Fixed return"], rows)


def arm_rows(eps):
    rows = []
    for arm in ("pre-arm", "learned", "learned (eval mode)", "scripted", "sim"):
        g = [e for e in eps if e["arm"] == arm]
        if not g:
            continue
        k = sum(e["final"] == "success" for e in g)
        ins = [e for e in g if e["rim"] is not None]

        def ms(key, nd=1, src=ins):
            v = [e[key] for e in src if e[key] is not None]
            return "-" if not v else (f"{st.mean(v):.{nd}f}" + (f" ± {st.stdev(v):.{nd}f}" if len(v) > 1 else ""))
        rows.append([arm, len(g), rate(k, len(g)), ms("rim"), ms("floor"), ms("maxf"), ms("steps"), ms("dur", 0, g),
                     ms("fixed", 2, g)])
    return rows


def block_arms(eps, start):
    head = ["Arm", "Episodes", "Success [95 % Wilson]", "Rim touches", "Floor stops", "Max N", "Steps", "Duration (s)",
            "Fixed return"]
    train = [e for e in eps if e["mode"] == "rl"]
    win = [e for e in train if e["ep"] >= start]
    out = [f"All training episodes ({len(train)}):", "", table(head, arm_rows(train)), "",
           f"Comparison window, training episodes >= {start} ({len(win)} so far):", ""]
    out.append(table(head, arm_rows(win)) if win else "[TBD: no episodes in the window yet]")
    return "\n".join(out)


def block_eval(d, eps):
    ev = [r for r in jl(d / "rl_curve.jsonl") if r.get("kind") == "eval"]
    if not ev:
        return "[TBD: the fixed evaluation (10 deterministic episodes after 50 training episodes) has not run yet]"
    rows = [[r["after"], r["n"], rate(int(r["success"]), int(r["n"]))] for r in ev]
    out = [table(["After training episodes", "Episodes", "Success [95 % Wilson]"], rows)]
    ee = [e for e in eps if e["mode"] == "rl_eval"]
    if ee:
        out += ["", block_episodes(ee)]
    return "\n".join(out)


def block_resets(d, ev):
    rs = [e for e in ev if e.get("kind") == "reset"]
    if not rs:
        return "no resets yet"
    fails = []
    for e in rs:
        if e.get("ok"):
            continue
        r = jload(d / "resets" / f"reset_{int(e['reset']):04d}.json", {})
        big = [b for b in r.get("measured", []) if max(b.get("extent", [0])) > 0.35]
        why = ("drawer placed, but merged with the cabinet in the side-camera blobs" if big
               else "drawer not taken out (program reported fail)" if r.get("claimed_outcome") == "fail"
               else (r.get("report") or {}).get("reason", "?"))
        fails.append([e["reset"], e["t"][11:16], why])
    short = [e["duration"] for e in rs if e["duration"] < 90]
    long_ = [e["duration"] for e in rs if e["duration"] >= 90]
    orient = [e for e in ev if e.get("kind") == "reset_orientation"]
    lines = [f"- Resets: {len(rs)}, layout check passed {sum(bool(e.get('ok')) for e in rs)}, failed {len(fails)}; "
             f"VLM orientation check passed {sum(bool(e.get('success')) for e in orient)} of {len(orient)}.",
             f"- Duration: {len(short)} short resets (drawer already standing near the target; median {st.median(short):.0f} s) "
             f"and {len(long_)} long ones (pull-out and placing; median {st.median(long_):.0f} s, max {max(long_):.0f} s)."
             if short and long_ else f"- Median duration {st.median([e['duration'] for e in rs]):.0f} s.", ""]
    return "\n".join(lines + [table(["Reset", "Time", "Why it failed"], fails)] if fails else lines[:-1])


def block_cost(d, ev):
    us = jl(d / "usage.jsonl")
    if not us:
        return "no usage yet"
    tot = us[-1].get("totals", {})
    rows = []
    for role in ("reset", "execute"):
        g = [u for u in us if u.get("kind") == "agent" and u.get("role") == role]
        rows.append([f"{role} agent", len(g), sum(u.get("turns") or 0 for u in g), f"{sum(u.get('duration_s') or 0 for u in g) / 60:.0f}",
                     f"{sum(u.get('output_tokens') or 0 for u in g) / 1000:.0f} k", f"{sum(u.get('cost_usd') or 0 for u in g):.2f}"])
    v = [u for u in us if u.get("kind") == "vlm"]
    rows.append(["VLM judge (episodes)", len(v), "-", f"{sum(u.get('duration_s') or 0 for u in v) / 60:.0f}", "-",
                 f"{sum(u.get('cost_usd') or 0 for u in v):.2f}"])
    rows.append(["total", "", tot.get("agent_turns", ""), f"{tot.get('agent_min', 0):.0f}", f"{tot.get('tokens_out', 0) / 1000:.0f} k",
                 f"{tot.get('cost_usd', 0):.2f}"])
    t_end = us[-1]["t"][11:16]
    return "\n".join([table(["Item", "Calls", "Turns", "Minutes", "Output tokens", "Cost (USD, API list price)"], rows), "",
                      f"As of {t_end}: wall time {tot.get('wall_min', 0) / 60:.1f} h since the run was created, robot programs "
                      f"{tot.get('robot_min', 0):.0f} min, agent turns {tot.get('agent_min', 0):.0f} min."])


def block_summary(d, ev, eps):
    tr = [e for e in eps if e["mode"] == "rl"]
    if not tr:
        return "no training episodes yet"
    k = sum(e["final"] == "success" for e in tr)
    un = [e["ep"] for e in tr if e["final"] == "unsure"]
    seated = [e["ep"] for e in tr if e["seat"] == "seated"]
    first = next((e for e in tr if e["final"] == "success"), None)
    since9 = [e for e in tr if e["ep"] >= 9]
    gaps = [(_t(b["end"]) - _t(a["end"])) / 60 for a, b in zip(tr, tr[1:])]
    out = [f"- Training episodes: {len(tr)} (indices {tr[0]['ep']}-{tr[-1]['ep']}); counted successes {k}; unsure {len(un)} "
           f"({', '.join(map(str, un)) or 'none'}); failures {sum(e['final'] == 'fail' for e in tr)}.",
           f"- Drawer seated at the compartment floor (program log): {len(seated)} episodes ({', '.join(map(str, seated))}).",
           f"- First counted success: episode {first['ep']} at {first['end'][11:16]}." if first else "- No counted success yet.",
           f"- Episodes >= 9: {rate(sum(e['final'] == 'success' for e in since9), len(since9))}.",
           f"- Time between consecutive training episodes: median {st.median(gaps):.1f} min, {sum(g > 15 for g in gaps)} gaps "
           f"over 15 min." if gaps else "", f"- Per arm: " + "; ".join(
               f"{a} {sum(e['final'] == 'success' for e in tr if e['arm'] == a)}/{sum(e['arm'] == a for e in tr)}"
               for a in ARM.values() if any(e["arm"] == a for e in tr)) + "."]
    return "\n".join(x for x in out if x)


def _t(iso):
    return time.mktime(time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S"))


def block_force(eps):
    """Sim-to-real check: lateral force at each rim touch vs the direction from the touch to where the drawer went in."""
    rows, allc = [], []
    for e in eps:
        if not e["entry"]:
            continue
        cs = []
        for c in e["cmap"]:
            if c[3] <= RIM_TOUCH_MM:
                continue
            da, db, fa, fb = e["entry"][0] - c[0], e["entry"][1] - c[1], c[4], c[5]
            n1, n2 = math.hypot(da, db), math.hypot(fa, fb)
            if n1 >= 1.0 and n2 >= 0.5:
                cs.append((da * fa + db * fb) / n1 / n2)
        if cs:
            allc += cs
            rows.append([e["ep"], len(cs), f"{min(cs):.2f}", f"{st.median(cs):.2f}",
                         f"{st.mean([c[4] for c in e['cmap'] if c[3] > RIM_TOUCH_MM]):+.1f}",
                         f"{st.mean([c[5] for c in e['cmap'] if c[3] > RIM_TOUCH_MM]):+.1f}"])
    if not allc:
        return "no rim touches before an entry yet"
    return "\n".join([table(["Ep", "Rim touches", "Min cos", "Median cos", "Mean f_a (N)", "Mean f_b (N)"], rows), "",
                      f"All: {len(allc)} rim touches, cosine > 0 in {sum(c > 0 for c in allc)}, median {st.median(allc):.2f}, "
                      f"> 0.9 in {sum(c > 0.9 for c in allc)}."])


def block_sim():
    names = {"carry": "carry (full ranges)", "carry_early": "carry, released when seated", "small": "small (opening +-3 mm / +-1 deg)",
             "big_open": "big_open (opening +-15 mm / +-5 deg)", "carry_descend8": "after 8 straight-descent steps",
             "carry_drift10": "after 10 drift steps"}
    stress = {"noisy_force": "force noise sd 3-5 N, bias +-6 N", "tight": "clearance 1-2 mm", "poor_est": "in-hand estimate sd 5-10 mm, 3-6 deg",
              "no_tilt_est": "tilt estimate passed as zero", "high_friction": "friction 0.6-0.9", "loose_grip": "loose grasp",
              "soft_arm": "arm 5e3-2e4 N/m", "slow_steps": "decision period 0.5-1.0 s", "low_stop": "force stop 25-40 N"}
    ctl = ("straight", "spiral", "guided", "policy")
    head = ["Start / variant", "Straight descent", "Spiral search", "Force-guided script", "PPO policy"]

    def rows(m):
        out = []
        for k, lab in m.items():
            r = jload(SIM / f"eval_{k}.json")
            if r:
                out.append([lab] + [f"{100 * r[c]['summary']['success']:.1f} %" for c in ctl])
        return out
    nv = jload(SIM / "newton_validate.json", {})
    car = (jload(SIM / "eval_carry.json") or {}).get("policy", {}).get("summary", {})
    return "\n".join([f"500 episodes per cell (MuJoCo CPU reference, seeds 100000-100499, same episodes for every controller).", "",
                      table(head, rows(names)), "", "Stress tests (carry start):", "", table(head, rows(stress)), "",
                      f"Policy on carry starts: median true peak force {car.get('max_f_median', 0):.0f} N, "
                      f"{car.get('stops_mean', 0):.1f} force stops per episode, median {car.get('first_success_median', 0):.0f} "
                      f"decisions to the first success state.", "",
                      f"Newton vs MuJoCo (straight descent, {nv.get('n')} carry episodes): success {100 * nv.get('success_newton', 0):.1f} % vs "
                      f"{100 * nv.get('success_cpu', 0):.1f} %, per-episode agreement {100 * nv.get('agree', 0):.1f} %, median depth "
                      f"difference {nv.get('depth_absdiff_med_mm', 0):.2f} mm."])


def figures(d):
    FIG.mkdir(parents=True, exist_ok=True)
    for p in PLOTS:
        if (d / "plots" / p).exists():
            shutil.copy(d / "plots" / p, FIG / p)
    for dst, src in STILLS.items():
        if (d / src).exists() and not (FIG / dst).exists():
            shutil.copy(d / src, FIG / dst)


def main():
    args = sys.argv[1:]
    run = next((a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--from", "--until"))),
               "20261123_1616_drawer")
    start = int(args[args.index("--from") + 1]) if "--from" in args else 17
    until = int(args[args.index("--until") + 1]) if "--until" in args else None
    d, ev, eps = load(run, until)
    blocks = dict(summary=block_summary(d, ev, eps), episodes=block_episodes([e for e in eps if e["mode"] == "rl"]),
                  arms=block_arms(eps, start), eval=block_eval(d, eps), resets=block_resets(d, ev), cost=block_cost(d, ev),
                  force=block_force(eps), sim=block_sim())
    cut = "" if until is None else f", episodes up to {until}"
    stamp = f"<!-- generated by docs/report/make_tables.py from runs/{run}{cut} at {time.strftime('%Y-%m-%d %H:%M')} -->"
    if "--figures" in args:
        figures(d)
    if "--update" in args:
        txt = REPORT.read_text()
        for k, v in blocks.items():
            txt, n = re.subn(rf"(<!-- BEGIN:{k} -->).*?(<!-- END:{k} -->)",
                             lambda m: f"{m.group(1)}\n{stamp}\n\n{v}\n{m.group(2)}", txt, flags=re.S)
            if not n:
                print(f"no block {k} in the report")
        REPORT.write_text(txt)
        print(f"updated {REPORT.relative_to(ROOT)}")
    else:
        for k, v in blocks.items():
            print(f"## {k}\n\n{v}\n")


if __name__ == "__main__":
    main()
