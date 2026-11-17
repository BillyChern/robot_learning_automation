"""Blind zero-shot test of a trained policy on a new object, with a person at the robot (no reset program).

Nothing about the object is given to the robot: the instruction is generic, the look move detects "the object that is
not the bowl", and the VLM judge gets a generic success description. Per rollout: the person places the object and
presses Enter; the rollout runs through rlpj.orchestrator.attempts.run_attempt (recorded; Ctrl+C stops the arm); the
person answers y/n/u; then the VLM judge rates the before/after images (shown after the person's answer).
Results: runs/<name>/results.jsonl and summary.json.

    .venv/bin/python scripts/zero_shot_eval.py --name zeroshot_blind [--n 10] [--instruction "..."] [--host 100.122.29.80]
"""
import argparse
import json
import shlex
import time
from pathlib import Path

from rlpj import config
from rlpj.orchestrator.attempts import run_attempt
from rlpj.success import vlm_judge

LEARN = config.ROOT / "runs" / "20261104_1510_bee" / "agents" / "learn"
TASK = dict(name="toy_in_bowl",
            goal="A toy object and a bowl are on the table. Pick up the toy and put it into the bowl.",
            success="The toy lies inside the bowl and the gripper is clear of it.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="zeroshot_blind")
    p.add_argument("--instruction", default="pick up the toy and put it into the bowl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--host", default="100.122.29.80")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-seconds", type=int, default=240)
    a = p.parse_args()
    out = config.ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cmd = (f"python {shlex.quote(str(config.ROOT / 'scripts' / 'zero_shot_rollout.py'))} --host {a.host} --port {a.port} "
           f"--instruction {shlex.quote(a.instruction)} --max-seconds {a.max_seconds}")
    results = [json.loads(l) for l in open(out / "results.jsonl")] if (out / "results.jsonl").exists() else []
    print(f"Blind zero-shot test: {a.n} rollouts, instruction {a.instruction!r}, policy at {a.host}:{a.port}. "
          f"{len(results)} already done.")
    for i in range(len(results) + 1, a.n + 1):
        input(f"\n[{i}/{a.n}] Arm at home, gripper empty. Put the object at a new spot in front of the robot "
              f"(not touching the bowl), step back, then press Enter... ")
        d = out / f"r{i:02d}"
        t0 = time.time()
        r = run_attempt(str(LEARN), cmd, d / "attempt", moves_robot=True, record_dir=str(d), timeout_s=a.max_seconds + 240)
        ans = ""
        while ans not in ("y", "n", "u"):
            ans = input(f"[{i}/{a.n}] Is the object in the bowl and the gripper clear? y = success, n = fail, u = unsure: ").strip().lower()
        note = input("Optional note, e.g. what it did (Enter to skip): ").strip()
        imgs = {k: d / f"{k}.jpg" for k in ("before_side", "after_side", "before_wrist", "after_wrist") if (d / f"{k}.jpg").exists()}
        vlm = vlm_judge.judge(imgs, TASK) if imgs else dict(success=None, reason="no images recorded")
        human = {"y": "success", "n": "fail", "u": "unsure"}[ans]
        rec = dict(i=i, human=human, vlm=vlm.get("success"), vlm_reason=str(vlm.get("reason", ""))[:300], note=note,
                   exit_code=r.exit_code, seconds=round(time.time() - t0, 1), dir=str(d))
        results.append(rec)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        ok = sum(x["human"] == "success" for x in results)
        print(f"   VLM judge: {rec['vlm']} ({rec['vlm_reason'][:120]})")
        print(f"== {ok}/{len(results)} successes so far (your answers)")
    s = dict(name=a.name, instruction=a.instruction, n=len(results),
             success_human=sum(x["human"] == "success" for x in results),
             success_vlm=sum(x["vlm"] is True for x in results),
             agree=sum((x["human"] == "success") == (x["vlm"] is True) for x in results),
             policy="runs/20261104_1510_bee/checkpoints/full_v7/30000 (served on %s:%d)" % (a.host, a.port))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
