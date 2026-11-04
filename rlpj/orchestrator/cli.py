"""`rlpj` command line: check, fit-kin, new, develop, collect, learn, eval, eval-zero-shot, run-all, status, review, report, vla."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from rlpj import config
from . import state as S

CONDITIONS = ("text", "one_demo")


def build_parser():
    p = argparse.ArgumentParser(prog="rlpj", description="Orchestrator: agents write the robot programs, this runs them, "
                                "collects episodes, fine-tunes pi0.5 and evaluates it.")
    p.add_argument("--hands-free", action="store_true", help="robot gates count down and go without a keypress (operator on the e-stop)")
    p.add_argument("--yes", action="store_true", help="no prompt for steps that do not move the robot")
    p.add_argument("--task", help="task name (config/tasks/<task>.yaml); also RLPJ_TASK")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="robot, camera and calibration checks (nothing moves)")
    sub.add_parser("fit-kin", help="python -m rlpj.robot.fit_kin <args...>")
    s = sub.add_parser("new", help="new run; prints the run id")
    s.add_argument("--task", default=argparse.SUPPRESS)
    s.add_argument("--condition", choices=CONDITIONS, required=True)
    per_run = (("develop", "agent 1/2 writes the execute|reset program", [("role", dict(choices=("execute", "reset")))]),
               ("collect", "collection round (--n successes, default collection.round_size)", [("--n", dict(type=int))]),
               ("learn", "agent 3: convert, train, evaluate", []),
               ("eval", "evaluate a checkpoint on the robot", [("--checkpoint", dict(required=True)), ("--n", dict(type=int))]),
               ("eval-zero-shot", "evaluate training.base_checkpoint", [("--n", dict(type=int))]),
               ("run-all", "check -> develop -> collect -> learn -> report, resumable", []),
               ("status", "state table", []), ("review", "label the unsure episodes", []), ("report", "write REPORT.md", []))
    for name, help_, extra in per_run:
        s = sub.add_parser(name, help=help_)
        s.add_argument("run", help="run id (runs/<run>)")
        for flag, kw in extra:
            s.add_argument(flag, **kw)
    sub.add_parser("vla", help="convert | train | serve <args...> (GPU machine; thin wrappers over scripts/ and rlpj.vla)")
    return p


def split(argv):
    """(argv for argparse, passthrough args): everything after `fit-kin` or `vla` goes to the delegate untouched."""
    for i, tok in enumerate(argv):
        if tok in ("fit-kin", "vla"):
            return argv[:i + 1], argv[i + 1:]
    return argv, []


def main(argv=None):
    argv, rest = split(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    if args.task:
        os.environ["RLPJ_TASK"] = args.task
    if args.cmd == "fit-kin":
        return subprocess.call([sys.executable, "-m", "rlpj.robot.fit_kin", *rest])
    if args.cmd == "vla":
        return vla(rest)
    cfg = config.load()
    if args.cmd == "check":
        from rlpj.robot import check
        return 0 if check.run(log=print).get("ok") else 1
    if args.cmd == "new":
        task = cfg["task"]["name"]
        st = S.RunState.create(S.new_run_id(task), task, args.condition)
        print(st.run_id)
        return 0
    from .loop import Pipeline
    from .operator import Operator
    try:
        st = S.RunState.open(args.run)
    except FileNotFoundError as e:
        print(e)
        return 2
    if args.cmd == "status":
        print("\n".join(S.status_lines(st, cfg)))
        return 0
    op = Operator(hands_free=args.hands_free, yes=args.yes, countdown_s=cfg.get("operator", {}).get("countdown_s", 5))
    p = Pipeline(st, op, cfg)
    try:
        if args.cmd == "develop":
            return 0 if p.develop_program(args.role) == "done" else 1
        if args.cmd == "collect":
            p.collect(args.n)
        elif args.cmd == "learn":
            print(p.learn())
        elif args.cmd == "eval":
            r = p.evaluate(args.checkpoint, args.n)
            return 0 if r.get("met") else 1
        elif args.cmd == "eval-zero-shot":
            p.evaluate_zero_shot(args.n)
        elif args.cmd == "run-all":
            print(p.run_all())
        elif args.cmd == "review":
            review(st, op, p)
        elif args.cmd == "report":
            print(p.report())
        return 0
    except KeyboardInterrupt:
        st.save()
        print("\n== interrupted; state saved (rerun the same command to resume)")
        return 130


def review(st, op, p):
    """Walk the operator queue: show the episode's files and verdict, ask success/fail, write the label."""
    pending = S.pending_reviews(st)
    if not pending:
        print("nothing to review")
        return
    for rec in pending:
        ep, v = Path(rec["episode"]), rec.get("verdict") or {}
        print(f"\n== {ep.name}: {v.get('final')} - {v.get('reason')}")
        for f in sorted(ep.glob("*.jpg")) + sorted(ep.glob("*.mp4")):
            print(f"   {f}")
        a = op.ask("label", ["skip", "success", "fail", "quit"])
        if a == "quit":
            break
        if a in ("success", "fail"):
            p.relabel(ep, a, op.ask("note (Enter = none)"))


def vla(args):
    """rlpj vla convert --run-dir R --name N [...] | train --run-dir R --dataset N --exp-name E [overrides] | serve --checkpoint C [--port P]"""
    if not args or args[0] not in ("convert", "train", "serve"):
        print(vla.__doc__)
        return 2
    kind, rest = args[0], args[1:]

    def take(flag, default=None):
        if flag in rest[:-1]:
            i = rest.index(flag)
            v = rest[i + 1]
            del rest[i:i + 2]
            return v
        return default

    root = config.ROOT
    if kind == "convert":
        cmd = f"cd {shlex.quote(str(root / 'third_party/openpi'))} && uv run python -m rlpj.vla.convert {shlex.join(rest)}"
    elif kind == "train":
        run_dir, dataset, exp = take("--run-dir"), take("--dataset"), take("--exp-name")
        if not (run_dir and dataset and exp):
            print("usage: rlpj vla train --run-dir R --dataset NAME --exp-name EXP [tyro overrides]")
            return 2
        cmd = f"bash {shlex.quote(str(root / 'scripts/train.sh'))} {shlex.join([run_dir, dataset, exp, *rest])}"
    else:
        ckpt, port = take("--checkpoint"), take("--port", "8000")
        if not ckpt:
            print("usage: rlpj vla serve --checkpoint DIR [--port P]")
            return 2
        cmd = f"bash {shlex.quote(str(root / 'scripts/serve.sh'))} {shlex.join([ckpt, port])}"
    print(f"$ {cmd}", flush=True)
    return subprocess.call(cmd, shell=True, executable="/bin/bash", cwd=str(root))


if __name__ == "__main__":
    sys.exit(main())
