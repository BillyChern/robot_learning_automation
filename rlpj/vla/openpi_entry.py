"""openpi's scripts with our configs registered (5090, openpi venv, PYTHONPATH=<repo>, cwd third_party/openpi):

  python -m rlpj.vla.openpi_entry norm-stats --run-dir R --dataset NAME [--config-name pi05_kinova_lora] [--max-frames N] [--batch-size B]
  python -m rlpj.vla.openpi_entry train --run-dir R --dataset NAME [pi05_kinova_lora] --exp-name E [tyro overrides, e.g.
                                        --num-train-steps 5000 --batch-size 16 --overwrite | --resume --lr-schedule.peak-lr 1e-4]
  python -m rlpj.vla.openpi_entry serve policy:checkpoint --policy.config pi05_kinova_lora --policy.dir CKPT [--port 8000]

HF_LEROBOT_HOME = R/datasets (dataset rlpj/NAME); checkpoints R/checkpoints/E/<step>; norm stats R/assets/<config>/rlpj/NAME;
wandb disabled unless RLPJ_WANDB=1; serve reads the dataset id back from the checkpoint's assets/ dir.
RLPJ_RECIPE=<file.py>: agent 3's recipe (configs(run_dir, repo_id) -> {name: TrainConfig}) registered next to ours."""
import argparse
import dataclasses
import importlib
import json
import logging
import os
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI = ROOT / "third_party" / "openpi"
USAGE = __doc__


def _script(name):
    """Import third_party/openpi/scripts/<name>.py as scripts.<name> (importable again in spawned loader workers)."""
    if str(OPENPI) not in sys.path:
        sys.path.insert(0, str(OPENPI))
    return importlib.import_module(f"scripts.{name}")


def load_recipe(path, run_dir=None, repo_id=None):
    """Agent 3's training recipe: a .py file with configs(run_dir, repo_id) -> {name: TrainConfig}. Imported by module
    name with its folder on sys.path and PYTHONPATH, so spawned data-loader workers can import it too."""
    path = pathlib.Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(path.parent)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    mod = importlib.import_module(path.stem)
    return dict(mod.configs(run_dir, repo_id))


def _register(run_dir=None, repo_id=None):
    """Our configs plus the recipe in $RLPJ_RECIPE (if set), registered into openpi's config registry."""
    from rlpj.vla import train_config as tc
    import openpi.training.config as _config
    cfgs = tc.register(run_dir, repo_id)
    if os.environ.get("RLPJ_RECIPE"):
        extra = load_recipe(os.environ["RLPJ_RECIPE"], run_dir, repo_id)
        for c in extra.values():
            _config._CONFIGS_DICT[c.name] = c
            _config._CONFIGS[:] = [x for x in _config._CONFIGS if x.name != c.name] + [c]
        cfgs = {**cfgs, **extra}
    return cfgs


def _repo_id(dataset):
    return None if not dataset else (dataset if "/" in dataset else f"rlpj/{dataset}")


def _setup_env(run_dir):
    run_dir = pathlib.Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(run_dir / "datasets")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    return run_dir


def _parser(prog, dataset_required=False):
    p = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", required=dataset_required, help="dataset NAME (repo_id rlpj/NAME) or a full repo_id")
    return p


def asset_id_from_checkpoint(ckpt_dir):
    """Dataset id (e.g. rlpj/bee_v1) from <ckpt>/assets/<id>/norm_stats.json, or None."""
    d = pathlib.Path(ckpt_dir)
    found = sorted(d.glob("assets/**/norm_stats.json"))
    return str(found[0].parent.relative_to(d / "assets")) if found else None


def cmd_norm_stats(argv):
    p = _parser("openpi_entry norm-stats", dataset_required=True)
    p.add_argument("--config-name", default="pi05_kinova_lora")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="loader batch (must be <= dataset frames)")
    ns = p.parse_args(argv)
    run_dir = _setup_env(ns.run_dir)
    import openpi.training.config as _config
    cfgs = _register(run_dir, _repo_id(ns.dataset))
    cfg = cfgs[ns.config_name]
    if ns.batch_size:
        cfg = dataclasses.replace(cfg, batch_size=ns.batch_size)
        _config._CONFIGS_DICT[cfg.name] = cfg
    t0 = time.time()
    _script("compute_norm_stats").main(ns.config_name, ns.max_frames)
    print(f"norm stats done in {time.time() - t0:.1f}s")


def cmd_train(argv):
    ns, rest = _parser("openpi_entry train").parse_known_args(argv)
    run_dir = _setup_env(ns.run_dir)
    from rlpj.vla import train_config as tc
    import tyro
    cfgs = _register(run_dir, _repo_id(ns.dataset))
    if not rest or rest[0].startswith("-"):
        rest = [tc.LORA_NAME, *rest]
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in cfgs.items()}, args=rest,
                                                prog="openpi_entry train")
    t0 = time.time()
    _script("train").main(config)
    info = dict(config=config.name, exp_name=config.exp_name, repo_id=config.data.repo_id,
                checkpoint_dir=str(config.checkpoint_dir), num_train_steps=config.num_train_steps,
                batch_size=config.batch_size, lr_schedule=dataclasses.asdict(config.lr_schedule),
                hf_lerobot_home=os.environ["HF_LEROBOT_HOME"], host=platform.node(), seconds=round(time.time() - t0, 1),
                recipe=os.environ.get("RLPJ_RECIPE"),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:])
    (config.checkpoint_dir / "train_info.json").write_text(json.dumps(info, indent=1))
    print(f"train done in {info['seconds']}s -> {config.checkpoint_dir}")


def cmd_serve(argv):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from rlpj.vla import train_config as tc
    import tyro
    sp = _script("serve_policy")
    args = tyro.cli(sp.Args, args=argv, prog="openpi_entry serve")
    repo_id = asset_id_from_checkpoint(args.policy.dir) if isinstance(args.policy, sp.Checkpoint) else None
    _register(repo_id=repo_id)
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("serving %s with dataset id %s", getattr(args.policy, "dir", args.policy), repo_id)
    sp.main(args)


def _manage_pytorch_checkpoints(tp):
    """openpi's PyTorch trainer keeps every checkpoint (about 20 GB each with the optimizer). After each save keep the
    newest whole (to resume), keep_period multiples as weights only, and remove the rest. RLPJ_NO_SAVE=1 skips saving
    (memory tests)."""
    import shutil
    orig = tp.save_checkpoint

    def save(model, optimizer, global_step, config, is_main, data_config):
        if os.environ.get("RLPJ_NO_SAVE") == "1":
            return
        orig(model, optimizer, global_step, config, is_main, data_config)
        if not is_main or not config.checkpoint_dir.exists():
            return
        steps = sorted(int(p.name) for p in config.checkpoint_dir.iterdir() if p.name.isdigit())
        for s in steps[:-1]:
            d = config.checkpoint_dir / str(s)
            if config.keep_period and s % config.keep_period == 0:
                (d / "optimizer.pt").unlink(missing_ok=True)
            else:
                shutil.rmtree(d, ignore_errors=True)

    tp.save_checkpoint = save


def _use_8bit_adamw():
    """openpi's PyTorch trainer builds torch.optim.AdamW. Swap in bitsandbytes' 8-bit AdamW (same update, state in 8 bits).
    bf16 AdamW state needs about 15 GB for pi0.5 and runs out of memory on the 32 GB 5090 (2026-11-07: 30 GB at the
    first optimizer step, batch 2); 8-bit state is about 7.5 GB. Every weight is still trained (full fine-tuning)."""
    import bitsandbytes as bnb
    import torch

    class AdamW8bit(bnb.optim.AdamW8bit):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, **_):
            super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    torch.optim.AdamW = AdamW8bit


def cmd_train_pytorch(argv):
    """Full fine-tuning with openpi's PyTorch trainer (default config pi05_kinova_full: bf16, batch 2, 30k steps)."""
    ns, rest = _parser("openpi_entry train-pytorch").parse_known_args(argv)
    run_dir = _setup_env(ns.run_dir)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from rlpj.vla import train_config as tc
    import tyro
    cfgs = _register(run_dir, _repo_id(ns.dataset))
    if not rest or rest[0].startswith("-"):
        rest = [tc.FULL_NAME, *rest]
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in cfgs.items()}, args=rest,
                                                prog="openpi_entry train-pytorch")
    tp = _script("train_pytorch")
    _manage_pytorch_checkpoints(tp)
    optimizer = os.environ.get("RLPJ_OPTIMIZER", "adamw8bit")
    if optimizer == "adamw8bit":
        _use_8bit_adamw()
    tp.init_logging()
    t0 = time.time()
    tp.train_loop(config)
    info = dict(trainer="pytorch", optimizer=optimizer, config=config.name, exp_name=config.exp_name, repo_id=config.data.repo_id,
                checkpoint_dir=str(config.checkpoint_dir), num_train_steps=config.num_train_steps,
                batch_size=config.batch_size, lr_schedule=dataclasses.asdict(config.lr_schedule),
                precision=config.pytorch_training_precision, hf_lerobot_home=os.environ["HF_LEROBOT_HOME"],
                host=platform.node(), seconds=round(time.time() - t0, 1), recipe=os.environ.get("RLPJ_RECIPE"),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:])
    if os.environ.get("RLPJ_NO_SAVE") != "1":
        (config.checkpoint_dir / "train_info.json").write_text(json.dumps(info, indent=1))
    print(f"train-pytorch done in {info['seconds']}s -> {config.checkpoint_dir}")


COMMANDS = {"norm-stats": cmd_norm_stats, "train": cmd_train, "train-pytorch": cmd_train_pytorch, "serve": cmd_serve}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(USAGE, file=sys.stderr)
        return 2
    COMMANDS[argv[0]](argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
