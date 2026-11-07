"""openpi TrainConfigs for the Kinova data (pi05_kinova_lora, pi05_kinova_full), registered into
openpi.training.config without editing openpi. Needs the openpi venv (5090). Norm stats are our own, computed by
openpi_entry norm-stats into <run>/assets/<config>/<repo_id>/ and copied into every checkpoint by openpi."""
import dataclasses
import os
import pathlib
from typing import Any

from typing_extensions import override

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from .kinova_policy import KinovaInputs, KinovaOutputs

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
PI05_BASE_PYTORCH = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"   # convert_jax_model_to_pytorch.py, bf16
DEFAULT_REPO_ID = "rlpj/kinova"
LORA_NAME, FULL_NAME = "pi05_kinova_lora", "pi05_kinova_full"
ACTION_HORIZON = 10
POLICY_METADATA: dict[str, Any] = {
    "robot": "kinova_gen3_6dof",
    "state": "q1..q6 rad wrapped to (-pi, pi], gripper reading / 100",
    "actions": "q1..q6 rad absolute targets, gripper command 0 open .. 1 closed",
    "action_horizon": ACTION_HORIZON,
    "image_size": 224,
}


@dataclasses.dataclass(frozen=True)
class LeRobotKinovaDataConfig(_config.DataConfigFactory):
    """Repack our LeRobot keys, KinovaInputs/Outputs, joint deltas (gripper absolute), standard model transforms."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(
            {"image": "image", "wrist_image": "wrist_image", "state": "state", "actions": "actions", "prompt": "prompt"})])
        data_transforms = _transforms.Group(inputs=[KinovaInputs(model_type=model_config.model_type)],
                                            outputs=[KinovaOutputs()])
        mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(mask)],
                                               outputs=[_transforms.AbsoluteActions(mask)])
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                   data_transforms=data_transforms, model_transforms=model_transforms)


@dataclasses.dataclass(frozen=True)
class KinovaTrainConfig(_config.TrainConfig):
    """TrainConfig whose checkpoints live directly at <checkpoint_base_dir>/<exp_name> (runs/<run>/checkpoints/<exp>)."""

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.exp_name).resolve()


def lora_model() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, paligemma_variant="gemma_2b_lora",
                                action_expert_variant="gemma_300m_lora")


def configs(run_dir=None, repo_id: str | None = None) -> dict[str, _config.TrainConfig]:
    """Our configs; run_dir sets assets/checkpoint dirs, repo_id the dataset (default rlpj/kinova)."""
    dirs = {}
    if run_dir:
        run_dir = pathlib.Path(run_dir).resolve()
        dirs = dict(assets_base_dir=str(run_dir / "assets"), checkpoint_base_dir=str(run_dir / "checkpoints"))
    data = LeRobotKinovaDataConfig(repo_id=repo_id or DEFAULT_REPO_ID,
                                   base_config=_config.DataConfig(prompt_from_task=True))
    common = dict(data=data, weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE_PARAMS),
                  lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=1_000_000,
                                                             decay_lr=5e-5),
                  optimizer=_optimizer.AdamW(clip_gradient_norm=1.0), batch_size=16, num_train_steps=5000,
                  log_interval=50, save_interval=500, keep_period=5000, num_workers=2,   # ~9 GB per kept checkpoint (5090 disk)
                  wandb_enabled=os.environ.get("RLPJ_WANDB") == "1", policy_metadata=POLICY_METADATA, **dirs)
    lora = KinovaTrainConfig(name=LORA_NAME, model=lora_model(), freeze_filter=lora_model().get_freeze_filter(),
                             ema_decay=None, **common)
    # full fine-tuning with openpi's PyTorch trainer (openpi_entry train-pytorch): bf16 weights and optimizer state,
    # gradient checkpointing, no EMA, small batch, so it fits the 32 GB 5090 (JAX full fine-tuning needs > 70 GB)
    full = KinovaTrainConfig(name=FULL_NAME, model=pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON,
                                                                        pytorch_compile_mode=None),
                             ema_decay=None, pytorch_weight_path=str(PI05_BASE_PYTORCH), pytorch_training_precision="bfloat16",
                             **{**common, "batch_size": 2, "num_train_steps": 30000})
    return {c.name: c for c in (lora, full)}


def register(run_dir=None, repo_id: str | None = None) -> dict[str, _config.TrainConfig]:
    """Insert our configs into openpi's registry (_CONFIGS_DICT and _CONFIGS); idempotent."""
    cfgs = configs(run_dir, repo_id)
    for c in cfgs.values():
        _config._CONFIGS_DICT[c.name] = c
        _config._CONFIGS[:] = [x for x in _config._CONFIGS if x.name != c.name] + [c]
    return cfgs
