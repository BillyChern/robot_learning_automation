"""Full fine-tuning recipe (PyTorch trainer) with the v4 representation:
  - wrap-safe joint deltas (wrap_pi(action - q)): joints 4/6 sit at +-pi; the stock deltas jump by 2 pi there
    (eval 08: 17/20 rollouts stopped by joint-limit chunks ~26 s in, at the look-around pose);
  - model state = gripper reading only: with joint angles the policy replays trajectories and ignores the images
    (programs/vision_check.py: other episode's images -> same chunk);
  - H=30 (rollout_v4: chunk 30, replan 20, targets = q_now + delta).
Initialised from full_v1/30000 (full fine-tune on bee_v3, task-adapted vision). Name contains "full" -> PyTorch trainer."""
import dataclasses
import pathlib

import numpy as np
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

from rlpj.vla import train_config as tc
from rlpj.vla.kinova_policy import KinovaInputs, KinovaOutputs

H = 30
INIT = "/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000"


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


@dataclasses.dataclass(frozen=True)
class WrapDelta(_transforms.DataTransformFn):
    def __call__(self, data):
        if "actions" not in data:
            return data
        a = np.array(data["actions"], np.float32)
        a[..., :6] = wrap(a[..., :6] - np.asarray(data["state"], np.float32)[..., None, :6])
        data["actions"] = a
        return data


@dataclasses.dataclass(frozen=True)
class GripperOnlyState(_transforms.DataTransformFn):
    def __call__(self, data):
        data["state"] = np.asarray(data["state"], np.float32)[..., 6:7]
        return data


@dataclasses.dataclass(frozen=True)
class VisionKinovaDataConfig(tc.LeRobotKinovaDataConfig):
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(
            {"image": "image", "wrist_image": "wrist_image", "state": "state", "actions": "actions", "prompt": "prompt"})])
        data_transforms = _transforms.Group(inputs=[KinovaInputs(model_type=model_config.model_type), WrapDelta(), GripperOnlyState()],
                                            outputs=[KinovaOutputs()])
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                   data_transforms=data_transforms, model_transforms=model_transforms)


def configs(run_dir=None, repo_id=None):
    base = tc.configs(run_dir, repo_id)[tc.FULL_NAME]
    data = VisionKinovaDataConfig(repo_id=repo_id or tc.DEFAULT_REPO_ID, base_config=_config.DataConfig(prompt_from_task=True))
    model = pi0_config.Pi0Config(pi05=True, action_horizon=H, pytorch_compile_mode=None)
    meta = dict(tc.POLICY_METADATA, action_horizon=H, state="gripper reading / 100 only (joints ignored)",
                actions="q1..q6 DELTA rad wrap_pi(target - q_now) (rollout adds q_now), gripper command 0..1")
    name = "pi05_kinova_full_v4"
    return {name: dataclasses.replace(base, name=name, data=data, model=model, pytorch_weight_path=INIT, policy_metadata=meta,
        # continuing from a fine-tuned model: half the stock peak, cosine to 1/10 over the run
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=2.5e-5, decay_steps=30000, decay_lr=2.5e-6))}
