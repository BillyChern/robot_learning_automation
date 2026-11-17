"""Training recipe v3: wrap-safe joint representation (H=30, LoRA).

Problem found after eval 05: joints 4 and 6 (index 3, 5) live around +-pi (joint 4 median 3.11 rad; joint 6 spans the
whole circle). State is wrapped to (-pi, pi], so the same pose appears as +3.1 or -3.1, and delta actions over a chunk
that crosses the wrap contain ~2 pi spikes (action std 0.37 / 0.60 rad for those joints vs 0.03-0.06 for the others).
Fix (transforms only; dataset unchanged):
  inputs:  KinovaInputs -> WrapDelta (delta = wrap_pi(action - state) for the 6 joints) -> EncodeState
           (state 7 -> 9: q1 q2 q3 sin q4 cos q4 q5 sin q6 cos q6 gripper)
  outputs: DecodeAbsolute (q4, q6 = atan2 from the encoded state; target = q + delta) -> KinovaOutputs
The rollout unwraps the absolute targets near the current joints, so any 2 pi offset is harmless."""
import dataclasses
import pathlib

import numpy as np
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from rlpj.vla import train_config as tc
from rlpj.vla.kinova_policy import KinovaInputs, KinovaOutputs

H = 30
STEPS = 12000
INIT = "/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_h30/11999/params"
SC = (3, 5)        # joints encoded as sin/cos


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def encode_state(s):
    s = np.asarray(s, np.float32)
    q, g = s[..., :6], s[..., 6:7]
    parts = [q[..., 0:3], np.sin(q[..., 3:4]), np.cos(q[..., 3:4]), q[..., 4:5], np.sin(q[..., 5:6]), np.cos(q[..., 5:6]), g]
    return np.concatenate(parts, -1).astype(np.float32)


def decode_joints(e):
    e = np.asarray(e, np.float32)
    return np.concatenate([e[..., 0:3], np.arctan2(e[..., 3:4], e[..., 4:5]), e[..., 5:6],
                           np.arctan2(e[..., 6:7], e[..., 7:8])], -1)


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
class EncodeState(_transforms.DataTransformFn):
    def __call__(self, data):
        data["state"] = encode_state(data["state"])
        return data


@dataclasses.dataclass(frozen=True)
class DecodeAbsolute(_transforms.DataTransformFn):
    def __call__(self, data):
        if "actions" not in data:
            return data
        a = np.array(data["actions"], np.float32)
        q = decode_joints(data["state"])
        a[..., :6] = a[..., :6] + q[..., None, :6]
        data["actions"] = a
        return data


@dataclasses.dataclass(frozen=True)
class WrapSafeKinovaDataConfig(tc.LeRobotKinovaDataConfig):
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(
            {"image": "image", "wrist_image": "wrist_image", "state": "state", "actions": "actions", "prompt": "prompt"})])
        data_transforms = _transforms.Group(inputs=[KinovaInputs(model_type=model_config.model_type), WrapDelta(), EncodeState()],
                                            outputs=[DecodeAbsolute(), KinovaOutputs()])
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                   data_transforms=data_transforms, model_transforms=model_transforms)


def _model():
    return pi0_config.Pi0Config(pi05=True, action_horizon=H, paligemma_variant="gemma_2b_lora",
                                action_expert_variant="gemma_300m_lora")


def configs(run_dir=None, repo_id=None):
    base = tc.configs(run_dir, repo_id)[tc.LORA_NAME]
    data = WrapSafeKinovaDataConfig(repo_id=repo_id or tc.DEFAULT_REPO_ID,
                                    base_config=_config.DataConfig(prompt_from_task=True))
    meta = dict(tc.POLICY_METADATA, action_horizon=H,
                state="q1..q6 rad wrapped to (-pi, pi], gripper reading / 100 (encoded to sin/cos for q4, q6 inside the policy)")
    name = "pi05_kinova_lora_v3"
    return {name: dataclasses.replace(
        base, name=name, data=data, model=_model(), freeze_filter=_model().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(INIT),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-5, decay_steps=STEPS, decay_lr=5e-6),
        num_train_steps=STEPS, policy_metadata=meta)}
