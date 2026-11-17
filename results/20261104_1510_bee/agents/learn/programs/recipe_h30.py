"""Training recipe: pi05 LoRA with action horizon 30 (2 s at 15 fps), initialised from lora_v1/14999, cosine decay.

Why: with H=10 the rollout executes 0.67 s chunks from rest; stop-and-go caused 1 deg tracking aborts (eval 03/04) and
slow episodes that hit the time limit. H=30 lets the rollout execute ~20 actions per stop and gives temporally
consistent chunks. Starting from lora_v1/14999 keeps what was learned (LoRA + base weights saved there); the action
expert is shared across time steps, so the change of horizon does not change parameter shapes."""
import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

from rlpj.vla import train_config as tc

H = 30
INIT = "/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v1/14999/params"


def _model():
    return pi0_config.Pi0Config(pi05=True, action_horizon=H, paligemma_variant="gemma_2b_lora",
                                action_expert_variant="gemma_300m_lora")


def configs(run_dir=None, repo_id=None):
    base = tc.configs(run_dir, repo_id)[tc.LORA_NAME]
    meta = dict(tc.POLICY_METADATA, action_horizon=H)
    out = {}
    for steps in (12000,):
        name = "pi05_kinova_lora_h30"
        out[name] = dataclasses.replace(
            base, name=name, model=_model(), freeze_filter=_model().get_freeze_filter(),
            weight_loader=weight_loaders.CheckpointWeightLoader(INIT),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-5, decay_steps=steps, decay_lr=5e-6),
            num_train_steps=steps, policy_metadata=meta)
    return out
