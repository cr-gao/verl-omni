# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.trainer.diffusion.diffusion_trainer_utils import skip_old_log_prob
from verl_omni.workers.utils.losses import diffusion_loss

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class TestSkipOldLogProb:
    def test_flow_grpo_recomputes(self):
        assert skip_old_log_prob(compose_cfg([])) is False

    def test_pure_distill_kl_skips(self):
        cfg = compose_cfg(["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"])
        assert skip_old_log_prob(cfg) is True

    def test_auxiliary_distill_on_flow_grpo_recomputes(self):
        cfg = compose_cfg(["actor_rollout_ref.actor.use_distill_loss=true"])
        assert skip_old_log_prob(cfg) is False

    def test_distill_kl_with_rollout_correction_recomputes(self):
        cfg = compose_cfg(
            [
                "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
                "algorithm.rollout_correction.rollout_is=sequence",
            ]
        )
        assert skip_old_log_prob(cfg) is False

    def test_kl_penalty_does_not_force_recompute(self):
        cfg = compose_cfg(
            ["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl", "actor_rollout_ref.actor.use_kl_loss=true"]
        )
        assert skip_old_log_prob(cfg) is True


def _actor_config(overrides):
    with initialize_config_dir(config_dir=os.path.join(CONFIG_DIR, "diffusion", "actor"), version_base=None):
        cfg = compose(
            config_name="dp_diffusion_actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=4", *overrides]
        )
    return omega_conf_to_dataclass(cfg)


def _step_inputs(with_old_log_probs):
    torch.manual_seed(0)
    model_output = {
        "log_probs": torch.randn(4),
        "prev_sample_mean": torch.randn(4, 16, 3),
        "std_dev_t": torch.ones(4, 1, 1),
    }
    tensors = {"advantages": torch.randn(4), "teacher_prev_sample_mean": torch.randn(4, 16, 3)}
    if with_old_log_probs:
        tensors["old_log_probs"] = torch.randn(4)
    data = tu.get_tensordict(tensors)
    tu.assign_non_tensor(data, gradient_accumulation_steps=1, sp_size=1)
    return model_output, data


class TestLossWithoutOldLogProbs:
    def test_distill_kl_trains_without_them(self):
        loss, _ = diffusion_loss(_actor_config(["diffusion_loss.loss_mode=distill_kl"]), *_step_inputs(False))
        assert torch.isfinite(loss)

    def test_log_prob_loss_names_the_missing_key(self):
        with pytest.raises(KeyError, match="old_log_probs"):
            diffusion_loss(_actor_config([]), *_step_inputs(False))
