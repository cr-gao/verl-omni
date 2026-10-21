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
"""vLLM fused MoE experts registered with Transformers.

The vLLM-Omni rollout runs the experts through vLLM's ``fused_experts`` kernel; calling the same kernel
in the actor makes the MoE output bitwise equal on both sides. The kernel has no backward, so the backward
pass recomputes the experts with Transformers' ``grouped_mm`` path and differentiates that instead.
"""

import torch

VLLM_FUSED_MOE_EXPERTS = "vllm_fused_moe"


def fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Run vLLM's fused experts on HF-layout weights.

    Args:
        hidden_states: ``(N, H)`` routed tokens.
        gate_up_proj: ``(E, 2I, H)`` gate and up projections, gate first.
        down_proj: ``(E, H, I)`` down projections.
        top_k_index: ``(N, K)`` selected experts.
        top_k_weights: ``(N, K)`` routing weights in the activation dtype.

    Returns:
        ``(N, H)`` combined expert output in the activation dtype.
    """
    from vllm.model_executor.layers.fused_moe import fused_experts as vllm_fused_experts

    return vllm_fused_experts(
        hidden_states, gate_up_proj, down_proj, top_k_weights.float(), top_k_index.to(torch.int32)
    )


class _GroupedMMExperts:
    """The attributes ``grouped_mm_experts_forward`` reads from an HF experts module."""

    has_gate = True
    has_bias = False
    is_transposed = False
    is_concatenated = True
    _is_expert_parallel = False

    def __init__(self, gate_up_proj, down_proj, act_fn):
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.num_experts = gate_up_proj.shape[0]
        self.act_fn = act_fn

    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up


class _FusedExpertsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act_fn):
        ctx.save_for_backward(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj)
        ctx.act_fn = act_fn
        return fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights)

    @staticmethod
    def backward(ctx, grad_output):
        from transformers.integrations.moe import grouped_mm_experts_forward

        hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj = ctx.saved_tensors
        with torch.enable_grad():
            inputs = [t.detach().requires_grad_() for t in (hidden_states, top_k_weights, gate_up_proj, down_proj)]
            experts = _GroupedMMExperts(inputs[2], inputs[3], ctx.act_fn)
            out = grouped_mm_experts_forward(experts, inputs[0], top_k_index, inputs[1])
        grad_hidden, grad_weights, grad_gate_up, grad_down = torch.autograd.grad(out, inputs, grad_output)
        return grad_hidden, None, grad_weights, grad_gate_up, grad_down, None


def vllm_fused_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Experts forward for ``ExpertsInterface``: vLLM ``fused_experts`` with a ``grouped_mm`` backward."""
    return _FusedExpertsFunction.apply(
        hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj, self.act_fn
    )


def register_vllm_fused_moe_experts() -> str:
    """Register :func:`vllm_fused_moe_experts_forward` under ``vllm_fused_moe``."""
    from transformers.integrations.moe import ExpertsInterface

    ExpertsInterface.register(VLLM_FUSED_MOE_EXPERTS, vllm_fused_moe_experts_forward)
    return VLLM_FUSED_MOE_EXPERTS
