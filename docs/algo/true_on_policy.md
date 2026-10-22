# True On-Policy for the Qwen3-Omni Thinker

Last updated: 10/21/2026.

With `actor_rollout_ref.model.true_on_policy=true`, the log-probs the vLLM-Omni rollout reports for the Qwen3-Omni thinker are bitwise equal to the log-probs the FSDP actor recomputes for the same tokens with the same weights. PPO's old policy and the sampling policy are then the same distribution, with no numerical gap to correct for.

## What 0-diff means

For every training step, over the valid response tokens only:

- rollout and actor use the same weight version, the same token history and the same sampling distribution;
- the raw log-probs have the same dtype, are all finite, and differ in zero elements. No `exp`, no tolerance.

The check is repeated after a real optimizer update and weight sync. verl reports it as `training/rollout_logprobs_mismatch_count` (with `training/rollout_logprobs_diff_max` and `_mean`); verl's `training/rollout_probs_diff_*` compare probabilities after `exp` and hide 1-ULP differences.

## Enable

```bash
python3 -m verl_omni.trainer.main_omni \
    actor_rollout_ref.model.true_on_policy=True \
    actor_rollout_ref.actor.fsdp_config.batch_invariant=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.lora_rank=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    [other config overrides...]
```

## Contract

The trainer refuses to start unless all of these hold, and lists every violation in one error:

| Setting | Required |
|---|---|
| `actor_rollout_ref.rollout.temperature` | `1.0` |
| `actor_rollout_ref.rollout.top_p` | `1.0` |
| `actor_rollout_ref.rollout.top_k` | `-1` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` |
| `actor_rollout_ref.rollout.calculate_log_probs` | `true` |
| `actor_rollout_ref.model.lora_rank` | `0` |
| `actor_rollout_ref.model.use_remove_padding` | `true` |
| `actor_rollout_ref.actor.strategy` | `fsdp` or `fsdp2` |
| `actor_rollout_ref.actor.fsdp_config.batch_invariant` | `true` |

`actor_rollout_ref.model.override_config.attn_implementation` may be left unset or set to `hub_fa3`, and `override_config.experts_implementation` unset or set to `vllm_fused_moe`; anything else is an error. CUDA graphs on the rollout (`enforce_eager=false`) are supported.

`actor_rollout_ref.actor.fsdp_config.batch_invariant=true` also exports `VLLM_BATCH_INVARIANT=1` to the rollout. Do not add `actor_rollout_ref.rollout.full_determinism=true` for this: it seeds every request identically, so the `n` samples of a prompt come out the same and group-relative advantages vanish.

`algorithm.rollout_correction.bypass_mode=true` is allowed. It skips the actor recompute and uses the rollout log-probs as the old policy, which is equivalent once they are bitwise equal. The mismatch metric needs the recompute, so validate a setup with `bypass_mode=false` first.

## How it works

Each source of difference is fixed on the side that owns it. The HF layers keep their own `forward`.

| Piece | Where | What |
|---|---|---|
| attention | verl-omni | `hub_fa3`: an attention function registered with Transformers' `AttentionInterface`. It derives `cu_seqlens` from the packed position ids and calls the FlashAttention-3 varlen kernel from the `kernels` hub (`kernels-community/vllm-flash-attn3`), which is the kernel vLLM runs on Hopper and brings its own backward. `OmniModelConfig` selects it automatically |
| MoE experts | verl-omni | `vllm_fused_moe`: an experts implementation registered with Transformers' `ExpertsInterface`. Its forward is vLLM's `fused_experts` kernel, the one the rollout runs, which is batch-invariant under `VLLM_BATCH_INVARIANT=1`; the kernel has no backward, so the backward recomputes the experts with Transformers' `grouped_mm` path and differentiates that. `OmniModelConfig` selects it automatically |
| rope frequencies | verl-omni | the thinker's text rotary embedding is replaced with one that multiplies positions and fp32 inverse frequencies elementwise, as the rollout does. It avoids the fp32 matmul some Transformers releases use, which the batch-invariant kernels run at TF32, and keeps the frequencies in fp32 when `fsdp_config.model_dtype=bfloat16` casts the model |
| RMSNorm, residual add, rope, MoE routing | vLLM-Omni | thinker-stage option `additional_config: {hf_numerics: true}`: HF operation order and rounding inside torch custom ops, so torch.compile cannot re-fuse them and they are still captured into CUDA graphs. The experts stay on `fused_experts`. verl-omni writes the option into the thinker stage of the generated deploy config |
| batch-invariant actor ops, log-prob formula | verl | `actor_rollout_ref.actor.fsdp_config.batch_invariant`: installs vLLM's batch-invariant kernels in the actor and computes log-probs with the vLLM sampler formula (fp32 `log_softmax` + gather); verl-omni's entrypoint exports `VLLM_BATCH_INVARIANT=1` to the rollout under the same flag |
| gate metric | verl | `training/rollout_logprobs_mismatch_count` |

Dense GEMMs, the embedding, `lm_head` and synced weights are already bitwise equal on both sides.

## Scope

Supported: Qwen3-Omni thinker, text-only prompts, full-parameter training, rollout TP=1, FSDP/FSDP2 actor, Hopper GPUs.

Not supported:

- LoRA: merged and unmerged weights differ by construction.
- Multimodal inputs and the vision/audio towers.
- TP or EP > 1 on the rollout; the vLLM-Omni option refuses them.
- Run-to-run reproducibility of training: only the rollout-vs-actor log-probs within a run are covered.

## Cost

The rollout pays for the HF-order RMSNorm and rotary ops, which are unfused; the actor pays for vLLM's batch-invariant kernels and for the experts backward, which recomputes the `grouped_mm` path once per MoE layer. Keep the mode off for runs that do not need bitwise equality.

## Verify

```bash
bash tests/special_e2e/run_qwen3_omni_true_on_policy_gate.sh
```

Runs a tiny random Qwen3-Omni thinker on 2 GPUs. It checks `training/rollout_logprobs_mismatch_count == 0` at steps 1 and 2 with a nonzero `actor/grad_norm`, then reruns step 1 with `bypass_mode=true` and checks that `actor/ppo_kl` is exactly 0, i.e. the update forward matches the rollout log-probs used as the old policy.

CPU tests:

```bash
pytest tests/utils/test_config_on_cpu.py tests/pipelines/test_qwen3_omni_hub_fa3_attention_on_cpu.py \
    tests/pipelines/test_qwen3_omni_vllm_fused_moe_experts_on_cpu.py \
    tests/workers/config/test_omni_config_on_cpu.py tests/workers/rollout/rollout_vllm/test_vllm_omni_strategy_on_cpu.py
```

GPU test of the experts implementation against vLLM's kernel and the `grouped_mm` backward:

```bash
pytest tests/pipelines/test_qwen3_omni_vllm_fused_moe_experts.py
```
