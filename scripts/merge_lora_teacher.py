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
"""Merge a LoRA checkpoint into a base diffusion model to build a distillation teacher.

Produces a full model directory usable as ``actor_rollout_ref.ref.model_path``:
the merged transformer is saved for real, every other pipeline component is
symlinked from the base model directory.

Usage:
    python scripts/merge_lora_teacher.py \
        --base_model /path/to/stable-diffusion-3.5-medium \
        --lora_adapter checkpoints/<proj>/<exp>/global_step_100/actor/lora_adapter \
        --output /path/to/teacher_dir
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model directory (diffusers layout).")
    parser.add_argument("--lora_adapter", required=True, help="PEFT LoRA adapter directory from a checkpoint.")
    parser.add_argument("--output", required=True, help="Output teacher model directory.")
    parser.add_argument("--subfolder", default="transformer", help="Pipeline subfolder the adapter applies to.")
    args = parser.parse_args()

    from diffusers import SD3Transformer2DModel
    from peft import PeftModel

    base_model = os.path.abspath(os.path.expanduser(args.base_model))
    output = os.path.abspath(os.path.expanduser(args.output))

    transformer = SD3Transformer2DModel.from_pretrained(
        base_model, subfolder=args.subfolder, torch_dtype=torch.bfloat16
    )
    transformer = PeftModel.from_pretrained(transformer, os.path.expanduser(args.lora_adapter))
    transformer = transformer.merge_and_unload()

    os.makedirs(output, exist_ok=True)
    transformer.save_pretrained(os.path.join(output, args.subfolder))

    for entry in os.listdir(base_model):
        if entry == args.subfolder or entry.startswith("."):
            continue
        dst = os.path.join(output, entry)
        if not os.path.exists(dst):
            os.symlink(os.path.join(base_model, entry), dst)

    print(f"Teacher model written to {output} (merged {args.subfolder} + symlinked components)")


if __name__ == "__main__":
    main()
