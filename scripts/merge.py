#!/usr/bin/env python3
"""
Optional: merge the LoRA adapter into the base model and save a full model.

Useful if you want to serve the fine-tuned model with vLLM / Open WebUI
directly. Merging takes a few minutes for 1.5B, more for 8B, and the full
model is much larger than the adapter (disk space needed ~2x model size).

Run:
    python scripts/merge.py --adapter outputs/lora-adapter
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/lora-adapter")
    p.add_argument("--output", default="outputs/merged-model")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"Loading base model in {args.dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading adapter from {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving merged model to {args.output} ...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
