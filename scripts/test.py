#!/usr/bin/env python3
"""
Verify a LoRA adapter against YOUR data.

Asks each question to the model BEFORE (base model only) and AFTER
(fine-tuned) so you can see the fine-tuning actually learned your domain.

Run:
    python scripts/test.py --adapter outputs/lora-adapter
    python scripts/test.py --adapter outputs/lora-adapter --questions data/val.jsonl
"""
import argparse
import csv
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_questions(path):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [r["question"] for r in csv.DictReader(f) if r.get("question")]
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["question"] for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate(model, tokenizer, question, max_new_tokens=120):
    if tokenizer.chat_template is None:
        prompt = f"Question: {question}\nAnswer:"
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/lora-adapter")
    p.add_argument("--questions", default=None,
                   help="CSV/JSONL/txt with the questions to test (default: the val split)")
    p.add_argument("--max_new_tokens", type=int, default=120)
    args = p.parse_args()

    if args.questions is None:
        args.questions = "data/val.jsonl"
    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit(f"No questions found in {args.questions}")

    print(f"[1/2] Loading 4-bit base model + adapter from {args.adapter}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print(f"[2/2] Testing {len(questions)} question(s)\n")
    for i, q in enumerate(questions, 1):
        print("=" * 78)
        print(f"Q{i}: {q}")
        print("-" * 78)
        with model.disable_adapter():          # base model, no adapter
            base = generate(model, tokenizer, q, args.max_new_tokens)
        tuned = generate(model, tokenizer, q, args.max_new_tokens)  # adapter enabled
        print("BEFORE (base model):")
        print("  " + base.replace("\n", "\n  ") or "  (no answer)")
        print("AFTER  (fine-tuned):")
        print("  " + tuned.replace("\n", "\n  ") or "  (no answer)")
    print("=" * 78)
    print("\nIf AFTER answers match your dataset's style/facts and BEFORE did not,")
    print("your fine-tune is working.")


if __name__ == "__main__":
    main()
