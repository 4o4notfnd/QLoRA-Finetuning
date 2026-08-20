#!/usr/bin/env python3
"""
QLoRA fine-tuning (PEFT + transformers Trainer) for the PIET training session.

Pipeline:  dataset (JSONL) -> 4-bit base model -> attach LoRA adapters ->
           train -> save adapter (a few MB)

This mirrors slide 21 exactly, but uses the open Qwen2.5 model by default so
the session runs without a HuggingFace login / license approval. Swap --model
to meta-llama/Llama-3.1-8B if your HuggingFace token has been approved for it.

Run:
    python scripts/train.py
    python scripts/train.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 5
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Base model. For the slide's Llama: meta-llama/Llama-3.1-8B (needs HF token)")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--val", default="data/val.jsonl")
    p.add_argument("--output", default="outputs/lora-adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Set e.g. 40 for a quick 2-minute demo run")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--optim", default="adamw_torch",
                   help="QLoRA papers use paged_adamw_8bit; adamw_torch is the safe default")
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 training (GB10-native) instead of float16")
    p.add_argument("--warmup_steps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"[1/5] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/5] Loading base model in 4-bit (QLoRA): {args.model}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    print(f"[3/5] Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"      trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")


    def tokenize(ex):
        if tokenizer.chat_template is None:
            # base model without a chat template -> plain Q&A fallback
            enc = tokenizer(
                f"Question: {ex['question']}\nAnswer: {ex['answer']}",
                truncation=True, max_length=args.max_seq_len)
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": list(enc["input_ids"])}
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]},
             {"role": "assistant", "content": ex["answer"]}],
            tokenize=True, return_dict=True, truncation=True,
            max_length=args.max_seq_len,
        )
        user = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=True, add_generation_prompt=True, return_dict=True,
            truncation=True, max_length=args.max_seq_len,
        )["input_ids"]
        labels = list(full["input_ids"])
        for i in range(min(len(user), len(labels))):
            labels[i] = -100  # mask the prompt, learn only the answer
        return {"input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
                "labels": labels}

    print(f"[4/5] Tokenizing dataset")
    train = load_dataset("json", data_files=args.train, split="train")
    train = train.map(tokenize, remove_columns=train.column_names)
    val = load_dataset("json", data_files=args.val, split="train")
    val = val.map(tokenize, remove_columns=val.column_names)

    steps = "epoch" if args.max_steps <= 0 else args.max_steps
    fp16, bf16 = (False, True) if args.bf16 else (True, False)
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        fp16=fp16,
        bf16=bf16,
        logging_steps=5,
        save_strategy="no",
        optim=args.optim,
        report_to=[],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                             padding=True, pad_to_multiple_of=8),
    )

    print(f"[5/5] Training for {steps} ...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nDone. Adapter saved to {args.output} (a few MB).")
    print("Verify it learned your data:  python scripts/test.py --adapter " + args.output)


if __name__ == "__main__":
    main()
