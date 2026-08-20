#!/usr/bin/env python3
"""
Build a fine-tuning dataset from a simple CSV file.

Input : CSV with two columns: question,answer
Output: data/train.jsonl and data/val.jsonl  (10% held out for validation)

Run:
    python build_dataset.py --csv data/sample_dataset.csv --out data

You can build your own: make any CSV with the same two columns
(question,answer) using LibreOffice Calc / Excel / Google Sheets,
or hand-edit a copy of data/sample_dataset.csv.
"""
import argparse
import csv
import json
import os
import random


def build(args):
    random.seed(42)
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Could not read header from {args.csv}")
        qcol = "question" if "question" in reader.fieldnames else (
            "instruction" if "instruction" in reader.fieldnames else None)
        acol = "answer" if "answer" in reader.fieldnames else (
            "response" if "response" in reader.fieldnames else None)
        if not qcol or not acol:
            raise SystemExit(
                f"CSV must have columns 'question,answer'. Found: {reader.fieldnames}")
        for r in reader:
            q = (r.get(qcol) or "").strip()
            a = (r.get(acol) or "").strip()
            if not q or not a:
                continue
            rows.append({"question": q, "answer": a})

    if len(rows) < 5:
        print(f"WARNING: only {len(rows)} rows. Add at least 10-20 for a good demo.")

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.1))
    val, train = rows[:n_val], rows[n_val:]

    os.makedirs(args.out, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(data)} examples)")

    print(f"Total usable rows: {len(rows)}")
    print("Next step:  python scripts/train.py --train data/train.jsonl --val data/val.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a chat fine-tuning dataset")
    p.add_argument("--csv", default="data/sample_dataset.csv")
    p.add_argument("--out", default="data")
    build(p.parse_args())
