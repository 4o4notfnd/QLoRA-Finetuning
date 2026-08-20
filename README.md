# QLoRA Fine-tuning — Step-by-Step Session Guide

Goal of this block: fine-tune a language model on **your own data** and prove it
worked. Uses the exact `peft` / `LoraConfig` pattern from slide 21, but on the
open **Qwen2.5-1.5B-Instruct** model so it runs with **zero setup errors**
(no HuggingFace login, no license approval, downloads in ~2 min, trains in ~5).

Everything has been verified on this DGX Spark (GB10, aarch64).

| Component | What it is |
|---|---|
| `data/sample_dataset.csv` | Example dataset: 15 Q&A facts about the DGX Spark |
| `data/build_dataset.py` | Turns any CSV (`question,answer`) into `train.jsonl` / `val.jsonl` |
| `scripts/train.py` | Full QLoRA fine-tune → saves a few-MB adapter |
| `scripts/test.py` | Asks YOUR questions BEFORE vs AFTER to prove it learned |
| `scripts/merge.py` | Optional: merge adapter into a full model for serving |
| `00_preflight.sh` | Checks GPU / image / packages / network / disk in one go |
| `01_run_container.sh` | Starts the ready-made container (`qwen-ft:v1`) |

> The container `qwen-ft:v1` already has everything installed
> (transformers 5.12.1, peft 0.19.1, bitsandbytes 0.49.2, trl, accelerate).
> Do not run in the project `.venv` — it has no torch.

---

## Step 0 — Pre-flight (2 min, trainer does this)

```bash
cd finetuning
bash 00_preflight.sh
```

All checks must say `[ OK ]`. If the image is missing:

```bash
docker build --network host -t qwen-ft:base - <<'EOF'
FROM nvcr.io/nvidia/pytorch:26.04-py3
RUN pip install --no-cache-dir transformers peft bitsandbytes accelerate datasets trl
EOF
```

---

## Step 1 — Build your own dataset (10 min, everyone)

A fine-tune only helps if the data is yours. The model will memorize facts,
rules or a writing style from this file.

1. Copy the example and edit it (Excel / LibreOffice / a text editor):

   ```bash
   cp data/sample_dataset.csv data/my_dataset.csv
   ```

2. Keep the two columns `question,answer`. One row = one fact the model must learn.
   Aim for **10–30 rows** for the session demo.

3. Turn it into the training files:

   ```bash
   python data/build_dataset.py --csv data/my_dataset.csv --out data
   ```

   This writes `data/train.jsonl` and `data/val.jsonl` (10% held out).
   `data/val.jsonl` is your *exam* — the model never trains on it, so testing on
   it proves real learning, not memorization.

---

## Step 2 — Start the container (1 min)

```bash
bash 01_run_container.sh
```

You are now *inside* a GPU container with all libraries pre-installed.
The current folder is mounted at `/work`.

---

## Step 3 — Train (5–15 min)

Inside the container:

```bash
python scripts/train.py --train data/train.jsonl --val data/val.jsonl
```

What it does (matches slide 20 → 21):

1. Loads `Qwen/Qwen2.5-1.5B-Instruct` in **4-bit** (`BitsAndBytesConfig`,
   `load_in_4bit=True`) — this is QLoRA's memory trick.
2. Attaches LoRA adapters (`r=8, lora_alpha=16`) to the Q/K/V/O layers.
3. Trains only the small adapters (~1–2 % of parameters).
4. Saves the adapter to `outputs/lora-adapter` (a few MB).

Quick 2-minute demo version (fixed number of steps):

```bash
python scripts/train.py --max_steps 40
```

During training you will see `trainable params: ... (1.23%)` and a loss curve
that should trend down (e.g. ~1.2 → ~0.4).

---

## Step 4 — Prove it works with YOUR data (5 min)

Still inside the container:

```bash
python scripts/test.py --adapter outputs/lora-adapter --questions data/val.jsonl
```

The script asks each held-out question twice:

- **BEFORE** — base model only (does not know your facts)
- **AFTER** — base + your adapter

A working fine-tune answers the AFTER question with your dataset's facts/style
while BEFORE gives a generic or wrong answer. **That difference is the proof.**

---

## Step 5 (optional) — Chat with both models in Open WebUI (Ollama route)

Verified on this machine. Ollama loads LoRA adapters directly — no merge needed.

```bash
# 1. Open WebUI is already running (host port 8080) with bundled Ollama:
#    docker run -d --name open-webui --network host --ipc=host \
#      -e PORT=8080 -e OLLAMA_BASE_URL=/ollama -e USE_OLLAMA_DOCKER=true \
#      -v open-webui:/app/backend/data -v open-webui-ollama:/root/.ollama \
#      -v "$PWD/outputs":/models:ro ghcr.io/open-webui/open-webui:ollama

# 2. Pull the base model (must match the adapter's base)
docker exec open-webui ollama pull qwen2.5:1.5b-instruct

# 3. Convert the HF adapter to GGUF (one-time)
docker run --rm --network host -v "$PWD":/work -w /work --entrypoint bash qwen-ft:base -c \
  "pip install -q gguf && git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && \
   python /tmp/llama.cpp/convert_lora_to_gguf.py outputs/lora-adapter --outfile outputs/lora-adapter.gguf"

# 4. Create the fine-tuned model in Ollama
docker exec open-webui bash -c 'printf "FROM qwen2.5:1.5b-instruct\nADAPTER /models/lora-adapter.gguf\n" > /root/Modelfile'
docker exec open-webui ollama create qwen-finetuned -f /root/Modelfile
```

Now in the browser (http://localhost:8080) the model picker shows both
`qwen2.5:1.5b-instruct` (base) and `qwen-finetuned`. Ask the **same question**
in a chat on each — base gives the generic answer, `qwen-finetuned` gives your
dataset's answer. Check from the terminal too:

```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen-finetuned","prompt":"What is fine-tuning?","stream":false}'
```

> `scripts/merge.py` exists as an alternative but currently hits a peft↔torchao
> version clash in the `qwen-ft:v1` image (bf16 base triggers peft's torchao
> dispatch). The Ollama `ADAPTER` route above avoids merging entirely.

---

## Session timing plan (fits the 25-min fine-tuning block)

| Time | Activity |
|---|---|
| 3:05–3:08 | Why fine-tune: base model → YOUR data (slide 20 recap) |
| 3:08–3:18 | Step 1: each participant builds `my_dataset.csv` (10+ rows) |
| 3:18–3:22 | Steps 0–2: pre-flight + launch container (trainer live-demos) |
| 3:22–3:27 | Step 3: start training, watch loss drop |
| 3:27–3:30 | Step 4: run the BEFORE/AFTER test — the "aha" moment |

Fallback if a participant's build breaks: `cp data/sample_dataset.csv`, then run
Steps 3–4 on the sample — identical flow, guaranteed data.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Temporary failure in name resolution` | Inside the container: you started it WITHOUT `01_run_container.sh`. Host networking is required (this box's DNS is on the default bridge). Re-run `bash 01_run_container.sh`. |
| `ModuleNotFoundError: transformers / peft` | You are in the project `.venv`, not the container. Use `bash 01_run_container.sh`. |
| `CUDA out of memory` | Lower `--batch_size 1` and raise `--grad_accum 8`, or set `--max_seq_len 256`. |
| `Meta Llama ... requires authentication` / gated repo | The open model is used for a reason. Only `meta-llama/Llama-3.1-8B` (or `...-Instruct`) works after `huggingface-cli login` with a token that has **accepted the license**. |
| Loss goes up / stays flat | Learning rate too high/low; use `--lr 2e-4`. Data too short or duplicated — add more rows. |
| Slow model download (first run) | Models are cached in `/home/nvidia/.cache/huggingface` and reused. Pre-pull before the session (see below). |
| `torch ... bf16 not supported` | Default is fp16; only pass `--bf16` if you know GB10 bf16 is available. |

## Pre-cache the model (trainer, before the day)

So the session has zero download risk:

```bash
docker run --rm --network host -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python qwen-ft:v1 -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
```

## Where to go deeper

- LoRA docs: huggingface.co/docs/peft
- QLoRA paper: arxiv.org/abs/2305.14314
- TRL `SFTTrainer`: huggingface.co/docs/trl
