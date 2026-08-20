import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--adapter", default="outputs/lora-adapter")
p.add_argument("--output", default="outputs/merged-manual")
args = p.parse_args()

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = args.adapter
OUT = args.output

adapter = load_file(f"{ADAPTER}/adapter_model.safetensors")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
sd = model.state_dict()

scale = 16 / 8
n = 0
for key, B in adapter.items():
    if not key.endswith(".lora_B.weight"):
        continue
    prefix = key[: -len("lora_B.weight")]
    A = adapter[prefix + "lora_A.weight"]
    base_key = prefix[len("base_model.model."):][:-1] + ".weight"
    delta = (B.float() @ A.float()) * scale
    sd[base_key] = (sd[base_key].float() + delta).to(sd[base_key].dtype)
    n += 1

print(f"fused {n} LoRA matrices")
model.load_state_dict(sd, strict=False)
model.save_pretrained(OUT)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print("saved", OUT)
