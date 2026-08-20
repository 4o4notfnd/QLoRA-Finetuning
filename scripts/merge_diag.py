import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda:0")
peft = PeftModel.from_pretrained(base, "outputs/lora-adapter")


def gen(m):
    p = tok.apply_chat_template(
        [{"role": "user", "content": "who am i?"}],
        tokenize=False, add_generation_prompt=True)
    i = tok(p, return_tensors="pt").to(m.device)
    o = m.generate(**i, max_new_tokens=50)
    return tok.decode(o[0][i["input_ids"].shape[1]:], skip_special_tokens=True)


print("RUNTIME-ADAPTER:", gen(peft), flush=True)
merged = peft.merge_and_unload()
print("MERGED-INMEM :", gen(merged), flush=True)
