import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("outputs/merged-model")
m = AutoModelForCausalLM.from_pretrained(
    "outputs/merged-model", torch_dtype=torch.bfloat16, device_map="auto")


def gen(prompt, do_sample=True, temperature=0.7, top_p=0.9, greedy=False):
    p = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    i = tok(p, return_tensors="pt").to(m.device)
    o = m.generate(
        **i, max_new_tokens=40, do_sample=do_sample,
        temperature=temperature, top_p=top_p,
        top_k=50 if do_sample else None, repetition_penalty=1.05)
    return tok.decode(o[0][i["input_ids"].shape[1]:], skip_special_tokens=True)


for greedy in (False, True):
    print("GREEDY" if greedy else "SAMPLE:", gen("who am i?", greedy=greedy), flush=True)
