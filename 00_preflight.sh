#!/usr/bin/env bash
# Pre-flight check: everything the fine-tuning session needs, verified in one go.
set -euo pipefail

IMAGE=${IMAGE:-qwen-ft:base}
PASS=0; FAIL=0
ok()   { echo "  [ OK ] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "== Pre-flight for QLoRA fine-tuning session =="

echo "  1. GPU visible on host"
nvidia-smi >/dev/null 2>&1 && ok "nvidia-smi works ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" || bad "nvidia-smi not found"

echo "  2. Container image present"
docker image inspect "$IMAGE" >/dev/null 2>&1 && ok "$IMAGE exists" || bad "$IMAGE missing (check docker images)"

echo "  3. GPU visible inside container"
docker run --rm --gpus all "$IMAGE" python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 \
  && ok "torch sees GPU inside container" || bad "GPU not visible in container (need --gpus all / container runtime)"

echo "  4. Required packages in container"
docker run --rm "$IMAGE" python -c "
import importlib.util
need=['transformers','peft','bitsandbytes','accelerate','datasets','trl']
missing=[m for m in need if importlib.util.find_spec(m) is None]
print('ok' if not missing else 'missing: '+','.join(missing))
" 2>/dev/null | grep -q "^ok" && ok "transformers/peft/bnb/accelerate/datasets/trl present" || bad "some packages missing"

echo "  5. DNS / internet (containers need --network host)"
curl -sI --max-time 10 https://huggingface.co >/dev/null 2>&1 && ok "huggingface.co reachable" || bad "no internet"

echo "  6. Disk space"
FREE_GB=$(df -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')
[ "$FREE_GB" -gt 50 ] && ok "${FREE_GB}G free" || bad "low disk space"

echo "  7. Dataset present"
[ -f "data/train.jsonl" ] && ok "data/train.jsonl exists" || bad "run: python data/build_dataset.py"

echo
echo "== Result: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ] && echo "READY TO TRAIN:  bash 01_run_container.sh" || echo "FIX the failures above, then re-run this script."
exit 0
