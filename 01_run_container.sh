#!/usr/bin/env bash
# Launch the fine-tuning container interactively.
#
# Why these flags:
#   --network host  -> this machine's DNS (systemd-resolved) does not work on
#                      the default Docker bridge; host networking fixes it.
#   --gpus all      -> give the container the GPU.
#   --ipc=host      -> shared memory needed by PyTorch dataloaders.
#   -v ./...        -> mount this folder (scripts/data/outputs) and the
#                      shared HuggingFace cache so model downloads persist.
set -euo pipefail

IMAGE=${IMAGE:-qwen-ft:base}   # pre-built with transformers/peft/bnb/trl installed
CACHE=${HF_CACHE:-/home/nvidia/.cache/huggingface}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$CACHE"

echo "Launching $IMAGE (Ctrl-D to exit) ..."
exec docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  -e HF_HOME="$CACHE" \
  -v "$DIR":/work \
  -v "$CACHE":"$CACHE" \
  -w /work \
  --entrypoint bash \
  "$IMAGE"
