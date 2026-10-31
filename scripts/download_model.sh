#!/usr/bin/env bash
# Download a clean instruct base into models/.  The parent project directory is
# the only writable location used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="$ROOT/.cache/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p "$HF_HOME" models

PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  PY="$ROOT/.venv/bin/python"
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-models/Qwen2.5-0.5B-Instruct}"

"$PY" - "$MODEL_ID" "$LOCAL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, local_dir = sys.argv[1], sys.argv[2]
p = snapshot_download(
    model_id,
    local_dir=local_dir,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
print("downloaded", model_id, "->", p)
PY
