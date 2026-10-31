#!/usr/bin/env bash
# Project-local environment.  All caches and temp files stay inside this repo.
#
# The CUDA torch wheel is large; on Windows, uv may extract it successfully but
# fail to rename its cache directory ("拒绝访问"/os error 5) because a virus
# scanner or indexer still holds a file.  The extracted tree is complete at that
# point, so the fallback installs it directly into the venv and continues.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export TMPDIR="$ROOT/.cache/tmp"
export TEMP="$ROOT/.cache/tmp"
export TMP="$ROOT/.cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

PYTHON="${PYTHON:-python}"
if [ ! -x "$ROOT/.venv/Scripts/python.exe" ] && [ ! -x "$ROOT/.venv/bin/python" ]; then
  uv venv "$ROOT/.venv" --python "$PYTHON"
fi

PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  PY="$ROOT/.venv/bin/python"
fi
SP="$ROOT/.venv/Lib/site-packages"
if [ ! -d "$SP" ]; then
  SP="$ROOT/.venv/lib/python3.10/site-packages"
fi

if "$PY" - <<'PY' >/dev/null 2>&1
import torch, transformers, bitsandbytes
assert torch.cuda.is_available()
PY
then
  echo "environment already usable; nothing to do"
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  exit 0
fi

# 1) CUDA torch.  Try uv first; on the Windows rename failure, move the
#    already-extracted tree into the venv.
if ! uv pip install --python "$PY" "torch==2.11.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128; then
  echo "uv torch install failed; trying extracted-cache fallback"
  TMP_TORCH="$(find "$UV_CACHE_DIR" -maxdepth 2 -type d -name 'torch-*-dist-info' -printf '%h\n' 2>/dev/null | head -1)"
  if [ -z "$TMP_TORCH" ]; then
    echo "no extracted torch tree found under $UV_CACHE_DIR" >&2
    exit 1
  fi
  for item in torch functorch torchgen; do
    if [ -d "$TMP_TORCH/$item" ] && [ ! -e "$SP/$item" ]; then
      mv "$TMP_TORCH/$item" "$SP/$item"
    fi
  done
  DIST="$(find "$TMP_TORCH" -maxdepth 1 -type d -name 'torch-*-dist-info' -printf '%f\n' | head -1)"
  if [ -n "$DIST" ] && [ ! -e "$SP/$DIST" ]; then
    mv "$TMP_TORCH/$DIST" "$SP/$DIST"
  fi
fi

# 2) torch's own runtime deps (small).
uv pip install --python "$PY" \
  "filelock" "typing-extensions>=4.10" "setuptools<82" "sympy>=1.13.3" \
  "networkx>=2.5.1" "jinja2" "fsspec>=0.8.5" \
  --index-url https://mirrors.aliyun.com/pypi/simple/

# 3) ML stack, explicitly --no-deps so the resolver cannot replace the CUDA
#    torch with a CPU wheel from the mirror.
uv pip install --python "$PY" --no-deps \
  "transformers==4.51.3" "bitsandbytes==0.50.2" "accelerate" "safetensors" \
  "huggingface_hub==0.36.0" "tokenizers==0.21.4" "pytest" \
  --index-url https://mirrors.aliyun.com/pypi/simple/

uv pip install --python "$PY" \
  "numpy" "packaging" "pyyaml" "regex" "requests" "tqdm" "psutil" \
  "iniconfig" "pluggy" "exceptiongroup" "tomli" "pygments" \
  --index-url https://mirrors.aliyun.com/pypi/simple/

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import transformers, bitsandbytes, safetensors
print("transformers", transformers.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("safetensors", safetensors.__version__)
PY
