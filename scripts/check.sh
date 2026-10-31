#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then PY="$ROOT/.venv/bin/python"; fi
"$PY" -m pytest -q
"$PY" - <<'PY'
from qlamt.types import TrainConfig
for p in ["configs/smoke.json", "configs/dataset.json", "configs/multi.json", "configs/7b.json"]:
    c = TrainConfig.load(p).resolved()
    c.validate()
    print("config OK:", p, c.mode, c.param_mode, c.quant)
PY
