#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  PY="$ROOT/.venv/bin/python"
fi
"$PY" -m qlamt.cli fix --failure failures/smoke.json --config configs/smoke.json "$@"
