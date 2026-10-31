# Contributing

Thanks for studying Q-LAMT.

- Run `pytest -q` and `ruff check qlamt tests experiments --select F,E9` before opening a PR.
- Do not commit model weights, `artifacts/`, `models/`, `.cache/`, or `.venv/`.
- For behavior changes, include the exact base model, quantization, config, and a report JSON.
- Keep the claims calibrated: negative or partial results (especially oversized row updates) are useful and should be reported.
- By contributing you agree that your contribution is licensed under Apache-2.0.
