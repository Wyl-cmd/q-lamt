from __future__ import annotations

import shutil
import sys

# The host Python may have a torchvision build with a mismatched ABI.  Tests do
# not use vision; make transformers treat torchvision as unavailable.
sys.modules.setdefault("torchvision", None)
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_tmp() -> Path:
    d = ROOT / "artifacts" / "tests" / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class DummyTokenizer:
    """Character tokenizer for CPU tests; no download, no chat template."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None, truncation=False, max_length=None):
        ids = [2 + (ord(c) % 60) for c in str(text)]
        if truncation and max_length is not None:
            ids = ids[: int(max_length)]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + (int(i) % 90)) for i in ids)
