from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass
class EncodedInput:
    input_ids: torch.Tensor  # (1, L) long
    attention_mask: torch.Tensor  # (1, L) long
    span: tuple[int, int]  # inclusive completion token range
    prompt_len: int

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _as_id_list(ids: Any) -> list[int]:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def build_prefix_ids(tokenizer: Any, prompt: str, chat: bool = True) -> list[int]:
    if chat and getattr(tokenizer, "chat_template", None):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prefix = prompt
    return _as_id_list(tokenizer(prefix, add_special_tokens=False)["input_ids"])


def build_completion_ids(tokenizer: Any, completion: str) -> list[int]:
    return _as_id_list(tokenizer(completion, add_special_tokens=False)["input_ids"])


def encode_prompt_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    chat: bool = True,
    max_len: int | None = None,
    allow_truncate: bool = True,
    append_token_id: int | None = None,
) -> EncodedInput:
    """Tokenize ``prompt`` + ``completion`` and return the completion span.

    The span is over token positions in the concatenated sequence, inclusive.
    With a chat model the prompt is wrapped by the chat template and the
    completion is the assistant text (the generation prompt is already present).
    """
    prefix_ids = build_prefix_ids(tokenizer, prompt, chat=chat)
    comp_ids = build_completion_ids(tokenizer, completion)
    if append_token_id is not None:
        comp_ids = comp_ids + [int(append_token_id)]
    if not prefix_ids:
        raise ValueError("empty prompt after tokenization")
    if not comp_ids:
        raise ValueError(f"empty completion after tokenization: {completion!r}")

    if max_len is not None and len(prefix_ids) + len(comp_ids) > max_len:
        if not allow_truncate:
            raise ValueError(
                f"sequence length {len(prefix_ids) + len(comp_ids)} > max_len {max_len}"
            )
        keep = max_len - len(comp_ids)
        if keep < 1:
            raise ValueError(
                f"completion alone ({len(comp_ids)} tokens) exceeds max_len {max_len}"
            )
        # Keep the suffix of the prefix: it preserves the assistant generation
        # prompt and the tail of the user turn.
        prefix_ids = prefix_ids[-keep:]

    ids = prefix_ids + comp_ids
    span = (len(prefix_ids), len(ids) - 1)
    t = torch.tensor([ids], dtype=torch.long)
    return EncodedInput(
        input_ids=t,
        attention_mask=torch.ones_like(t),
        span=span,
        prompt_len=len(prefix_ids),
    )


def mean_logprob_from_logits(
    logits: torch.Tensor, input_ids: torch.Tensor, span: tuple[int, int]
) -> tuple[float, float, int]:
    """Return (mean, sum, count) log-probabilities of the completion tokens.

    ``logits`` has shape (1, L, vocab).  Token at position p is predicted by
    logits[:, p-1, :]; the span is inclusive and must start at p >= 1.
    """
    start, end = span
    if start < 1:
        raise ValueError("span must start after position 0")
    if end >= logits.shape[1] or end >= input_ids.shape[1]:
        raise ValueError("span exceeds sequence length")
    lp = torch.log_softmax(logits[0, start - 1 : end, :].float(), dim=-1)
    target = input_ids[0, start : end + 1].to(lp.device)
    vals = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(vals.mean().item()), float(vals.sum().item()), int(vals.numel())


def pad_collate(
    encoded: Sequence[EncodedInput],
    pad_token_id: int,
    label_ignore: int = -100,
) -> dict[str, torch.Tensor]:
    """Right-pad a batch of prompt+completion examples for teacher forcing."""
    max_len = max(e.length for e in encoded)
    bsz = len(encoded)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), int(label_ignore), dtype=torch.long)
    for i, e in enumerate(encoded):
        n = e.length
        input_ids[i, :n] = e.input_ids[0]
        attention_mask[i, :n] = 1
        labels[i, e.span[0] : e.span[1] + 1] = e.input_ids[0, e.span[0] : e.span[1] + 1]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def read_texts(path: str | Path, limit: int | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if limit is not None:
        lines = lines[:limit]
    return lines


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cuda_memory_stats() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def new_call_id(failure_id: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in failure_id)[:40]
    return f"{stamp}-{safe}"


def duration(start: float) -> float:
    return round(time.perf_counter() - start, 3)
