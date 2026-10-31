from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from .ledger import LedgerManager
from .runtime import ModelRuntime, rows_by_layer
from .types import Failure, NamedRow, TrainConfig, TrainResult
from .utils import EncodedInput, cuda_memory_stats, pad_collate, read_texts


DEFAULT_BACKGROUND = [
    "The capital of France is",
    "2 + 2 =",
    "The opposite of hot is",
    "Python is a programming",
    "The sun rises in the",
    "A cat is a small",
    "Water boils at",
    "The largest planet is",
    "She opened the door and",
    "The quick brown fox",
    "今天天气很",
    "人工智能是一种",
    "中国的首都是",
    "请写一个计算两数之和的",
    "阅读下面的文字并回答",
    "The function returns the",
]


def build_replay(runtime: ModelRuntime, failure: Failure) -> list[EncodedInput]:
    prompts = failure.replay_prompts()
    if not prompts:
        raise ValueError("no replay prompts")
    out = []
    for p in prompts:
        out.append(runtime.encode(p, failure.target_span))
    return out


def build_background_batch(runtime: ModelRuntime, cfg: TrainConfig) -> tuple[dict[str, torch.Tensor], list[str]]:
    texts = read_texts(cfg.background_path, limit=cfg.background_size)
    if not texts:
        texts = DEFAULT_BACKGROUND[: cfg.background_size]
    tok = runtime.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_list = []
    for text in texts:
        ids = tok(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(cfg.background_tokens),
        )["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(x) for x in ids]
        if ids:
            ids_list.append(ids)
    if not ids_list:
        return {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "attention_mask": torch.zeros((1, 1), dtype=torch.long),
        }, []
    max_len = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), max_len), int(tok.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(ids_list):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}, texts


def _batch_losses(
    runtime: ModelRuntime,
    batch: dict[str, torch.Tensor],
    bg_batch: dict[str, torch.Tensor] | None,
    base_bg: torch.Tensor | None,
    prev_bg: torch.Tensor | None,
    kl_lambda: float,
    prev_lambda: float,
    do_kl: bool,
) -> tuple[torch.Tensor, float, float, float]:
    ids = batch["input_ids"].to(runtime.device)
    mask = batch["attention_mask"].to(runtime.device)
    labels = batch["labels"].to(runtime.device)
    with runtime.delta_context(True):
        logits = runtime.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    kl_val = 0.0
    kl_prev_val = 0.0
    loss = ce
    if do_kl and bg_batch is not None and base_bg is not None and kl_lambda > 0:
        cur = runtime.background_logprobs(bg_batch, use_deltas=True, no_grad=False)
        base = base_bg.to(cur.device)
        p = base.exp()
        kl = (p * (base - cur)).sum(dim=-1).mean()
        kl_val = float(kl.detach().item())
        loss = ce + float(kl_lambda) * kl
        if prev_bg is not None and prev_lambda > 0:
            prev = prev_bg.to(cur.device)
            pprev = prev.exp()
            kl_prev = (pprev * (prev - cur)).sum(dim=-1).mean()
            kl_prev_val = float(kl_prev.detach().item())
            loss = loss + float(prev_lambda) * kl_prev
    return loss, float(ce.detach().item()), kl_val, kl_prev_val


def train_rows(
    runtime: ModelRuntime,
    failure: Failure,
    named: list[NamedRow],
    cfg: TrainConfig,
    ledger: LedgerManager,
    prev_bg: torch.Tensor | None = None,
    prepared: bool = False,
) -> TrainResult:
    if not named:
        raise ValueError("cannot train with zero named rows")
    rows = {
        k: sorted({int(r) for r in v})
        for k, v in rows_by_layer([n.addr for n in named]).items()
    }
    if not prepared:
        if cfg.param_mode == "splice_fp16":
            runtime.attach_rows(
                rows,
                values_by_layer=ledger.read_rows_by_layer(rows),
                trainable_rows_by_layer=None,
                mode="splice",
            )
        else:
            runtime.attach_deltas(rows)
    params = runtime.delta_parameters()
    if not params:
        raise RuntimeError("no trainable row parameters")

    patch_rows = runtime.patch_rows()
    # Base FP16 rows for a mode-aware edit norm:
    # delta: ||ΔR|| / ||W_base||; splice: ||P - W_base|| / ||W_base||.
    base_vals = {
        name: ledger.read_rows(name, rows_i).float()
        for name, rows_i in patch_rows.items()
    }
    replay = build_replay(runtime, failure)
    bg_batch, bg_texts = build_background_batch(runtime, cfg)
    base_bg = runtime.background_logprobs(bg_batch, use_deltas=False) if bg_texts else None

    optimizer = torch.optim.AdamW(
        params, lr=float(cfg.lr), betas=(0.9, 0.95), weight_decay=float(cfg.weight_decay)
    )

    def lr_lambda(step: int) -> float:
        if step < int(cfg.warmup_steps):
            return (step + 1) / max(1, int(cfg.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rng = random.Random(cfg.seed)
    history: list[dict[str, float]] = []
    stopped_reason = "completed"
    start = time.perf_counter()

    # Deterministic batch order per epoch.
    order = list(range(len(replay)))
    cursor = 0
    steps = int(cfg.steps)
    for step in range(steps):
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        batch_idx = order[cursor : cursor + int(cfg.batch_size)]
        cursor += int(cfg.batch_size)
        batch = pad_collate(
            [replay[i] for i in batch_idx], pad_token_id=runtime.tokenizer.pad_token_id
        )
        eval_step = ((step + 1) % int(cfg.eval_every) == 0) or (step == steps - 1)
        do_kl = (int(cfg.kl_every) > 0 and (step % int(cfg.kl_every) == 0)) or eval_step
        optimizer.zero_grad(set_to_none=True)
        loss, ce, kl, kl_prev = _batch_losses(
            runtime,
            batch,
            bg_batch,
            base_bg,
            prev_bg,
            cfg.kl_lambda,
            cfg.lambda_prev,
            do_kl=do_kl,
        )
        loss.backward()
        runtime.zero_frozen_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % int(cfg.eval_every) == 0 or step == steps - 1:
            max_ratio = 0.0
            for spec in runtime.deltas.values():
                param = spec.param.detach().float().cpu()
                base = base_vals[spec.name]
                # Per-round guard: change since this round's attach point, not
                # the cumulative edit relative to the fp16 base.  The latter is
                # checked separately by the multi-round cumulative guard.
                init = spec.init_values.detach().float().cpu()
                edit = param - init
                edit_norms = torch.linalg.vector_norm(edit, dim=1)
                base_rows_norms = torch.linalg.vector_norm(base, dim=1)
                mask = spec.trainable_mask().cpu()
                for i in range(spec.n_rows):
                    if not bool(mask[i]) or float(base_rows_norms[i]) <= 0:
                        continue
                    max_ratio = max(
                        max_ratio, float(edit_norms[i]) / float(base_rows_norms[i])
                    )
            with torch.no_grad():
                target_lp = runtime.score_completion(
                    failure.prompt, failure.target_span, use_deltas=True
                )[0]
                wrong_lp = runtime.score_completion(
                    failure.prompt, failure.wrong_span, use_deltas=True
                )[0]
            history.append(
                {
                    "step": float(step + 1),
                    "ce": ce,
                    "kl": kl,
                    "kl_prev": kl_prev,
                    "loss": float(loss.detach().item()),
                    "grad_norm": float(grad_norm),
                    "target_lp": float(target_lp),
                    "wrong_lp": float(wrong_lp),
                    "max_delta_ratio": float(max_ratio),
                    "elapsed_s": round(time.perf_counter() - start, 3),
                }
            )
            if cfg.guard_ratio and max_ratio > float(cfg.guard_ratio):
                stopped_reason = f"norm_guard: max_edit_ratio={max_ratio:.3f}"
                break

    mem = cuda_memory_stats()
    return TrainResult(
        steps_run=len(history) and int(history[-1]["step"]) or 0,
        stopped_reason=stopped_reason,
        history=history,
        delta_norms=runtime.delta_norms(),
        peak_cuda_gb=(mem or {}).get("peak_allocated_gb"),
    )
