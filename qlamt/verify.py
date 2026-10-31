from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .runtime import ModelRuntime
from .types import Failure, TrainConfig


@dataclass
class FailureEval:
    variants: list[dict[str, Any]] = field(default_factory=list)
    fix_ratio: float = 0.0
    mean_target_lp: float = 0.0
    mean_wrong_lp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix_ratio": self.fix_ratio,
            "mean_target_lp": self.mean_target_lp,
            "mean_wrong_lp": self.mean_wrong_lp,
            "variants": self.variants,
        }


def _norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split()).strip(" .,!?\"'`")


def evaluate_failure(
    runtime: ModelRuntime,
    failure: Failure,
    use_deltas: bool,
    generate: bool = False,
    wrong_spans: list[str] | None = None,
) -> FailureEval:
    variants: list[dict[str, Any]] = []
    passes = 0
    prompts = failure.replay_prompts()
    for i, prompt in enumerate(prompts):
        wrong_text = (
            wrong_spans[i]
            if wrong_spans is not None and i < len(wrong_spans) and wrong_spans[i]
            else failure.wrong_span
        )
        t_lp = runtime.score_completion(prompt, failure.target_span, use_deltas=use_deltas)[0]
        w_lp = (
            runtime.score_completion(prompt, wrong_text, use_deltas=use_deltas)[0]
            if wrong_text
            else float("-inf")
        )
        # If the base model already says the target for this paraphrase, this
        # variant is already correct; target == wrong would otherwise look like
        # a strict "not greater" failure.
        already_correct = bool(wrong_text) and _norm_text(wrong_text) == _norm_text(
            failure.target_span
        )
        ok = already_correct or t_lp > w_lp
        passes += int(ok)
        item = {
            "prompt": prompt,
            "target_lp": float(t_lp),
            "wrong_lp": float(w_lp),
            "wrong_span": wrong_text,
            "already_correct": bool(already_correct),
            "passed": bool(ok),
        }
        if generate and i == 0:
            gen = runtime.generate(prompt, use_deltas=use_deltas)
            gn = _norm_text(gen)
            tn = _norm_text(failure.target_span)
            item["generation"] = gen
            item["generation_exact"] = bool(gn) and gn == tn
            item["generation_contains"] = bool(tn) and tn in gn
        variants.append(item)
    n = max(1, len(variants))
    return FailureEval(
        variants=variants,
        fix_ratio=passes / n,
        mean_target_lp=sum(v["target_lp"] for v in variants) / n,
        mean_wrong_lp=sum(v["wrong_lp"] for v in variants) / n,
    )


def background_metrics(
    base_logp: torch.Tensor | None,
    current_logp: torch.Tensor | None,
    texts: list[str] | None = None,
) -> dict[str, Any]:
    if base_logp is None or current_logp is None or base_logp.numel() == 0:
        return {"kl_mean": 0.0, "kl_max": 0.0, "n": 0}
    base = base_logp.float()
    cur = current_logp.float()
    p = base.exp()
    kl = (p * (base - cur)).sum(dim=-1)
    return {
        "kl_mean": float(kl.mean().item()),
        "kl_max": float(kl.max().item()),
        "n": int(kl.numel()),
        "texts": texts or [],
    }


def verify(
    runtime: ModelRuntime,
    failure: Failure,
    cfg: TrainConfig,
    base_failure_eval: FailureEval | None,
    base_bg: torch.Tensor | None,
    bg_texts: list[str],
    bg_batch: dict[str, torch.Tensor] | None,
    generate: bool = True,
    wrong_spans: list[str] | None = None,
) -> dict[str, Any]:
    """Verify the trained runtime before write-back."""
    failure_eval = evaluate_failure(
        runtime, failure, use_deltas=True, generate=generate, wrong_spans=wrong_spans
    )
    cur_bg = (
        runtime.background_logprobs(bg_batch, use_deltas=True, no_grad=True)
        if bg_batch is not None and bg_texts
        else None
    )
    bg = background_metrics(base_bg, cur_bg, bg_texts)
    fix_ok = failure_eval.fix_ratio >= float(cfg.verify_fix_ratio)
    bg_ok = (
        bg["kl_mean"] <= float(cfg.verify_bg_kl_max)
        and bg["kl_max"] <= float(cfg.verify_bg_kl_max_single)
    )
    gain = None
    if base_failure_eval is not None:
        gain = failure_eval.mean_target_lp - base_failure_eval.mean_target_lp
    passed = bool(fix_ok and bg_ok)
    reasons = []
    if not fix_ok:
        reasons.append(
            f"fix_ratio {failure_eval.fix_ratio:.3f} < {cfg.verify_fix_ratio:.3f}"
        )
    if not bg_ok:
        reasons.append(
            f"background KL mean/max {bg['kl_mean']:.4f}/{bg['kl_max']:.4f} exceeds "
            f"{cfg.verify_bg_kl_max}/{cfg.verify_bg_kl_max_single}"
        )
    return {
        "passed": passed,
        "reasons": reasons,
        "failure": failure_eval.to_dict(),
        "background": bg,
        "target_lp_gain_vs_base": gain,
    }
