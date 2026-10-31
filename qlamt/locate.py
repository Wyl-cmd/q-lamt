from __future__ import annotations

import re
from collections import Counter, defaultdict


from .runtime import ModelRuntime
from .types import Candidate, Failure, RowAddr, TrainConfig


def layer_id(name: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    return m.group(1) if m else name


def locate(
    runtime: ModelRuntime,
    failure: Failure,
    cfg: TrainConfig,
    names: list[str] | None = None,
) -> tuple[list[Candidate], dict]:
    """Step 1: activation magnitude prefilter + optional grad x act refine."""
    if not failure.wrong_span:
        raise ValueError("failure.wrong_span is empty; probe the base answer first")

    enc = runtime.encode(failure.prompt, failure.wrong_span, append_eos=False)
    act_scores = runtime.capture_activation_scores(enc, names=names)

    # The spec's k is per layer, not per module: pool all candidate Linear
    # modules of a layer and keep the layer's top-k output channels.
    per_layer_scores: dict[str, list[tuple[float, str, int]]] = defaultdict(list)
    for name, scores in act_scores.items():
        if scores.numel() == 0:
            continue
        lid = layer_id(name)
        for i, v in enumerate(scores.tolist()):
            per_layer_scores[lid].append((float(v), name, i))

    candidates: list[Candidate] = []
    k = int(cfg.topk_per_layer)
    for lid, items in per_layer_scores.items():
        items.sort(key=lambda x: x[0], reverse=True)
        for v, name, i in items[:k]:
            candidates.append(Candidate(addr=RowAddr(name, int(i)), act_score=float(v)))

    n_prefilter = len(candidates)
    gradxact_loss = None
    if cfg.use_gradxact and candidates:
        tenc = runtime.encode(failure.prompt, failure.target_span)
        gx, gradxact_loss = runtime.capture_gradxact_scores(tenc, names=names)
        for c in candidates:
            vec = gx.get(c.addr.layer)
            c.gradxact_score = (
                float(vec[c.addr.row].item()) if vec is not None and c.addr.row < vec.numel() else 0.0
            )
        # Keep the better half per layer by grad x act; this is a ranking, not
        # a causal decision.  The ablator makes the causal decision.
        by_layer: dict[str, list[Candidate]] = defaultdict(list)
        for c in candidates:
            by_layer[layer_id(c.addr.layer)].append(c)
        kept: list[Candidate] = []
        for layer_cands in by_layer.values():
            layer_cands.sort(key=lambda x: x.gradxact_score, reverse=True)
            keep_n = max(1, (len(layer_cands) + 1) // 2)
            kept.extend(layer_cands[:keep_n])
        candidates = kept

    if cfg.balance_layers and len(candidates) > 4:
        cap = max(1, int(0.25 * len(candidates)))
        by_layer: dict[str, list[Candidate]] = defaultdict(list)
        for c in candidates:
            by_layer[layer_id(c.addr.layer)].append(c)
        balanced: list[Candidate] = []
        for layer_cands in by_layer.values():
            layer_cands.sort(key=lambda x: x.act_score, reverse=True)
            balanced.extend(layer_cands[:cap])
        candidates = balanced

    candidates.sort(key=lambda c: ((c.addr.layer), -c.act_score, c.addr.row))
    if len(candidates) > cfg.max_candidates:
        candidates = sorted(candidates, key=lambda c: c.act_score, reverse=True)[: cfg.max_candidates]

    counts = Counter(c.addr.layer for c in candidates)
    layer_counts = Counter(layer_id(c.addr.layer) for c in candidates)
    report = {
        "n_prefilter": n_prefilter,
        "n_candidates": len(candidates),
        "n_layers": len(layer_counts),
        "per_layer": dict(sorted(layer_counts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)),
        "per_module": dict(sorted(counts.items())),
        "gradxact_loss": gradxact_loss,
        "wrong_span": failure.wrong_span,
        "target_span": failure.target_span,
        "wrong_span_tokens": int(enc.span[1] - enc.span[0] + 1),
    }
    return candidates, report
