from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .ablate import Ablator
from .locate import locate
from .runtime import ModelRuntime
from .train import build_background_batch, train_rows
from .types import Failure, NamedRow, TrainConfig
from .utils import (
    cuda_memory_stats,
    duration,
    new_call_id,
    reset_peak_memory,
    set_seed,
    write_json,
)
from .verify import evaluate_failure, verify
from .versions import VersionStore


def _canonical_rows(named: list[NamedRow]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    for n in named:
        out.setdefault(n.addr.layer, set()).add(int(n.addr.row))
    return {k: sorted(v) for k, v in out.items()}


def _runtime_mode(param_mode: str) -> str:
    return "splice" if param_mode in {"splice", "splice_fp16"} else "delta"


def _build_union_init(
    store: VersionStore,
    current: Any,
    new_rows: dict[str, list[int]],
    param_mode: str,
) -> tuple[dict[str, list[int]], dict[str, torch.Tensor], dict[str, list[int]]]:
    """Union of current sidecar rows and this failure's new rows.

    Current rows keep their sidecar value; new rows start from W_fp16
    (splice) or zero (delta).  Values are aligned to the sorted union rows.
    """
    side_rows = current.rows_by_layer if current is not None else {}
    side_maps = (
        {name: current.row_map(name) for name in side_rows} if current is not None else {}
    )
    names = sorted(set(side_rows) | set(new_rows))
    union: dict[str, list[int]] = {}
    values: dict[str, torch.Tensor] = {}
    trainable: dict[str, list[int]] = {}
    for name in names:
        rows = sorted(set(side_rows.get(name, [])) | set(new_rows.get(name, [])))
        if not rows:
            continue
        vals = []
        for r in rows:
            if name in side_maps and int(r) in side_maps[name]:
                vals.append(side_maps[name][int(r)].float().cpu())
            elif _runtime_mode(param_mode) == "splice":
                vals.append(
                    store.ledger.read_rows(name, [int(r)])[0].detach().float().cpu()
                )
            else:
                vals.append(
                    torch.zeros(store.ledger.in_features(name), dtype=torch.float32)
                )
        union[name] = rows
        values[name] = torch.stack(vals).float()
        trainable[name] = sorted({int(r) for r in new_rows.get(name, [])})
    return union, values, trainable


def _cumulative_guard(
    runtime: ModelRuntime, store: VersionStore
) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    max_ratio = 0.0
    for spec in runtime.deltas.values():
        rows = spec.rows.cpu().tolist()
        base = store.ledger.read_rows(spec.name, rows).float()
        param = spec.param.detach().float().cpu()
        edit = param - base if spec.mode == "splice" else param
        edit_norms = torch.linalg.vector_norm(edit, dim=1)
        base_norms = torch.linalg.vector_norm(base, dim=1)
        for r, e, b in zip(rows, edit_norms.tolist(), base_norms.tolist()):
            if b <= 0:
                continue
            ratio = float(e) / float(b)
            max_ratio = max(max_ratio, ratio)
            details.append({"layer": spec.name, "row": int(r), "ratio": ratio})
    details.sort(key=lambda x: x["ratio"], reverse=True)
    return max_ratio, details[:8]


def _regression_eval(
    runtime: ModelRuntime, store: VersionStore, cfg: TrainConfig
) -> tuple[list[dict[str, Any]], bool, float]:
    items = store.regression_items(limit=cfg.regression_size)
    rows: list[dict[str, Any]] = []
    for item in items:
        f = Failure.from_dict(item["failure"])
        ev = evaluate_failure(
            runtime,
            f,
            use_deltas=True,
            generate=False,
            wrong_spans=item.get("base_answers"),
        )
        prev = item.get("metrics", {}).get("hybrid_fix_ratio")
        floor = (
            max(0.0, float(prev) - 0.01)
            if prev is not None
            else float(cfg.verify_fix_ratio)
        )
        rows.append(
            {
                "version_id": item["version_id"],
                "failure_id": f.id,
                "fix_ratio": ev.fix_ratio,
                "previous_fix_ratio": prev,
                "passed": bool(ev.fix_ratio >= floor),
                "generation": (ev.variants[0].get("generation") if ev.variants else None),
            }
        )
    ratios = [r["fix_ratio"] for r in rows]
    mean = sum(ratios) / len(ratios) if ratios else 1.0
    return rows, all(r["passed"] for r in rows), mean


def run_fix_multi(
    failure: Failure,
    cfg: TrainConfig,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Multi-round: immutable base, current sidecar frozen, new rows trainable.

    No base write.  An accepted version only advances the ``current`` pointer;
    release is the only step that merges and quantizes.
    """
    cfg = cfg.resolved()
    cfg.validate()
    set_seed(cfg.seed)
    t_store = time.perf_counter()
    store = VersionStore(cfg.model_path, cfg.versions_root)
    store_init_s = duration(t_store)
    base_model = cfg.model_path
    cfg = TrainConfig.from_dict({**cfg.to_dict(), "model_path": str(store.base_dir), "mode": "sidecar"})
    call_id = new_call_id(failure.id)
    run_dir = Path(run_dir) if run_dir is not None else Path(cfg.out_dir) / call_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "call_id": call_id,
        "run_dir": str(run_dir.resolve()),
        "mode": "sidecar",
        "param_mode": cfg.param_mode,
        "base_model": base_model,
        "base_sha256": store.base_sha,
        "failure": failure.to_dict(),
        "config": cfg.to_dict(),
        "timings": {"store_init": store_init_s},
        "memory": {},
        "stages": {},
        "result": "running",
    }
    t_all = time.perf_counter()
    runtime: ModelRuntime | None = None
    try:
        reset_peak_memory()
        t0 = time.perf_counter()
        runtime = ModelRuntime.load(cfg)
        report["timings"]["load"] = duration(t0)
        report["memory"]["load"] = cuda_memory_stats()

        bg_batch, bg_texts = build_background_batch(runtime, cfg)
        # Absolute drift reference: clean base, before any sidecar.
        base_bg = (
            runtime.background_logprobs(bg_batch, use_deltas=False, no_grad=True)
            if bg_texts
            else None
        )

        current = store.load_current()
        if current is not None:
            if current.param_mode != cfg.param_mode:
                raise ValueError(
                    f"current sidecar mode {current.param_mode} != cfg.param_mode "
                    f"{cfg.param_mode}"
                )
            runtime.attach_rows(
                current.rows_by_layer,
                values_by_layer=current.values_by_layer,
                trainable_rows_by_layer={k: [] for k in current.rows_by_layer},
                mode=_runtime_mode(cfg.param_mode),
            )
            prev_bg = (
                runtime.background_logprobs(bg_batch, use_deltas=True, no_grad=True)
                if bg_texts
                else None
            )
        else:
            prev_bg = None
        report["stages"]["current_version"] = (
            current.manifest if current is not None else None
        )

        was_auto = not failure.wrong_span.strip()
        if was_auto:
            generated = runtime.generate(
                failure.prompt, use_deltas=True, max_new_tokens=cfg.max_new_tokens
            ).strip()
            if not generated:
                raise RuntimeError("auto_wrong produced an empty answer")
            failure = Failure.from_dict({**failure.to_dict(), "wrong_span": generated})

        t0 = time.perf_counter()
        prompts = failure.replay_prompts()
        if cfg.probe_variants:
            base_answers = [
                (
                    runtime.generate(
                        p, use_deltas=True, max_new_tokens=cfg.max_new_tokens
                    ).strip()
                    or failure.wrong_span
                )
                for p in prompts
            ]
        else:
            base_answers = [failure.wrong_span] * len(prompts)
        if was_auto and base_answers[0]:
            failure = Failure.from_dict(
                {**failure.to_dict(), "wrong_span": base_answers[0]}
            )
        report["effective_failure"] = failure.to_dict()
        report["stages"]["base_probe"] = {"prompts": prompts, "answers": base_answers}
        base_eval = evaluate_failure(
            runtime,
            failure,
            use_deltas=True,
            generate=False,
            wrong_spans=base_answers,
        )
        report["stages"]["base_failure"] = base_eval.to_dict()
        report["timings"]["base_probe"] = duration(t0)

        reset_peak_memory()
        t0 = time.perf_counter()
        candidates, locate_report = locate(runtime, failure, cfg)
        report["stages"]["locate"] = locate_report
        report["timings"]["locate"] = duration(t0)
        report["memory"]["locate"] = cuda_memory_stats()

        reset_peak_memory()
        t0 = time.perf_counter()
        ablator = Ablator(runtime, failure, cfg)
        named, ablate_report = ablator.confirm(candidates)
        report["stages"]["ablate"] = ablate_report
        report["timings"]["ablate"] = duration(t0)
        report["memory"]["ablate"] = cuda_memory_stats()
        if not named:
            report["result"] = "no_named_rows"
            report["error"] = "ablation confirmed no causal rows"
            write_json(run_dir / "report.json", report)
            return report

        new_rows = _canonical_rows(named)
        union_rows, values, trainable = _build_union_init(
            store, current, new_rows, cfg.param_mode
        )
        runtime.attach_rows(
            union_rows,
            values_by_layer=values,
            trainable_rows_by_layer=trainable,
            mode=_runtime_mode(cfg.param_mode),
        )
        n_rows = int(sum(len(v) for v in union_rows.values()))
        param_elems = int(
            sum(
                len(union_rows[name]) * store.ledger.in_features(name)
                for name in union_rows
            )
        )
        report["stages"]["materialize"] = {
            "base_ledger": str(store.ledger.model_dir.resolve()),
            "n_new_rows": int(sum(len(v) for v in new_rows.values())),
            "n_union_rows": n_rows,
            "n_union_layers": len(union_rows),
            "delta_param_mb": round(param_elems * 4 / 2**20, 2),
            "adam_estimate_mb": round(param_elems * 16 / 2**20, 2),
        }

        reset_peak_memory()
        t0 = time.perf_counter()
        train_result = train_rows(
            runtime,
            failure,
            named,
            cfg,
            store.ledger,
            prev_bg=prev_bg,
            prepared=True,
        )
        report["stages"]["train"] = train_result.to_dict()
        report["timings"]["train"] = duration(t0)
        report["memory"]["train"] = cuda_memory_stats()

        reset_peak_memory()
        t0 = time.perf_counter()
        verification = verify(
            runtime,
            failure,
            cfg,
            base_failure_eval=base_eval,
            base_bg=base_bg,
            bg_texts=bg_texts,
            bg_batch=bg_batch if bg_texts else None,
            generate=True,
            wrong_spans=base_answers,
        )
        report["stages"]["verify"] = verification
        report["timings"]["verify"] = duration(t0)
        report["memory"]["verify"] = cuda_memory_stats()

        reg_rows, reg_ok, reg_mean = _regression_eval(runtime, store, cfg)
        cum_max, cum_top = _cumulative_guard(runtime, store)
        cum_ok = cum_max <= float(cfg.cum_guard)
        report["stages"]["regression"] = {
            "passed": reg_ok,
            "mean_fix_ratio": reg_mean,
            "items": reg_rows,
        }
        report["stages"]["cumulative"] = {
            "passed": cum_ok,
            "max_ratio": cum_max,
            "limit": cfg.cum_guard,
            "top": cum_top,
        }

        accepted = bool(verification["passed"] and reg_ok and cum_ok)
        metrics = {
            "hybrid_fix_ratio": verification["failure"]["fix_ratio"],
            "post_quant_fix_ratio": None,
            "bg_kl_mean": verification["background"].get("kl_mean"),
            "target_lp_gain": verification.get("target_lp_gain_vs_base"),
            "regression_fix_ratio": reg_mean,
            "cumulative_row_ratio_max": cum_max,
            "n_rows": n_rows,
            "n_new_rows": int(sum(len(v) for v in new_rows.values())),
        }
        if not verification["passed"]:
            metrics["reject_reason"] = "; ".join(verification["reasons"])
        elif not reg_ok:
            metrics["reject_reason"] = "regression"
        elif not cum_ok:
            metrics["reject_reason"] = f"cumulative_guard {cum_max:.3f} > {cfg.cum_guard}"

        version = store.commit(
            union_rows,
            runtime.delta_values(),
            cfg.param_mode,
            failure,
            base_answers,
            metrics,
            cfg,
            parent=current.version_id if current is not None else None,
            status="accepted" if accepted else "rejected",
            advance=accepted,
            expected_current=current.version_id if current is not None else None,
            check_current=accepted,
        )
        report["version"] = version.manifest
        report["result"] = "version_committed" if accepted else "version_rejected"
        if not accepted:
            report["error"] = metrics.get("reject_reason")
        report["timings"]["total"] = duration(t_all)
        write_json(run_dir / "report.json", report)
        return report

    except Exception as exc:
        report["result"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["timings"]["total"] = duration(t_all)
        write_json(run_dir / "report.json", report)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
