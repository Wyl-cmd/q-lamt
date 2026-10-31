from __future__ import annotations

import gc
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .ablate import Ablator
from .ledger import LedgerManager
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
from .verify import background_metrics, evaluate_failure, verify


def _canonical_rows(named: list[NamedRow]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    for n in named:
        out.setdefault(n.addr.layer, set()).add(int(n.addr.row))
    return {k: sorted(v) for k, v in out.items()}


def _resolve_wrong(runtime: ModelRuntime, failure: Failure, cfg: TrainConfig) -> Failure:
    if failure.wrong_span.strip():
        return failure
    if not cfg.auto_wrong:
        raise ValueError("failure.wrong_span is empty and auto_wrong is disabled")
    generated = runtime.generate(
        failure.prompt, use_deltas=False, max_new_tokens=cfg.max_new_tokens
    ).strip()
    if not generated:
        raise RuntimeError("auto_wrong produced an empty answer")
    return replace(failure, wrong_span=generated)


def _merge_deltas(runtime: ModelRuntime, ledger: LedgerManager) -> tuple[dict[str, list[int]], dict[str, torch.Tensor], dict[str, Any]]:
    rows = {name: spec.rows.cpu().tolist() for name, spec in runtime.deltas.items()}
    deltas = {name: spec.param.detach().float().cpu() for name, spec in runtime.deltas.items()}
    originals = ledger.read_rows_by_layer(rows)
    merged: dict[str, torch.Tensor] = {}
    modes: dict[str, str] = {}
    for name, rows_i in rows.items():
        spec = runtime.deltas[name]
        orig = originals[name]
        d = deltas[name]
        if orig.shape != d.shape:
            raise ValueError(f"{name}: ledger rows shape {tuple(orig.shape)} != delta {tuple(d.shape)}")
        modes[name] = spec.mode
        if spec.mode == "splice":
            merged[name] = d.to(orig.dtype)
        else:
            merged[name] = (orig.float() + d).to(orig.dtype)
    return rows, merged, {
        "modes": modes,
        "original_dtype": {k: str(v.dtype) for k, v in originals.items()},
    }


def run_fix(
    failure: Failure,
    cfg: TrainConfig,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full pipeline on one failure and return the report.

    Stages: locate -> ablate -> materialize (row norms + undo snapshot) ->
    train -> verify -> write back -> requantize and verify again.
    """
    cfg = cfg.resolved()
    cfg.validate()
    set_seed(cfg.seed)
    call_id = new_call_id(failure.id)
    run_dir = Path(run_dir) if run_dir is not None else Path(cfg.out_dir) / call_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "call_id": call_id,
        "run_dir": str(run_dir.resolve()),
        "failure": failure.to_dict(),
        "config": cfg.to_dict(),
        "timings": {},
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
        report["runtime"] = runtime.info()

        was_auto = not failure.wrong_span.strip()
        failure = _resolve_wrong(runtime, failure, cfg)
        if not failure.wrong_span:
            raise RuntimeError("wrong_span is still empty")

        # Base measurements are taken before any delta exists.  For each
        # paraphrase we probe the base answer, so the fix check compares the
        # target against what the model actually says for that phrasing rather
        # than against the main prompt's wrong answer.
        t0 = time.perf_counter()
        prompts = failure.replay_prompts()
        if cfg.probe_variants:
            base_answers = [
                (
                    runtime.generate(
                        p, use_deltas=False, max_new_tokens=cfg.max_new_tokens
                    ).strip()
                    or failure.wrong_span
                )
                for p in prompts
            ]
        else:
            base_answers = [failure.wrong_span] * len(prompts)
        if was_auto and base_answers[0]:
            failure = replace(failure, wrong_span=base_answers[0])
        report["effective_failure"] = failure.to_dict()
        report["stages"]["base_probe"] = {
            "prompts": prompts,
            "answers": base_answers,
        }
        base_eval = evaluate_failure(
            runtime, failure, use_deltas=False, generate=False, wrong_spans=base_answers
        )
        report["stages"]["base_failure"] = base_eval.to_dict()
        report["timings"]["base_failure"] = duration(t0)

        bg_batch, bg_texts = build_background_batch(runtime, cfg)
        base_bg = (
            runtime.background_logprobs(bg_batch, use_deltas=False, no_grad=True)
            if bg_texts
            else None
        )

        # ---- step 1: locate
        reset_peak_memory()
        t0 = time.perf_counter()
        candidates, locate_report = locate(runtime, failure, cfg)
        report["stages"]["locate"] = locate_report
        report["timings"]["locate"] = duration(t0)
        report["memory"]["locate"] = cuda_memory_stats()

        # ---- step 2: ablate
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

        # ---- step 3: ledger copy, row norms, undo snapshot
        reset_peak_memory()
        t0 = time.perf_counter()
        ledger = LedgerManager.prepare(cfg.model_path, run_dir / "ledger")
        rows = _canonical_rows(named)
        row_norms = ledger.row_norms(rows)
        undo_dir = ledger.snapshot_rows(rows, run_dir / "undo")
        n_rows = int(sum(len(v) for v in rows.values()))
        param_elems = int(
            sum(len(rows[name]) * ledger.in_features(name) for name in rows)
        )
        report["stages"]["materialize"] = {
            "ledger": str(ledger.model_dir.resolve()),
            "undo": str(undo_dir.resolve()),
            "n_rows": n_rows,
            "n_layers": len(rows),
            "delta_param_elems": param_elems,
            "delta_param_mb": round(param_elems * 4 / 2**20, 2),
            "adam_estimate_mb": round(param_elems * 16 / 2**20, 2),
            "row_norms_preview": dict(list(row_norms.items())[:8]),
        }
        report["timings"]["materialize"] = duration(t0)
        report["memory"]["materialize"] = cuda_memory_stats()

        # ---- step 4: local training on row deltas
        reset_peak_memory()
        t0 = time.perf_counter()
        train_result = train_rows(runtime, failure, named, cfg, ledger)
        report["stages"]["train"] = train_result.to_dict()
        report["timings"]["train"] = duration(t0)
        report["memory"]["train"] = cuda_memory_stats()

        # ---- verification before write-back
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

        if not verification["passed"]:
            report["result"] = "not_written_verification_failed"
            report["error"] = "; ".join(verification["reasons"])
            write_json(run_dir / "report.json", report)
            return report
        if not cfg.write_back:
            report["result"] = "not_written_disabled"
            write_json(run_dir / "report.json", report)
            return report

        # ---- step 5: merge into the fp16 ledger
        t0 = time.perf_counter()
        merged_rows, merged_values, merge_meta = _merge_deltas(runtime, ledger)
        write_report = ledger.write_rows(
            merged_values, merged_rows, undo_dir=None, verify=True
        )
        write_report["undo"] = str(undo_dir.resolve())
        write_report["merge"] = merge_meta
        report["stages"]["write_back"] = write_report
        if not write_report.get("verified"):
            ledger.rollback(undo_dir)
            raise RuntimeError("write-back read-back verification failed; rolled back")
        report["timings"]["write_back"] = duration(t0)
        report["result"] = "written"
        write_json(run_dir / "report.json", report)

        # ---- closed loop: reload the ledger as NF4 and verify again
        if cfg.verify_post_requant:
            t0 = time.perf_counter()
            post_runtime: ModelRuntime | None = None
            try:
                runtime.close()
                runtime = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                post_cfg = replace(cfg, model_path=str(ledger.model_dir))
                post_runtime = ModelRuntime.load(post_cfg)
                post_eval = evaluate_failure(
                    post_runtime,
                    failure,
                    use_deltas=False,
                    generate=True,
                    wrong_spans=base_answers,
                )
                post_bg = (
                    post_runtime.background_logprobs(bg_batch, use_deltas=False, no_grad=True)
                    if bg_texts
                    else None
                )
                post_bg_metrics = background_metrics(base_bg, post_bg, bg_texts)
                post_pass = bool(
                    post_eval.fix_ratio >= float(cfg.verify_fix_ratio)
                    and post_bg_metrics["kl_mean"] <= float(cfg.verify_bg_kl_max)
                    and post_bg_metrics["kl_max"] <= float(cfg.verify_bg_kl_max_single)
                )
                report["stages"]["post_requant"] = {
                    "passed": post_pass,
                    "failure": post_eval.to_dict(),
                    "background": post_bg_metrics,
                }
                if not post_pass:
                    report["result"] = "written_but_post_requant_failed"
            except Exception as exc:
                report["stages"]["post_requant"] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["result"] = "written_but_post_requant_failed"
            finally:
                if post_runtime is not None:
                    post_runtime.close()
                report["timings"]["post_requant"] = duration(t0)

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
