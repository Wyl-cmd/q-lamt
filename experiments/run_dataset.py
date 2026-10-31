from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import time
from collections import defaultdict
from typing import Any

from qlamt.pipeline import run_fix
from qlamt.types import Failure, PROJECT_ROOT, TrainConfig
from qlamt.utils import write_json


def _load_failures(path: Path) -> list[Failure]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(Failure.from_dict(json.loads(line)))
    return rows


def _pick_failures(
    rows: list[Failure], limit: int, ids: list[str] | None
) -> list[Failure]:
    if ids:
        wanted = {x.strip() for x in ids if x.strip()}
        rows = [r for r in rows if r.id in wanted or r.category in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def _row_summary(failure: Failure, report: dict[str, Any], elapsed: float) -> dict[str, Any]:
    stages = report.get("stages", {})
    base = stages.get("base_failure", {})
    ver = stages.get("verify", {})
    post = stages.get("post_requant", {})
    ablate = stages.get("ablate", {})
    train = stages.get("train", {})
    materialize = stages.get("materialize", {})
    mem_train = report.get("memory", {}).get("train") or {}
    ver_variants = ver.get("failure", {}).get("variants", [])
    main = ver_variants[0] if ver_variants else {}
    post_variants = post.get("failure", {}).get("variants", [])
    post_main = post_variants[0] if post_variants else {}
    return {
        "id": failure.id,
        "category": failure.category,
        "result": report.get("result"),
        "error": report.get("error"),
        "n_named": ablate.get("n_named"),
        "n_fallback": ablate.get("n_fallback"),
        "n_rows_written": materialize.get("n_rows"),
        "delta_param_mb": materialize.get("delta_param_mb"),
        "train_steps": train.get("steps_run"),
        "stopped_reason": train.get("stopped_reason"),
        "base_fix_ratio": base.get("fix_ratio"),
        "trained_fix_ratio": ver.get("failure", {}).get("fix_ratio"),
        "target_lp_gain": ver.get("target_lp_gain_vs_base"),
        "bg_kl_mean": ver.get("background", {}).get("kl_mean"),
        "bg_kl_max": ver.get("background", {}).get("kl_max"),
        "main_generation": main.get("generation"),
        "main_gen_exact": main.get("generation_exact"),
        "main_gen_contains": main.get("generation_contains"),
        "post_requant_pass": post.get("passed"),
        "post_requant_fix_ratio": post.get("failure", {}).get("fix_ratio"),
        "post_main_generation": post_main.get("generation"),
        "post_main_gen_exact": post_main.get("generation_exact"),
        "post_main_gen_contains": post_main.get("generation_contains"),
        "peak_train_gb": mem_train.get("peak_allocated_gb"),
        "elapsed_s": round(elapsed, 2),
        "run_dir": report.get("run_dir"),
    }


def _mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the LAMT pipeline over a small dataset")
    ap.add_argument("--data", default="data/e2e_small.jsonl")
    ap.add_argument("--config", default="configs/dataset.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated failure ids or categories")
    ap.add_argument("--out", default="", help="dataset run root")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-post-requant", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig.load(args.config)
    if args.set:
        cfg.apply_overrides(args.set)
    if args.no_post_requant:
        cfg.verify_post_requant = False
    cfg = cfg.resolved()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    failures = _pick_failures(
        _load_failures(data_path),
        args.limit,
        [x for x in args.ids.split(",")] if args.ids else None,
    )
    if not failures:
        print("no failures selected")
        return 1
    if args.dry_run:
        for f in failures:
            print(f.id, f.category, f.prompt)
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) if args.out else (PROJECT_ROOT / "artifacts" / "datasets" / stamp)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, failure in enumerate(failures):
        t0 = time.perf_counter()
        print(f"[{i + 1}/{len(failures)}] {failure.id} ({failure.category}) ...", flush=True)
        try:
            report = run_fix(failure, cfg, run_dir=root / f"{i:02d}-{failure.id}")
            row = _row_summary(failure, report, time.perf_counter() - t0)
        except Exception as exc:
            row = {
                "id": failure.id,
                "category": failure.category,
                "result": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
        rows.append(row)
        print(
            f"    result={row.get('result')} named={row.get('n_named')} "
            f"base={row.get('base_fix_ratio')} trained={row.get('trained_fix_ratio')} "
            f"post={row.get('post_requant_fix_ratio')} kl={row.get('bg_kl_mean')}",
            flush=True,
        )

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("category") or "")].append(row)

    def agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        peaks = [float(r["peak_train_gb"]) for r in subset if r.get("peak_train_gb") is not None]
        return {
            "n": len(subset),
            "written": sum(
                1
                for r in subset
                if r.get("result") in {"written", "written_but_post_requant_failed"}
            ),
            "post_pass": sum(1 for r in subset if r.get("post_requant_pass") is True),
            "errors": sum(1 for r in subset if r.get("result") == "error"),
            "base_fix_ratio_mean": _mean([r.get("base_fix_ratio") for r in subset]),
            "trained_fix_ratio_mean": _mean([r.get("trained_fix_ratio") for r in subset]),
            "post_fix_ratio_mean": _mean([r.get("post_requant_fix_ratio") for r in subset]),
            "target_lp_gain_mean": _mean([r.get("target_lp_gain") for r in subset]),
            "bg_kl_mean": _mean([r.get("bg_kl_mean") for r in subset]),
            "n_named_mean": _mean([r.get("n_named") for r in subset]),
            "elapsed_s_mean": _mean([r.get("elapsed_s") for r in subset]),
            "peak_train_gb_max": round(max(peaks), 4) if peaks else None,
            "main_gen_exact": sum(1 for r in subset if r.get("main_gen_exact") is True),
            "main_gen_contains": sum(1 for r in subset if r.get("main_gen_contains") is True),
            "post_main_gen_exact": sum(1 for r in subset if r.get("post_main_gen_exact") is True),
            "post_main_gen_contains": sum(1 for r in subset if r.get("post_main_gen_contains") is True),
        }

    summary = {
        "data": str(data_path),
        "config": cfg.to_dict(),
        "root": str(root),
        "overall": agg(rows),
        "by_category": {k: agg(v) for k, v in sorted(by_cat.items())},
        "rows": rows,
    }
    write_json(root / "summary.json", summary)
    print("\nSUMMARY")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"summary: {root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
