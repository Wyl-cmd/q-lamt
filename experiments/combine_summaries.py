from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import Counter
from typing import Any

from qlamt.types import PROJECT_ROOT
from qlamt.utils import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default="artifacts/datasets/eos12_combined.json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    seen = set()
    for s in args.summaries:
        p = Path(s)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        data = json.loads(p.read_text(encoding="utf-8"))
        for row in data["rows"]:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            report_path = Path(row.get("run_dir") or "") / "report.json"
            train = {}
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                train = report.get("stages", {}).get("train", {})
                hist = train.get("history") or []
                row["max_delta_ratio"] = hist[-1].get("max_delta_ratio") if hist else None
            else:
                row["max_delta_ratio"] = None
            row["stop"] = train.get("stopped_reason") or row.get("stopped_reason")
            rows.append(row)

    def mean(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        vals = [
            float(r[key])
            for r in (subset or rows)
            if r.get(key) is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

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
            "base_fix_ratio_mean": mean("base_fix_ratio", subset),
            "trained_fix_ratio_mean": mean("trained_fix_ratio", subset),
            "post_fix_ratio_mean": mean("post_requant_fix_ratio", subset),
            "target_lp_gain_mean": mean("target_lp_gain", subset),
            "bg_kl_mean": mean("bg_kl_mean", subset),
            "n_named_mean": mean("n_named", subset),
            "n_fallback_mean": mean("n_fallback", subset),
            "elapsed_s_mean": mean("elapsed_s", subset),
            "peak_train_gb_max": round(max(peaks), 4) if peaks else None,
            "gen_exact": sum(1 for r in subset if r.get("main_gen_exact") is True),
            "gen_contains": sum(1 for r in subset if r.get("main_gen_contains") is True),
            "post_gen_exact": sum(
                1 for r in subset if r.get("post_main_gen_exact") is True
            ),
            "post_gen_contains": sum(
                1 for r in subset if r.get("post_main_gen_contains") is True
            ),
            "max_delta_ratio_max": max(
                (float(r["max_delta_ratio"]) for r in subset if r.get("max_delta_ratio") is not None),
                default=None,
            ),
        }

    cats: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cats.setdefault(str(r.get("category") or ""), []).append(r)

    out = {
        "summaries": [str(Path(s)) for s in args.summaries],
        "overall": agg(rows),
        "by_category": {k: agg(v) for k, v in sorted(cats.items())},
        "stop_reasons": dict(Counter(str(r.get("stop")) for r in rows)),
        "rows": sorted(rows, key=lambda r: str(r["id"])),
    }
    write_json(args.out, out)
    print(json.dumps(out["overall"], ensure_ascii=False, indent=2))
    print("by_category:")
    print(json.dumps(out["by_category"], ensure_ascii=False, indent=2))
    print("stop_reasons:", out["stop_reasons"])
    print(f"combined: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
