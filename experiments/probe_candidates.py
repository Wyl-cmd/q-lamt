from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import torch

from qlamt.runtime import ModelRuntime
from qlamt.types import TrainConfig
from qlamt.utils import set_seed


PROBES = [
    ("What is the capital of Australia? Answer with only the city name.", "Canberra"),
    ("What is the capital of Canada? Answer with only the city name.", "Ottawa"),
    ("What is the capital of Turkey? Answer with only the city name.", "Ankara"),
    ("What is the capital of Brazil? Answer with only the city name.", "Brasilia"),
    ("What is the capital of Switzerland? Answer with only the city name.", "Bern"),
    ("In which city is the Eiffel Tower located? Answer with only the city name.", "Paris"),
    ("What is the capital city of Valdoria? Answer with only the city name.", "Mervik"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.json")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    cfg = TrainConfig.load(args.config).resolved()
    set_seed(cfg.seed)
    rt = ModelRuntime.load(cfg)
    try:
        out = []
        for prompt, target in PROBES:
            answer = rt.generate(prompt, use_deltas=False, max_new_tokens=args.max_new_tokens)
            t_lp = rt.score_completion(prompt, target, use_deltas=False)[0]
            w_lp = rt.score_completion(prompt, answer, use_deltas=False)[0] if answer else float("-inf")
            item = {
                "prompt": prompt,
                "target": target,
                "answer": answer,
                "target_lp": t_lp,
                "answer_lp": w_lp,
                "target_wins": t_lp > w_lp,
            }
            out.append(item)
            print(json.dumps(item, ensure_ascii=False))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("\nJSON:")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        rt.close()


if __name__ == "__main__":
    main()
