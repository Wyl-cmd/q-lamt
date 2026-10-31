from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .runtime import ModelRuntime
from .types import Candidate, Failure, Metric, NamedRow, RowAddr, TrainConfig


def _key(rows: Iterable[RowAddr]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((r.layer, int(r.row)) for r in rows))


def _group_rows(rows: list[RowAddr], by_layer: bool, group_size: int) -> list[list[RowAddr]]:
    if not by_layer:
        return [rows[i : i + group_size] for i in range(0, len(rows), group_size)]
    buckets: dict[str, list[RowAddr]] = defaultdict(list)
    for r in rows:
        buckets[r.layer].append(r)
    groups: list[list[RowAddr]] = []
    for layer in sorted(buckets):
        bucket = buckets[layer]
        groups.extend(bucket[i : i + group_size] for i in range(0, len(bucket), group_size))
    return groups


class Ablator:
    """Step 2: group ablation + bisection, with a calibrated null threshold.

    Ablating a row means zeroing its output channel.  The ablation is applied
    as a forward hook, so it is exact and does not mutate the NF4 weights.
    """

    def __init__(self, runtime: ModelRuntime, failure: Failure, cfg: TrainConfig):
        if not failure.wrong_span:
            raise ValueError("failure.wrong_span is empty")
        self.runtime = runtime
        self.failure = failure
        self.cfg = cfg
        self._metric_cache: dict[tuple[tuple[str, int], ...], Metric] = {}
        self._n_forward_pairs = 0
        self.baseline = self._measure_raw()
        self.epsilon_ablate: float = 0.0
        self.epsilon_row: float = 0.0
        self.null_deltas: list[float] = []

    # ------------------------------------------------------------- measures
    def _measure_raw(self) -> Metric:
        # Apply the current sidecar (if any); single-call has no patches.
        wrong = self.runtime.score_completion(
            self.failure.prompt, self.failure.wrong_span, use_deltas=True
        )[0]
        target = self.runtime.score_completion(
            self.failure.prompt, self.failure.target_span, use_deltas=True
        )[0]
        self._n_forward_pairs += 1
        return Metric(wrong_lp=wrong, target_lp=target)

    def measure(self, rows: list[RowAddr]) -> Metric:
        """Ablation delta (ablated - baseline) for a set of rows."""
        k = _key(rows)
        if k in self._metric_cache:
            return self._metric_cache[k]
        by_layer: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_layer[r.layer].append(r.row)
        with self.runtime.ablate(by_layer):
            m = self._measure_raw()
        delta = m.delta(self.baseline)
        self._metric_cache[k] = delta
        return delta

    # ---------------------------------------------------------- calibration
    def calibrate(self, candidates: list[Candidate]) -> None:
        if self.cfg.epsilon_ablate is not None:
            self.epsilon_ablate = float(self.cfg.epsilon_ablate)
        elif self.cfg.calibrate_epsilon and candidates:
            rng = random.Random(self.cfg.seed)
            layers = sorted({c.addr.layer for c in candidates})
            n_group = max(1, int(self.cfg.null_groups))
            for _ in range(n_group):
                layer = rng.choice(layers)
                n_rows = self.runtime.num_rows(layer)
                g = min(int(self.cfg.group_size), n_rows)
                rows = rng.sample(range(n_rows), g)
                delta = self.measure([RowAddr(layer, r) for r in rows])
                self.null_deltas.append(float(abs(delta.m)))
            if self.null_deltas:
                null_max = max(self.null_deltas)
                null_sorted = sorted(self.null_deltas)
                q = null_sorted[min(len(null_sorted) - 1, int(0.95 * len(null_sorted)))]
                self.epsilon_ablate = max(
                    float(self.cfg.min_epsilon), 1.5 * max(q, 0.5 * null_max)
                )
            else:
                self.epsilon_ablate = 0.02
        else:
            self.epsilon_ablate = 0.02

        if self.cfg.epsilon_row is not None:
            self.epsilon_row = float(self.cfg.epsilon_row)
        else:
            self.epsilon_row = 0.5 * self.epsilon_ablate

    # ------------------------------------------------------------- confirm
    def confirm(self, candidates: list[Candidate]) -> tuple[list[NamedRow], dict]:
        self.calibrate(candidates)
        groups = _group_rows(
            [c.addr for c in candidates], by_layer=True, group_size=int(self.cfg.group_size)
        )
        eps_bisect = max(self.epsilon_row, 0.5 * self.epsilon_ablate)
        named: dict[tuple[str, int], NamedRow] = {}
        n_groups_hit = 0

        def add_row(addr: RowAddr, delta: Metric) -> None:
            key = (addr.layer, addr.row)
            if abs(delta.m) < self.epsilon_row:
                return
            named[key] = NamedRow(
                addr=addr,
                delta_m=float(delta.m),
                delta_target=float(delta.target_lp),
                delta_wrong=float(delta.wrong_lp),
            )

        def bisect(rows: list[RowAddr], depth: int = 0) -> None:
            delta = self.measure(rows)
            if len(rows) == 1:
                add_row(rows[0], delta)
                return
            if abs(delta.m) < eps_bisect:
                return
            mid = len(rows) // 2
            for half in (rows[:mid], rows[mid:]):
                if not half:
                    continue
                d = self.measure(half)
                if abs(d.m) >= eps_bisect:
                    bisect(half, depth + 1)

        for g in groups:
            d = self.measure(g)
            if abs(d.m) < self.epsilon_ablate:
                continue
            n_groups_hit += 1
            bisect(g)

        # The threshold may be too strict for a tiny model / tiny effect.  Do
        # not silently pretend these are causal: measure a bounded number of
        # top-activation rows individually and mark them as fallback.
        if len(named) < int(self.cfg.min_named):
            ranked = sorted(candidates, key=lambda c: c.act_score, reverse=True)
            probe = ranked[: max(int(self.cfg.min_named) * 3, int(self.cfg.min_named))]
            fallback_pool: list[tuple[float, RowAddr, Metric]] = []
            for c in probe:
                if (c.addr.layer, c.addr.row) in named:
                    continue
                d = self.measure([c.addr])
                fallback_pool.append((abs(d.m), c.addr, d))
            fallback_pool.sort(key=lambda x: x[0], reverse=True)
            for _, addr, d in fallback_pool:
                if len(named) >= int(self.cfg.min_named):
                    break
                key = (addr.layer, addr.row)
                if key in named:
                    continue
                named[key] = NamedRow(
                    addr=addr,
                    delta_m=float(d.m),
                    delta_target=float(d.target_lp),
                    delta_wrong=float(d.wrong_lp),
                    fallback=True,
                )

        out = sorted(named.values(), key=lambda n: abs(n.delta_m), reverse=True)
        if len(out) > int(self.cfg.max_named):
            out = out[: int(self.cfg.max_named)]
        report = {
            "baseline": self.baseline.to_dict(),
            "epsilon_ablate": self.epsilon_ablate,
            "epsilon_row": self.epsilon_row,
            "null_deltas": self.null_deltas,
            "n_groups": len(groups),
            "n_groups_hit": n_groups_hit,
            "n_named": len(out),
            "n_fallback": sum(1 for n in out if n.fallback),
            "metric_evals": len(self._metric_cache),
            "forward_pairs": self._n_forward_pairs,
            "named": [n.to_dict() for n in out],
        }
        return out, report
