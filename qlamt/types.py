from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, order=True)
class RowAddr:
    """A named row: one output channel of one Linear module.

    ``layer`` is the full module name in the HF model, e.g.
    ``model.layers.17.mlp.down_proj``.  ``row`` is the output-channel index.
    NF4 does not change ``out x in`` shape or row numbering, so this address is
    valid in both the quantized runtime and the fp16 ledger.
    """

    layer: str
    row: int

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "row": self.row}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RowAddr":
        return cls(layer=str(d["layer"]), row=int(d["row"]))


@dataclass
class Failure:
    id: str
    prompt: str
    wrong_span: str
    target_span: str
    paraphrases: list[str] = field(default_factory=list)
    category: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Failure":
        return cls(
            id=str(data.get("id", "failure")),
            prompt=str(data["prompt"]),
            wrong_span=str(data.get("wrong_span", "")),
            target_span=str(data["target_span"]),
            paraphrases=[str(x) for x in data.get("paraphrases", [])],
            category=str(data.get("category", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Failure":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replay_prompts(self) -> list[str]:
        out = [self.prompt]
        for p in self.paraphrases:
            if p and p not in out:
                out.append(p)
        return out


@dataclass
class Candidate:
    addr: RowAddr
    act_score: float = 0.0
    gradxact_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.addr.layer,
            "row": self.addr.row,
            "act_score": self.act_score,
            "gradxact_score": self.gradxact_score,
        }


@dataclass
class NamedRow:
    addr: RowAddr
    delta_m: float
    delta_target: float
    delta_wrong: float
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.addr.layer,
            "row": self.addr.row,
            "delta_m": self.delta_m,
            "delta_target": self.delta_target,
            "delta_wrong": self.delta_wrong,
            "fallback": self.fallback,
        }


@dataclass
class Metric:
    """Failure metric used by the ablator.

    ``m = mean_logp(wrong) + mean_logp(target)``.  Zeroing a row that supports
    the wrong answer lowers the first term; zeroing a row that suppresses the
    target raises the second.  Both are causal, so the ablator ranks by |delta|.
    """

    wrong_lp: float
    target_lp: float

    @property
    def m(self) -> float:
        return self.wrong_lp + self.target_lp

    @property
    def wrong_nll(self) -> float:
        return -self.wrong_lp

    @property
    def target_nll(self) -> float:
        return -self.target_lp

    def delta(self, base: "Metric") -> "Metric":
        return Metric(
            wrong_lp=self.wrong_lp - base.wrong_lp,
            target_lp=self.target_lp - base.target_lp,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "m": self.m,
            "wrong_lp": self.wrong_lp,
            "target_lp": self.target_lp,
        }


@dataclass
class TrainConfig:
    # model / runtime
    model_path: str = "models/Qwen2.5-0.5B-Instruct"
    quant: str = "nf4"  # "nf4" | "none"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    chat: bool = True
    seed: int = 0
    grad_checkpoint: bool = False
    auto_wrong: bool = True
    probe_variants: bool = True
    append_eos: bool = False

    # multi-round persistence
    mode: str = "merge"  # "merge" (single-call) | "sidecar" (versioned)
    param_mode: str = "delta"  # "delta" | "splice_fp16"
    versions_root: str = "artifacts/versions"
    release_root: str = "artifacts/releases"
    lambda_prev: float = 0.3
    cum_guard: float = 0.3
    regression_size: int = 64

    # candidate layer pool.  q/k/v are intentionally excluded by default.
    layer_pool: str = (
        r"model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.o_proj)"
    )
    topk_per_layer: int = 8
    use_gradxact: bool = True
    max_candidates: int = 512
    balance_layers: bool = True

    # ablation
    group_size: int = 8
    calibrate_epsilon: bool = True
    null_groups: int = 6
    epsilon_ablate: float | None = None
    epsilon_row: float | None = None
    min_epsilon: float = 1e-4
    min_named: int = 8
    max_named: int = 512

    # materialize / train
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 150
    batch_size: int = 4
    eval_every: int = 25
    max_seq_len: int = 256
    warmup_steps: int = 5
    grad_clip: float = 1.0
    kl_lambda: float = 0.3
    kl_every: int = 4
    background_path: str = "configs/background.txt"
    background_size: int = 16
    background_tokens: int = 48
    guard_ratio: float = 0.5
    max_new_tokens: int = 16

    # verification
    verify_fix_ratio: float = 0.8
    verify_bg_kl_max: float = 0.05
    verify_bg_kl_max_single: float = 0.2

    # persistence
    ledger_dir: str = "artifacts/ledger"
    out_dir: str = "artifacts/runs"
    copy_ledger: bool = True
    write_back: bool = True
    verify_post_requant: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path | None) -> "TrainConfig":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def apply_overrides(self, overrides: Iterable[str]) -> None:
        """Apply ``key=value`` overrides, JSON-decoding the value when possible."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got {item!r}")
            key, raw = item.split("=", 1)
            if key not in {f.name for f in fields(self)}:
                raise ValueError(f"unknown config key: {key}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            setattr(self, key, value)

    def resolved(self, root: Path = PROJECT_ROOT) -> "TrainConfig":
        """Return a copy with all filesystem fields resolved against ``root``."""
        data = asdict(self)
        for key in (
            "model_path",
            "background_path",
            "ledger_dir",
            "out_dir",
            "versions_root",
            "release_root",
        ):
            p = Path(str(data[key]))
            data[key] = str(p if p.is_absolute() else root / p)
        return TrainConfig.from_dict(data)

    def validate(self) -> None:
        if self.mode not in {"merge", "sidecar"}:
            raise ValueError(f"mode must be merge|sidecar, got {self.mode!r}")
        if self.param_mode not in {"delta", "splice_fp16"}:
            raise ValueError(f"param_mode must be delta|splice_fp16, got {self.param_mode!r}")
        if self.quant not in {"nf4", "none"}:
            raise ValueError(f"quant must be nf4|none, got {self.quant!r}")
        if self.quant == "nf4" and self.device != "cuda":
            raise ValueError("quant='nf4' requires device='cuda'")
        for name in (
            "steps",
            "batch_size",
            "eval_every",
            "topk_per_layer",
            "group_size",
            "warmup_steps",
            "max_seq_len",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("lr", "cum_guard", "guard_ratio", "grad_clip"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("kl_lambda", "lambda_prev", "weight_decay"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("verify_fix_ratio", "verify_bg_kl_max", "verify_bg_kl_max_single"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(self.model_path)
        if not str(self.versions_root).strip():
            raise ValueError("versions_root must not be empty")
        if not str(self.release_root).strip():
            raise ValueError("release_root must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    steps_run: int
    stopped_reason: str
    history: list[dict[str, float]]
    delta_norms: dict[str, float]
    peak_cuda_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
