from __future__ import annotations

import contextlib
import gc
import re
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .types import RowAddr, TrainConfig
from .utils import (
    EncodedInput,
    build_prefix_ids,
    cuda_memory_stats,
    encode_prompt_completion,
    mean_logprob_from_logits,
    torch_dtype,
)


# ---------------------------------------------------------------------------
# Row-delta parameterization
# ---------------------------------------------------------------------------
# The deployed artifact is the NF4 model, so the runtime is also the training
# environment.  Instead of dequantizing a whole layer and overwriting rows in a
# bf16 copy (spec section 5.2), we attach a sparse row correction:
#
#     delta:   y = QLinear(x) + scatter(ΔR @ x, rows)
#     splice:  y = QLinear(x) + scatter((P - deq(W_q)[rows]) @ x, rows)
#
# At ΔR=0 the delta output is bit-identical to the quantized base; splice with
# P=W_fp16 starts from the fp16 row.  Gradients only reach the row parameter,
# and memory is O(named_rows * in_features).  Rows from a previous sidecar can
# be attached frozen while this round's new rows are trainable.
@dataclass
class DeltaSpec:
    name: str
    rows: torch.Tensor  # long, sorted unique, on device
    param: nn.Parameter  # (n_rows, in_features), float32
    in_features: int
    # "delta": param = ΔR, effective row = deq(W_q) + ΔR
    # "splice": param = P (FP16 row), effective row = P; base_rows = deq(W_q)[rows]
    mode: str = "delta"
    base_rows: torch.Tensor | None = None  # deq(W_q)[rows], float32
    trainable: torch.Tensor | None = None  # bool (n_rows,); None = all trainable
    init_values: torch.Tensor | None = None  # value at attach; per-round edit origin

    @property
    def n_rows(self) -> int:
        return int(self.rows.numel())

    def trainable_mask(self) -> torch.Tensor:
        if self.trainable is None:
            return torch.ones(self.n_rows, dtype=torch.bool, device=self.rows.device)
        return self.trainable


class ModelRuntime:
    """NF4 (or plain) runtime with hooks, ablation and trainable row deltas."""

    def __init__(self, model: nn.Module, tokenizer: Any, cfg: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.deltas: dict[str, DeltaSpec] = {}
        self._delta_hooks: list[Any] = []
        self.delta_enabled = True
        self._modules = dict(model.named_modules())

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.target_names = self._discover_target_modules()
        if not self.target_names:
            raise RuntimeError(f"no target modules matched {cfg.layer_pool!r}")

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, cfg: TrainConfig) -> "ModelRuntime":
        dtype = torch_dtype(cfg.dtype)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "attn_implementation": cfg.attn_implementation,
            "low_cpu_mem_usage": True,
        }
        if cfg.quant == "nf4":
            if cfg.device != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("NF4 runtime requires CUDA")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        elif cfg.quant == "none":
            pass
        else:
            raise ValueError(f"unknown quant mode: {cfg.quant}")

        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **kwargs)
        if cfg.quant == "none":
            model.to(cfg.device)
        if cfg.grad_checkpoint:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return cls(model, tokenizer, cfg)

    def _discover_target_modules(self) -> list[str]:
        pat = re.compile(self.cfg.layer_pool)
        names = []
        for name, mod in self.model.named_modules():
            if not pat.fullmatch(name):
                continue
            if isinstance(mod, nn.Linear) or type(mod).__name__ in {
                "Linear4bit",
                "Linear8bitLt",
            }:
                names.append(name)
        def key(name: str) -> tuple[int, str]:
            m = re.search(r"layers\.(\d+)\.", name)
            return (int(m.group(1)) if m else -1, name)
        return sorted(names, key=key)

    # -------------------------------------------------------------- metadata
    def module(self, name: str) -> nn.Module:
        return self._modules[name]

    def num_rows(self, name: str) -> int:
        # bnb Params4bit stores packed data; its .shape is not the logical
        # out_features.  Linear4bit carries the original in/out features.
        mod = self.module(name)
        return int(getattr(mod, "out_features", mod.weight.shape[0]))

    def in_features(self, name: str) -> int:
        mod = self.module(name)
        return int(getattr(mod, "in_features", mod.weight.shape[1]))

    def row_norm(self, addrs: list[RowAddr]) -> dict[str, float]:
        """Read base row norms from the runtime weight (falls back to 1.0)."""
        out: dict[str, float] = {}
        for a in addrs:
            w = self.module(a.layer).weight
            try:
                row = w[a.row].detach().float()
                out[f"{a.layer}:{a.row}"] = float(torch.linalg.vector_norm(row).item())
            except Exception:
                out[f"{a.layer}:{a.row}"] = 1.0
        return out

    def info(self) -> dict[str, Any]:
        return {
            "model_path": self.cfg.model_path,
            "quant": self.cfg.quant,
            "dtype": self.cfg.dtype,
            "device": str(self.device),
            "num_target_modules": len(self.target_names),
            "target_layers": len({n.split(".layers.")[1].split(".")[0] for n in self.target_names}),
            "memory": cuda_memory_stats(),
        }

    # --------------------------------------------------------------- encode
    def encode(
        self, prompt: str, completion: str, append_eos: bool | None = None
    ) -> EncodedInput:
        if append_eos is None:
            append_eos = bool(self.cfg.append_eos)
        eos_id = self.tokenizer.eos_token_id if append_eos else None
        return encode_prompt_completion(
            self.tokenizer,
            prompt,
            completion,
            chat=self.cfg.chat,
            max_len=self.cfg.max_seq_len,
            append_token_id=eos_id,
        )

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device)

    # ------------------------------------------------------------- forward
    @contextlib.contextmanager
    def delta_context(self, enabled: bool) -> Iterator[None]:
        old = self.delta_enabled
        self.delta_enabled = enabled
        try:
            yield
        finally:
            self.delta_enabled = old

    def forward_logits(self, encoded: EncodedInput, use_deltas: bool = True) -> torch.Tensor:
        ids = self._to_device(encoded.input_ids)
        mask = self._to_device(encoded.attention_mask)
        with self.delta_context(use_deltas):
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return out.logits

    @torch.no_grad()
    def score_completion(
        self, prompt: str, completion: str, use_deltas: bool = True
    ) -> tuple[float, float, int]:
        enc = self.encode(prompt, completion)
        logits = self.forward_logits(enc, use_deltas=use_deltas)
        return mean_logprob_from_logits(logits, enc.input_ids, enc.span)

    @torch.no_grad()
    def generate(
        self, prompt: str, use_deltas: bool = True, max_new_tokens: int | None = None
    ) -> str:
        prefix = build_prefix_ids(self.tokenizer, prompt, chat=self.cfg.chat)
        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        kwargs = dict(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=int(max_new_tokens or self.cfg.max_new_tokens),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with self.delta_context(use_deltas):
            out = self.model.generate(**kwargs)
        new = out[0, ids.shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    # ------------------------------------------------------------ hook capture
    def capture_activation_scores(
        self, encoded: EncodedInput, names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Per-channel max |activation| over the encoded completion span."""
        names = names or self.target_names
        s, e = encoded.span
        scores: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name: str):
            def hook(mod: nn.Module, inp: Any, out: Any) -> None:
                if not isinstance(out, torch.Tensor):
                    return
                sl = out[0, s : e + 1, :]
                scores[name] = sl.abs().amax(dim=0).detach().float().cpu()

            return hook

        try:
            for name in names:
                handles.append(self.module(name).register_forward_hook(make_hook(name)))
            with torch.no_grad():
                # Apply any already-attached sidecar; in single-call mode there
                # are no patches yet, so this is identical to the NF4 base.
                self.forward_logits(encoded, use_deltas=True)
        finally:
            for h in handles:
                h.remove()
        return scores

    def capture_gradxact_scores(
        self, encoded: EncodedInput, names: list[str] | None = None
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Per-channel sum |h * dL/dh| over the completion span.

        This is the spec's Rule B.  The backward is taken with respect to
        ``-log p(target)``; all model parameters are frozen, so only activation
        gradients are materialized.
        """
        names = names or self.target_names
        s, e = encoded.span
        acts: dict[str, torch.Tensor] = {}
        grads: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name: str):
            def hook(mod: nn.Module, inp: Any, out: Any) -> None:
                if not isinstance(out, torch.Tensor):
                    return
                sl = out[0, s : e + 1, :]
                acts[name] = sl
                if sl.requires_grad:
                    sl.register_hook(
                        lambda g, name=name: grads.__setitem__(name, g.detach())
                    )

            return hook

        embeds = self.model.get_input_embeddings()(self._to_device(encoded.input_ids))
        embeds.requires_grad_(True)
        loss_val = float("nan")
        try:
            for name in names:
                handles.append(self.module(name).register_forward_hook(make_hook(name)))
            with torch.enable_grad():
                ids = self._to_device(encoded.input_ids)
                mask = self._to_device(encoded.attention_mask)
                out = self.model(
                    inputs_embeds=embeds, attention_mask=mask, use_cache=False
                )
                lp = torch.log_softmax(out.logits[0, s - 1 : e, :].float(), dim=-1)
                tgt = ids[0, s : e + 1]
                loss = -lp.gather(-1, tgt.unsqueeze(-1)).mean()
                loss.backward()
                loss_val = float(loss.detach().item())
            scores = {}
            for name in names:
                h = acts.get(name)
                g = grads.get(name)
                if h is None or g is None:
                    continue
                scores[name] = (
                    h.detach().abs() * g.detach().abs()
                ).sum(dim=0).float().cpu()
        finally:
            for h in handles:
                h.remove()
            del embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores, loss_val

    def capture_raw_linear(self, name: str, encoded: EncodedInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture the module input and output (for hook-correctness tests)."""
        captured: dict[str, torch.Tensor] = {}
        mod = self.module(name)

        def pre_hook(m: nn.Module, inp: Any) -> None:
            captured["input"] = inp[0].detach()

        def fwd_hook(m: nn.Module, inp: Any, out: Any) -> None:
            captured["output"] = out.detach()

        h1 = mod.register_forward_pre_hook(pre_hook)
        h2 = mod.register_forward_hook(fwd_hook)
        try:
            with torch.no_grad():
                self.forward_logits(encoded, use_deltas=False)
        finally:
            h1.remove()
            h2.remove()
        return captured["input"], captured["output"]

    # --------------------------------------------------------------- ablation
    @contextlib.contextmanager
    def ablate(self, rows_by_layer: dict[str, list[int]]) -> Iterator[None]:
        """Temporarily zero output channels (exact row ablation)."""
        handles = []
        for name, rows in rows_by_layer.items():
            if not rows:
                continue
            idx = torch.tensor(sorted(set(int(r) for r in rows)), dtype=torch.long, device=self.device)
            mod = self.module(name)

            def make_hook(index: torch.Tensor):
                def hook(m: nn.Module, inp: Any, out: Any) -> Any:
                    if not isinstance(out, torch.Tensor):
                        return out
                    return out.index_fill(-1, index, 0.0)

                return hook

            handles.append(mod.register_forward_hook(make_hook(idx)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------ row patches
    def _dequant_rows(self, name: str, rows: list[int]) -> torch.Tensor:
        """deq(W_q)[rows] as float32, for splice-mode subtraction."""
        mod = self.module(name)
        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        if self.cfg.quant == "nf4":
            import bitsandbytes as bnb

            w = mod.weight
            deq = bnb.functional.dequantize_4bit(
                w, quant_state=getattr(w, "quant_state", None)
            )
            return deq[idx].detach().float()
        return mod.weight.detach()[idx].float()

    def attach_rows(
        self,
        rows_by_layer: dict[str, list[int]],
        values_by_layer: dict[str, torch.Tensor] | None = None,
        trainable_rows_by_layer: dict[str, list[int]] | None = None,
        mode: str | None = None,
    ) -> dict[str, DeltaSpec]:
        """Attach row patches.

        ``mode="delta"``: values = ΔR (default zeros), effective row = D+ΔR.
        ``mode="splice"``: values = P (aligned to rows), effective row = P;
        the hook subtracts D so the NF4 base contribution is replaced.

        ``trainable_rows_by_layer`` may contain only the new failure's rows;
        all other rows (e.g. a previous sidecar) are applied but frozen.
        """
        self.remove_deltas()
        mode = mode or self.cfg.param_mode
        if mode not in {"delta", "splice"}:
            raise ValueError(f"unknown row mode: {mode}")

        for name, rows in rows_by_layer.items():
            uniq = sorted({int(r) for r in rows})
            if not uniq:
                continue
            n = self.num_rows(name)
            if uniq[0] < 0 or uniq[-1] >= n:
                raise IndexError(f"{name}: rows out of range (n_rows={n})")
            in_features = self.in_features(name)
            idx = torch.tensor(uniq, dtype=torch.long, device=self.device)

            if values_by_layer is not None and name in values_by_layer:
                vals = values_by_layer[name].detach().float().cpu()
                if tuple(vals.shape) != (len(uniq), in_features):
                    raise ValueError(
                        f"{name}: values shape {tuple(vals.shape)} != "
                        f"{(len(uniq), in_features)}"
                    )
            elif mode == "delta":
                vals = torch.zeros((len(uniq), in_features), dtype=torch.float32)
            else:
                raise ValueError(f"{name}: splice mode requires values_by_layer")

            if trainable_rows_by_layer is None:
                trainable = torch.ones(len(uniq), dtype=torch.bool)
            else:
                wanted = {int(r) for r in trainable_rows_by_layer.get(name, [])}
                trainable = torch.tensor([r in wanted for r in uniq], dtype=torch.bool)
            requires_grad = bool(trainable.any())

            base_rows = self._dequant_rows(name, uniq) if mode == "splice" else None
            init_vals = vals.to(device=self.device, dtype=torch.float32)
            param = nn.Parameter(init_vals.clone(), requires_grad=requires_grad)
            spec = DeltaSpec(
                name=name,
                rows=idx,
                param=param,
                in_features=in_features,
                mode=mode,
                base_rows=base_rows,
                trainable=trainable.to(self.device),
                init_values=init_vals.detach().clone(),
            )

            def make_hook(s: DeltaSpec):
                def hook(m: nn.Module, inp: Any, out: Any) -> Any:
                    if not self.delta_enabled or not isinstance(out, torch.Tensor):
                        return out
                    x = inp[0]
                    corr = F.linear(x.float(), s.param)
                    if s.mode == "splice" and s.base_rows is not None:
                        corr = corr - F.linear(x.float(), s.base_rows)
                    return torch.index_add(out, -1, s.rows, corr.to(out.dtype))

                return hook

            self._delta_hooks.append(self.module(name).register_forward_hook(make_hook(spec)))
            self.deltas[name] = spec
        return self.deltas

    def attach_deltas(self, rows_by_layer: dict[str, list[int]]) -> dict[str, DeltaSpec]:
        """Backward-compatible single-call delta attachment."""
        return self.attach_rows(rows_by_layer, mode="delta", trainable_rows_by_layer=None)

    def delta_parameters(self) -> list[nn.Parameter]:
        return [s.param for s in self.deltas.values() if s.param.requires_grad]

    def zero_frozen_grads(self) -> None:
        for spec in self.deltas.values():
            if spec.param.grad is None or spec.trainable is None:
                continue
            spec.param.grad[~spec.trainable] = 0.0

    def set_delta_enabled(self, enabled: bool) -> None:
        self.delta_enabled = bool(enabled)

    def delta_norms(self, trainable_only: bool = False) -> dict[str, float]:
        out: dict[str, float] = {}
        for spec in self.deltas.values():
            norms = torch.linalg.vector_norm(spec.param.detach().float(), dim=1).cpu()
            mask = spec.trainable_mask().cpu()
            for i, (row, norm) in enumerate(zip(spec.rows.cpu().tolist(), norms.tolist())):
                if trainable_only and not bool(mask[i]):
                    continue
                out[f"{spec.name}:{row}"] = float(norm)
        return out

    def delta_values(self) -> dict[str, torch.Tensor]:
        return {name: spec.param.detach().float().cpu() for name, spec in self.deltas.items()}

    def patch_rows(self) -> dict[str, list[int]]:
        return {name: spec.rows.cpu().tolist() for name, spec in self.deltas.items()}

    def patch_modes(self) -> dict[str, str]:
        return {name: spec.mode for name, spec in self.deltas.items()}

    def remove_deltas(self) -> None:
        for h in self._delta_hooks:
            h.remove()
        self._delta_hooks.clear()
        self.deltas.clear()

    # ------------------------------------------------------------- background
    def background_logprobs(
        self,
        batch: dict[str, torch.Tensor],
        use_deltas: bool = True,
        no_grad: bool = True,
    ) -> torch.Tensor:
        """Next-token log-probabilities at the last non-pad token of each text.

        ``no_grad=False`` keeps the graph so the KL term can regularize the
        deltas; the returned tensor then stays on the compute device.
        """
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx, self.delta_context(use_deltas):
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        last = mask.sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(ids.shape[0], device=self.device)
        logits = out.logits[rows, last, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        return logp.cpu() if no_grad else logp

    # ---------------------------------------------------------------- cleanup
    def memory(self) -> dict[str, float] | None:
        return cuda_memory_stats()

    def close(self) -> None:
        self.remove_deltas()
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rows_by_layer(addrs: list[RowAddr]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for a in addrs:
        out.setdefault(a.layer, []).append(int(a.row))
    return out
