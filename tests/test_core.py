from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from qlamt.ablate import Ablator
from qlamt.locate import locate
from qlamt.runtime import ModelRuntime
from qlamt.types import Failure, RowAddr, TrainConfig

from tests.conftest import DummyTokenizer


NAME = "model.layers.0.mlp.down_proj"


def tiny_model(seed: int = 0) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    return Qwen2ForCausalLM(cfg)


def tiny_cfg(**kwargs) -> TrainConfig:
    base = dict(
        model_path="",
        quant="none",
        device="cpu",
        dtype="float32",
        chat=False,
        topk_per_layer=2,
        use_gradxact=False,
        max_candidates=32,
        balance_layers=False,
        group_size=2,
        calibrate_epsilon=False,
        epsilon_ablate=1e-6,
        epsilon_row=1e-6,
        min_named=2,
        max_named=8,
        max_seq_len=64,
        eval_every=10,
        steps=2,
    )
    base.update(kwargs)
    return TrainConfig(**base)


def mk_runtime(seed: int = 0, **cfg_kwargs) -> ModelRuntime:
    return ModelRuntime(tiny_model(seed), DummyTokenizer(), tiny_cfg(**cfg_kwargs))


def failure() -> Failure:
    return Failure(
        id="tiny",
        prompt="hello world and more text",
        wrong_span="bad answer",
        target_span="good answer",
        paraphrases=["another prompt here"],
    )


def test_target_modules_and_shapes():
    rt = mk_runtime()
    assert NAME in rt.target_names
    assert rt.num_rows(NAME) == 16
    assert rt.in_features(NAME) == 32
    assert rt.module(NAME).bias is None


def test_hook_activation_matches_manual_linear():
    rt = mk_runtime()
    enc = rt.encode("hello", "world")
    inp, out = rt.capture_raw_linear(NAME, enc)
    manual = F.linear(inp, rt.module(NAME).weight)
    assert out.shape == manual.shape
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-5)


def test_ablation_is_exact_row_zeroing():
    rt = mk_runtime()
    enc = rt.encode("hello", "world")
    rows = [0, 3, 7]
    base = rt.forward_logits(enc, use_deltas=False)
    with rt.ablate({NAME: rows}):
        ablated = rt.forward_logits(enc, use_deltas=False)

    ref_model = copy.deepcopy(rt.model)
    with torch.no_grad():
        ref_model.get_submodule(NAME).weight[rows] = 0.0
    ref = ref_model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        use_cache=False,
    ).logits
    assert torch.allclose(ablated, ref, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(base, ref, atol=1e-6)


def test_delta_zero_equivalence_and_gradient_sparsity():
    rt = mk_runtime()
    enc = rt.encode("hello", "world")
    base = rt.forward_logits(enc, use_deltas=False)

    specs = rt.attach_deltas({NAME: [0, 3]})
    spec = specs[NAME]
    assert spec.param.shape == (2, rt.in_features(NAME))
    assert float(spec.param.detach().abs().max()) == 0.0

    zero = rt.forward_logits(enc, use_deltas=True)
    assert torch.allclose(base, zero, atol=1e-7, rtol=1e-6)

    logits = rt.forward_logits(enc, use_deltas=True)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        enc.input_ids[:, 1:].reshape(-1),
    )
    loss.backward()
    assert spec.param.grad is not None
    assert float(spec.param.grad.abs().sum()) > 0
    assert all(p.grad is None for p in rt.model.parameters())

    with torch.no_grad():
        spec.param[0, 0] = 0.1
    changed = rt.forward_logits(enc, use_deltas=True)
    assert not torch.allclose(base, changed, atol=1e-7)
    rt.remove_deltas()


def test_locator_returns_topk_per_layer():
    rt = mk_runtime()
    cands, rep = locate(rt, failure(), tiny_cfg(use_gradxact=False))
    n_layers = len({n.split(".layers.")[1].split(".")[0] for n in rt.target_names})
    assert len(cands) == 2 * n_layers
    assert rep["n_candidates"] == len(cands)
    assert all(v <= 2 for v in rep["per_layer"].values())
    assert all(0 <= c.addr.row < rt.num_rows(c.addr.layer) for c in cands)


def test_delta_matches_weight_replacement():
    rt = mk_runtime()
    enc = rt.encode("hello", "world")
    rows = [0, 5]
    base_model = copy.deepcopy(rt.model)
    specs = rt.attach_deltas({NAME: rows})
    with torch.no_grad():
        specs[NAME].param.copy_(torch.randn_like(specs[NAME].param) * 0.01)
    got = rt.forward_logits(enc, use_deltas=True)
    with torch.no_grad():
        base_model.get_submodule(NAME).weight[rows] += specs[NAME].param
    ref = base_model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        use_cache=False,
    ).logits
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    rt.remove_deltas()


def test_ablation_reproducible():
    rt = mk_runtime()
    f = failure()
    cfg = tiny_cfg()
    rows = [RowAddr(NAME, 1), RowAddr(NAME, 4)]
    a = Ablator(rt, f, cfg)
    b = Ablator(rt, f, cfg)
    d1 = a.measure(rows)
    d2 = b.measure(rows)
    assert abs(d1.m - d2.m) < 1e-6
    assert abs(d1.target_lp - d2.target_lp) < 1e-6


def test_locator_gradxact_runs():
    rt = mk_runtime()
    cands, rep = locate(rt, failure(), tiny_cfg(use_gradxact=True, topk_per_layer=1))
    assert len(cands) >= 1
    assert rep["gradxact_loss"] is not None
    assert all(c.gradxact_score >= 0 for c in cands)


def test_ablator_confirm_fallback_and_report():
    rt = mk_runtime()
    cfg = tiny_cfg()
    cands, _ = locate(rt, failure(), cfg)
    ablator = Ablator(rt, failure(), cfg)
    named, rep = ablator.confirm(cands)
    assert len(named) >= 1
    assert rep["n_named"] == len(named)
    assert rep["forward_pairs"] >= 1
    assert all(n.addr.layer in rt.target_names for n in named)


def test_splice_mode_matches_weight_replacement():
    rt = mk_runtime()
    enc = rt.encode("hello", "world")
    rows = [0, 3]
    base_model = copy.deepcopy(rt.model)
    base_vals = rt.module(NAME).weight.detach()[rows].float().cpu().clone()
    rt.attach_rows(
        {NAME: rows},
        values_by_layer={NAME: base_vals},
        trainable_rows_by_layer=None,
        mode="splice",
    )
    # P == deq(W) at step 0 -> identical to the base.
    base = rt.forward_logits(enc, use_deltas=False)
    spliced = rt.forward_logits(enc, use_deltas=True)
    assert torch.allclose(base, spliced, atol=1e-6, rtol=1e-5)

    with torch.no_grad():
        rt.deltas[NAME].param[0, 0] += 0.1
    got = rt.forward_logits(enc, use_deltas=True)
    with torch.no_grad():
        base_model.get_submodule(NAME).weight[0, 0] += 0.1
    ref = base_model(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False
    ).logits
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    rt.remove_deltas()


def test_frozen_sidecar_rows_do_not_update():
    rt = mk_runtime()
    enc = rt.encode("hello", "world")
    rt.attach_rows(
        {NAME: [0, 3]},
        trainable_rows_by_layer={NAME: [0]},
        mode="delta",
    )
    logits = rt.forward_logits(enc, use_deltas=True)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        enc.input_ids[:, 1:].reshape(-1),
    )
    loss.backward()
    rt.zero_frozen_grads()
    spec = rt.deltas[NAME]
    assert spec.param.grad is not None
    assert float(spec.param.grad[0].abs().sum()) > 0
    assert float(spec.param.grad[1].abs().sum()) == 0.0
    assert all(p.grad is None for p in rt.model.parameters())
    rt.remove_deltas()
