from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from qlamt.ledger import LedgerManager
from qlamt.types import Failure, TrainConfig
from qlamt.versions import VersionStore, shard_and_patch


NAME = "model.layers.0.mlp.down_proj"


def _model_dir(project_tmp):
    d = project_tmp / "model"
    d.mkdir()
    save_file(
        {
            f"{NAME}.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
        },
        str(d / "model.safetensors"),
    )
    (d / "config.json").write_text("{}", encoding="utf-8")
    return d


def _cfg(d):
    return TrainConfig(
        model_path=str(d),
        quant="none",
        device="cpu",
        dtype="float32",
        mode="sidecar",
        param_mode="splice_fp16",
    )


def _failure(fid):
    return Failure(
        id=fid,
        prompt="p",
        wrong_span="w",
        target_span="t",
        paraphrases=[],
        category="fact",
    )


def test_commit_load_current_and_rollback(project_tmp):
    d = _model_dir(project_tmp)
    cfg = _cfg(d)
    store = VersionStore(d, project_tmp / "store")
    base_sha = store.base_sha
    v1 = store.commit(
        {NAME: [1, 3]},
        {NAME: torch.randn(2, 4)},
        "splice_fp16",
        _failure("f1"),
        ["w"],
        {"hybrid_fix_ratio": 1.0},
        cfg,
    )
    assert store.current_id() == v1.version_id
    assert v1.manifest["base_sha256"] == base_sha
    assert v1.manifest["status"] == "accepted"
    loaded = store.load(v1.version_id)
    assert loaded.rows_by_layer[NAME] == [1, 3]
    assert loaded.values_by_layer[NAME].shape == (2, 4)
    assert (project_tmp / "store" / "base" / "model.safetensors").exists()

    v2 = store.commit(
        {NAME: [1, 3, 5]},
        {NAME: torch.randn(3, 4)},
        "splice_fp16",
        _failure("f2"),
        ["w"],
        {"hybrid_fix_ratio": 1.0},
        cfg,
        parent=v1.version_id,
    )
    assert store.current_id() == v2.version_id
    assert len(store.regression_items()) == 2

    store.rollback(v1.version_id)
    assert store.current_id() == v1.version_id
    with pytest.raises(RuntimeError):
        store.commit(
            {NAME: [0]},
            {NAME: torch.zeros(1, 4)},
            "splice_fp16",
            _failure("stale"),
            ["w"],
            {"hybrid_fix_ratio": 1.0},
            cfg,
            expected_current=v2.version_id,
            check_current=True,
        )
    assert [x["version_id"] for x in store.regression_items()] == [v1.version_id]
    # base is untouched
    assert store.base_sha == base_sha

    v3 = store.commit(
        {NAME: [7]},
        {NAME: torch.randn(1, 4)},
        "splice_fp16",
        _failure("f3"),
        ["w"],
        {"hybrid_fix_ratio": 0.0},
        cfg,
        parent=v1.version_id,
        status="rejected",
        advance=False,
    )
    assert store.current_id() == v1.version_id
    assert v3.manifest["status"] == "rejected"
    assert {m["version_id"] for m in store.history()} >= {
        v1.version_id,
        v2.version_id,
        v3.version_id,
    }
    # manifests are valid JSON
    for m in store.history():
        json.dumps(m)


def test_shard_and_patch_streaming(project_tmp):
    d = _model_dir(project_tmp)
    out = project_tmp / "release"
    rows = {NAME: [1, 3]}
    vals = torch.ones(2, 4)
    rep = shard_and_patch(
        d,
        out,
        rows,
        {NAME: vals},
        "delta",
        max_shard_bytes=100,
    )
    assert rep["verified"] is True
    assert rep["sharded"] is True
    assert (out / "model.safetensors.index.json").exists()
    assert (out / "config.json").exists()
    led = LedgerManager(out)
    got = led.read_rows(NAME, rows[NAME])
    expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)[rows[NAME]] + vals
    assert torch.allclose(got, expected)
    # untouched tensor survives
    assert torch.allclose(
        led.read_rows("model.layers.0.mlp.gate_proj", [0, 7]),
        torch.ones(2, 4),
    )
