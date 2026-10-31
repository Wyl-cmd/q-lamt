from __future__ import annotations

import torch
from safetensors.torch import save_file

from qlamt.ledger import LedgerManager


NAME = "model.layers.0.mlp.down_proj.weight"
OTHER = "model.layers.0.mlp.gate_proj.weight"


def make_model_dir(d):
    a = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    b = torch.randn(8, 4)
    save_file({NAME: a, OTHER: b}, str(d / "model.safetensors"))
    (d / "config.json").write_text("{}")
    return a, b


def test_row_roundtrip_write_and_rollback(project_tmp):
    d = project_tmp / "model"
    d.mkdir()
    original, other_before = make_model_dir(d)
    led = LedgerManager(d)

    rows = [1, 5]
    got = led.read_rows(NAME, rows)
    assert torch.allclose(got, original[rows])

    undo = led.snapshot_rows({NAME: rows}, d / "undo")
    new = got + 0.5
    rep = led.write_rows({NAME: new}, {NAME: rows}, undo_dir=None, verify=True)
    assert rep["verified"] is True
    assert torch.allclose(led.read_rows(NAME, rows), new)
    # untouched tensor and untouched rows are unchanged
    assert torch.allclose(led.read_rows(NAME, [0, 2, 3, 4, 6, 7]), original[[0, 2, 3, 4, 6, 7]])
    assert torch.allclose(led.read_rows(OTHER, list(range(8))), other_before)

    led.rollback(undo)
    assert torch.allclose(led.read_rows(NAME, rows), original[rows])
    # idempotent rollback
    led.rollback(undo)
    assert torch.allclose(led.read_rows(NAME, rows), original[rows])


def test_prepare_hardlink_copy_isolated(project_tmp):
    src = project_tmp / "src"
    src.mkdir()
    original, _ = make_model_dir(src)
    led = LedgerManager.prepare(src, project_tmp / "ledger")
    rows = [2, 4]
    led.write_rows(
        {NAME: led.read_rows(NAME, rows) + 1.0},
        {NAME: rows},
        undo_dir=None,
        verify=True,
    )
    # The source model directory must be untouched.
    src_led = LedgerManager(src)
    assert torch.allclose(src_led.read_rows(NAME, rows), original[rows])


def test_row_norms(project_tmp):
    d = project_tmp / "model"
    d.mkdir()
    original, _ = make_model_dir(d)
    led = LedgerManager(d)
    norms = led.row_norms({NAME: [0, 3]})
    expected = torch.linalg.vector_norm(original[[0, 3]], dim=1)
    for key, exp in zip([f"{NAME}:0", f"{NAME}:3"], expected.tolist()):
        assert abs(norms[key] - exp) < 1e-6


def test_module_name_mapping(project_tmp):
    d = project_tmp / "model"
    d.mkdir()
    original, _ = make_model_dir(d)
    led = LedgerManager(d)
    layer = NAME.removesuffix(".weight")
    rows = [1, 6]
    assert torch.allclose(led.read_rows(layer, rows), original[rows])
    rep = led.write_rows(
        {layer: led.read_rows(layer, rows) + 2.0},
        {layer: rows},
        undo_dir=None,
        verify=True,
    )
    assert rep["verified"] is True
    assert torch.allclose(led.read_rows(layer, rows), original[rows] + 2.0)
