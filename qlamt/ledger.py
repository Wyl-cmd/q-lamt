from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .utils import sha256_file, write_json


WEIGHT_SUFFIXES = (".safetensors",)


class LedgerManager:
    """fp16/bf16 ledger: row-level reads, undo snapshots and atomic write-back.

    The ledger is a model directory.  It may be a hard-link copy of the source
    model: writes use ``temp + os.replace``, which breaks the link only in the
    ledger directory and leaves the source untouched.
    """

    def __init__(self, model_dir: str | Path, max_rewrite_gb: float = 6.0):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(self.model_dir)
        self.max_rewrite_bytes = int(max_rewrite_gb * 2**30)
        self.file_map: dict[str, Path] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self._discover()

    # --------------------------------------------------------------- discover
    def _discover(self) -> None:
        index_path = self.model_dir / "model.safetensors.index.json"
        files: list[Path] = []
        if index_path.exists():
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            for name, filename in idx["weight_map"].items():
                self.file_map[name] = self.model_dir / filename
            files = sorted(set(self.file_map.values()))
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no *.safetensors in {self.model_dir}")
            for f in files:
                with safe_open(str(f), framework="pt", device="cpu") as sf:
                    for name in sf.keys():
                        self.file_map[name] = f
        if not self.file_map:
            raise RuntimeError("empty ledger")
        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    sl = sf.get_slice(name)
                    self.shapes[name] = tuple(sl.get_shape())
                    try:
                        self.dtypes[name] = str(sl.get_dtype())
                    except Exception:
                        self.dtypes[name] = "unknown"

    # ------------------------------------------------------------------- read
    def tensor_name(self, layer: str) -> str:
        """Map a module name (RowAddr.layer) to its safetensors tensor name."""
        if layer in self.file_map:
            return layer
        weight = f"{layer}.weight"
        if weight in self.file_map:
            return weight
        raise KeyError(f"tensor not in ledger: {layer} (also tried {weight})")

    def in_features(self, layer: str) -> int:
        return int(self.shapes[self.tensor_name(layer)][1])

    def _check(self, name: str, rows: Iterable[int]) -> list[int]:
        tname = self.tensor_name(name)
        n = self.shapes[tname][0]
        out = [int(r) for r in rows]
        if any(r < 0 or r >= n for r in out):
            raise IndexError(f"{name}: row out of range (n_rows={n})")
        return out

    def read_rows(self, name: str, rows: Iterable[int]) -> torch.Tensor:
        rows = self._check(name, rows)
        tname = self.tensor_name(name)
        with safe_open(str(self.file_map[tname]), framework="pt", device="cpu") as f:
            t = f.get_tensor(tname)
        return t[rows].clone()

    def read_rows_by_layer(self, rows_by_layer: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        return {name: self.read_rows(name, rows) for name, rows in rows_by_layer.items()}

    def row_norms(self, rows_by_layer: dict[str, list[int]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, rows in rows_by_layer.items():
            vals = self.read_rows(name, rows).float()
            norms = torch.linalg.vector_norm(vals, dim=1)
            for r, n in zip(rows, norms.tolist()):
                out[f"{name}:{int(r)}"] = float(n)
        return out

    def original_rows(self, rows_by_layer: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        return self.read_rows_by_layer(rows_by_layer)

    # ---------------------------------------------------------------- snapshot
    def snapshot_rows(
        self,
        rows_by_layer: dict[str, list[int]],
        undo_dir: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        undo_dir = Path(undo_dir)
        undo_dir.mkdir(parents=True, exist_ok=True)
        rows = self.read_rows_by_layer(rows_by_layer)
        values_path = undo_dir / "rows.safetensors"
        save_file({k: v.contiguous() for k, v in rows.items()}, str(values_path))
        meta = {
            "rows": {k: [int(x) for x in v] for k, v in rows_by_layer.items()},
            "values_file": values_path.name,
            "model_dir": str(self.model_dir.resolve()),
            "dtypes": {k: str(v) for k, v in self.dtypes.items() if k in rows},
        }
        if extra:
            meta.update(extra)
        write_json(undo_dir / "meta.json", meta)
        return undo_dir

    # -------------------------------------------------------------- write-back
    def write_rows(
        self,
        values_by_layer: dict[str, torch.Tensor],
        rows_by_layer: dict[str, list[int]],
        undo_dir: str | Path | None = None,
        verify: bool = True,
    ) -> dict[str, Any]:
        if set(values_by_layer) != set(rows_by_layer):
            raise ValueError("values_by_layer and rows_by_layer must have the same keys")
        if undo_dir is not None:
            self.snapshot_rows(rows_by_layer, undo_dir)

        # Group edits by shard.  A shard is rewritten atomically; the rewrite
        # loads the shard, so very large single-file models are rejected here
        # rather than silently consuming host RAM.
        by_file: dict[Path, list[str]] = {}
        for name in values_by_layer:
            by_file.setdefault(self.file_map[self.tensor_name(name)], []).append(name)

        report: dict[str, Any] = {"files": {}, "verified": False}
        for f, names in by_file.items():
            size = f.stat().st_size
            if size > self.max_rewrite_bytes:
                raise RuntimeError(
                    f"shard {f.name} is {size / 2**30:.2f} GiB > max_rewrite_gb; "
                    "use row-patch output for this model"
                )
            sha_before = sha256_file(f)
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                metadata = sf.metadata()
                tensors = {k: sf.get_tensor(k) for k in sf.keys()}
            for name in names:
                tname = self.tensor_name(name)
                rows = self._check(name, rows_by_layer[name])
                vals = values_by_layer[name].detach().cpu()
                if vals.shape != (len(rows),) + tuple(self.shapes[tname][1:]):
                    raise ValueError(
                        f"{name}: values shape {tuple(vals.shape)} does not match rows"
                    )
                tensors[tname][rows] = vals.to(tensors[tname].dtype)
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f".{f.name}.", suffix=".tmp", dir=str(f.parent)
            )
            os.close(tmp_fd)
            try:
                save_file(
                    {k: v.contiguous() for k, v in tensors.items()},
                    tmp_name,
                    metadata=metadata,
                )
                os.replace(tmp_name, f)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            sha_after = sha256_file(f)
            report["files"][f.name] = {
                "sha256_before": sha_before,
                "sha256_after": sha_after,
                "size_bytes": size,
            }
            del tensors

        if verify:
            report["verified"] = self.verify_rows(values_by_layer, rows_by_layer)
        return report

    def verify_rows(
        self, values_by_layer: dict[str, torch.Tensor], rows_by_layer: dict[str, list[int]]
    ) -> bool:
        for name, rows in rows_by_layer.items():
            got = self.read_rows(name, rows)
            want = values_by_layer[name].detach().cpu().to(got.dtype)
            if not torch.allclose(got.float(), want.float(), atol=0.0, rtol=0.0):
                return False
        return True

    # --------------------------------------------------------------- rollback
    def rollback(self, undo: str | Path) -> dict[str, Any]:
        undo = Path(undo)
        meta = json.loads((undo / "meta.json").read_text(encoding="utf-8"))
        values_path = undo / str(meta["values_file"])
        values = load_file(str(values_path))
        rows_by_layer = {k: [int(x) for x in v] for k, v in meta["rows"].items()}
        missing = [k for k in rows_by_layer if k not in values]
        if missing:
            raise KeyError(f"undo snapshot missing tensors: {missing}")
        report = self.write_rows(
            {k: values[k] for k in rows_by_layer},
            rows_by_layer,
            undo_dir=None,
            verify=True,
        )
        report["undo"] = str(undo.resolve())
        return report

    # ------------------------------------------------------------------ copy
    @classmethod
    def prepare(
        cls,
        source_dir: str | Path,
        ledger_dir: str | Path,
        force: bool = False,
    ) -> "LedgerManager":
        """Create a ledger copy of the model directory (hard-link when possible)."""
        source_dir = Path(source_dir)
        ledger_dir = Path(ledger_dir)
        if ledger_dir.exists() and any(ledger_dir.glob("*.safetensors")):
            if not force:
                return cls(ledger_dir)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        for src in source_dir.iterdir():
            if not src.is_file():
                continue
            dst = ledger_dir / src.name
            if dst.exists() and not force:
                continue
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        return cls(ledger_dir)
