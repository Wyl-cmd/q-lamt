from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .ledger import LedgerManager
from .runtime import ModelRuntime
from .types import Failure, TrainConfig
from .utils import json_safe, sha256_file, write_json


# ---------------------------------------------------------------------------
# Sidecar / version store
# ---------------------------------------------------------------------------
@dataclass
class Sidecar:
    version_id: str
    rows_by_layer: dict[str, list[int]]
    values_by_layer: dict[str, torch.Tensor]  # float32, aligned to rows
    param_mode: str
    manifest: dict[str, Any]
    failure: dict[str, Any] | None = None
    base_answers: list[str] | None = None

    def row_map(self, name: str) -> dict[int, torch.Tensor]:
        return {
            int(r): self.values_by_layer[name][i]
            for i, r in enumerate(self.rows_by_layer[name])
        }


class VersionStore:
    """Immutable base + content-addressed-ish version chain.

    Base is a hard-link copy under ``<root>/base``.  Each version owns a full
    row snapshot (float32) and a manifest; ``current`` is an atomic pointer.
    """

    def __init__(
        self,
        source_model_dir: str | Path,
        root: str | Path,
        prepare_base: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.root / "base"
        source = Path(source_model_dir)
        if prepare_base and (
            not self.base_dir.exists()
            or not any(self.base_dir.glob("*.safetensors"))
        ):
            self.ledger = LedgerManager.prepare(source, self.base_dir)
        else:
            self.ledger = LedgerManager(self.base_dir)
        self._base_sha_file = self.root / "base_sha.json"
        self.base_sha = self._model_sha()
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self.root / "current"
        self.releases_dir = self.root / "releases"
        self.releases_dir.mkdir(parents=True, exist_ok=True)

    def _model_sha(self) -> str:
        """Hash base weights once; reuse while size/mtime are unchanged.

        Hashing 15GB on every fix would dominate a 7B session.  The base is
        immutable in multi-round mode, so a size+mtime fingerprint is enough to
        invalidate the cache; the cached value still contains per-file sha256.
        """
        weights = sorted(self.ledger.model_dir.glob("*.safetensors"))
        if not weights:
            raise RuntimeError(f"no weights in {self.ledger.model_dir}")
        index = self.ledger.model_dir / "model.safetensors.index.json"
        files = weights + ([index] if index.exists() else [])
        fingerprint = [
            {
                "name": f.name,
                "size": int(f.stat().st_size),
                "mtime_ns": int(f.stat().st_mtime_ns),
            }
            for f in files
        ]
        if self._base_sha_file.exists():
            try:
                cached = json.loads(self._base_sha_file.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    return str(cached["composite_sha256"])
            except Exception:
                pass
        h = hashlib.sha256()
        file_hashes = {}
        for f in files:
            digest = sha256_file(f)
            file_hashes[f.name] = digest
            h.update(f.name.encode("utf-8"))
            h.update(digest.encode("ascii"))
        composite = h.hexdigest()
        tmp = self._base_sha_file.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "file_sha256": file_hashes,
                    "composite_sha256": composite,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self._base_sha_file)
        return composite

    # ------------------------------------------------------------- current
    def current_id(self) -> str | None:
        if not self.current_file.exists():
            return None
        text = self.current_file.read_text(encoding="utf-8").strip()
        return text or None

    def _set_current(self, version_id: str) -> None:
        vdir = self.versions_dir / version_id
        if not vdir.exists():
            raise FileNotFoundError(vdir)
        tmp = self.current_file.with_suffix(".tmp")
        tmp.write_text(version_id, encoding="utf-8")
        os.replace(tmp, self.current_file)

    def rollback(self, version_id: str) -> dict[str, Any]:
        manifest = self._read_manifest(version_id)
        if manifest.get("status") != "accepted":
            raise ValueError(f"version {version_id} is not accepted")
        if manifest.get("base_sha256") != self.base_sha:
            raise RuntimeError(
                f"version {version_id} belongs to a different base; refusing rollback"
            )
        self._set_current(version_id)
        return manifest

    # ------------------------------------------------------------- loading
    def _version_dir(self, version_id: str) -> Path:
        return self.versions_dir / version_id

    def _read_manifest(self, version_id: str) -> dict[str, Any]:
        p = self._version_dir(version_id) / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(p)
        return json.loads(p.read_text(encoding="utf-8"))

    def load(self, version_id: str) -> Sidecar:
        vdir = self._version_dir(version_id)
        manifest = self._read_manifest(version_id)
        meta = json.loads((vdir / "sidecar_meta.json").read_text(encoding="utf-8"))
        if meta.get("base_sha256") != self.base_sha:
            raise RuntimeError(
                f"sidecar {version_id} belongs to base {meta.get('base_sha256')}, "
                f"store base is {self.base_sha}"
            )
        tensors = load_file(str(vdir / "sidecar.safetensors"))
        rows = {k: [int(r) for r in v] for k, v in meta["rows"].items()}
        values = {}
        for name, rows_i in rows.items():
            tensor = tensors[name].float().cpu()
            if tensor.ndim != 2 or tensor.shape[0] != len(rows_i):
                raise ValueError(f"sidecar {version_id}: bad tensor shape for {name}")
            if tensor.shape[1] != self.ledger.in_features(name):
                raise ValueError(f"sidecar {version_id}: in_features mismatch for {name}")
            values[name] = tensor
        failure = None
        base_answers = None
        fp = vdir / "failure.json"
        if fp.exists():
            failure = json.loads(fp.read_text(encoding="utf-8"))
        bp = vdir / "base_answers.json"
        if bp.exists():
            base_answers = json.loads(bp.read_text(encoding="utf-8"))
        return Sidecar(
            version_id=version_id,
            rows_by_layer=rows,
            values_by_layer=values,
            param_mode=str(meta.get("param_mode", manifest.get("param_mode", "delta"))),
            manifest=manifest,
            failure=failure,
            base_answers=base_answers,
        )

    def load_current(self) -> Sidecar | None:
        cid = self.current_id()
        return self.load(cid) if cid else None

    # ------------------------------------------------------------- commit
    def commit(
        self,
        rows_by_layer: dict[str, list[int]],
        values_by_layer: dict[str, torch.Tensor],
        param_mode: str,
        failure: Failure,
        base_answers: list[str] | None,
        metrics: dict[str, Any],
        cfg: TrainConfig,
        parent: str | None = None,
        status: str = "accepted",
        advance: bool = True,
        expected_current: str | None = None,
        check_current: bool = False,
    ) -> Sidecar:
        if check_current and self.current_id() != expected_current:
            raise RuntimeError(
                f"current moved: expected {expected_current!r}, got {self.current_id()!r}"
            )
        if not rows_by_layer:
            raise ValueError("empty sidecar")
        if set(rows_by_layer) != set(values_by_layer):
            raise ValueError("rows/values keys mismatch")
        if status not in {"accepted", "rejected", "candidate"}:
            raise ValueError(f"bad status: {status}")
        if parent is None:
            parent = self.current_id()

        version_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        final = self._version_dir(version_id)
        tmp = self.versions_dir / f".tmp-{version_id}"
        tmp.mkdir(parents=True, exist_ok=False)

        tensors: dict[str, torch.Tensor] = {}
        rows: dict[str, list[int]] = {}
        for name, rows_i in rows_by_layer.items():
            uniq = sorted({int(r) for r in rows_i})
            vals = values_by_layer[name].detach().float().cpu()
            if vals.ndim != 2 or tuple(vals.shape) != (
                len(uniq),
                self.ledger.in_features(name),
            ):
                raise ValueError(
                    f"{name}: values shape {tuple(vals.shape)} != "
                    f"{(len(uniq), self.ledger.in_features(name))}"
                )
            n_rows = self.ledger.shapes[self.ledger.tensor_name(name)][0]
            if uniq and (uniq[0] < 0 or uniq[-1] >= n_rows):
                raise IndexError(f"{name}: row out of range")
            rows[name] = uniq
            tensors[name] = vals.contiguous()
        save_file(tensors, str(tmp / "sidecar.safetensors"))
        write_json(
            tmp / "sidecar_meta.json",
            {
                "rows": rows,
                "param_mode": param_mode,
                "base_sha256": self.base_sha,
                "parent_version": parent,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        (tmp / "failure.json").write_text(
            json.dumps(failure.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (tmp / "base_answers.json").write_text(
            json.dumps(base_answers or [], ensure_ascii=False), encoding="utf-8"
        )

        config_hash = hashlib.sha256(
            json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        manifest = {
            "version_id": version_id,
            "parent_version": parent,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "base_sha256": self.base_sha,
            "sidecar_sha256": sha256_file(tmp / "sidecar.safetensors"),
            "param_mode": param_mode,
            "failure_ids": [failure.id],
            "failure_sha256": hashlib.sha256(
                json.dumps(failure.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "base_answers_sha256": hashlib.sha256(
                json.dumps(base_answers or [], ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "n_rows": int(sum(len(v) for v in rows.values())),
            "n_layers": len(rows),
            "config_hash": config_hash,
            "code_version": "qlamt-0.1.0",
            "metrics": json_safe(metrics),
        }
        write_json(tmp / "manifest.json", manifest)
        os.replace(tmp, final)
        if status == "accepted" and advance:
            self._set_current(version_id)
        return self.load(version_id)

    # ------------------------------------------------------------ regression
    def chain_ids(self) -> list[str]:
        cid = self.current_id()
        out: list[str] = []
        seen: set[str] = set()
        while cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
            try:
                cid = self._read_manifest(cid).get("parent_version")
            except FileNotFoundError:
                cid = None
        return list(reversed(out))

    def regression_items(self, limit: int = 64) -> list[dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        for vid in self.chain_ids():
            m = self._read_manifest(vid)
            if m.get("status") != "accepted":
                continue
            sc = self.load(vid)
            if sc.failure is None:
                continue
            fid = str(sc.failure.get("id", vid))
            items[fid] = {
                "version_id": vid,
                "failure": sc.failure,
                "base_answers": sc.base_answers,
                "metrics": m.get("metrics", {}),
            }
        out = list(items.values())
        return out[-max(0, int(limit)) :] if limit else out

    def history(self) -> list[dict[str, Any]]:
        out = []
        if not self.versions_dir.exists():
            return out
        for d in sorted(self.versions_dir.iterdir()):
            if not d.is_dir() or d.name.startswith(".tmp-"):
                continue
            mp = d / "manifest.json"
            if mp.exists():
                out.append(json.loads(mp.read_text(encoding="utf-8")))
        return out


# ---------------------------------------------------------------------------
# Streaming sharded release (for single-file 7B/13B models)
# ---------------------------------------------------------------------------
def shard_and_patch(
    source_dir: str | Path,
    dest_dir: str | Path,
    rows_by_layer: dict[str, list[int]],
    values_by_layer: dict[str, torch.Tensor],
    param_mode: str,
    max_shard_bytes: int = 1_500_000_000,
) -> dict[str, Any]:
    """Copy a model into shards while patching rows, one tensor at a time.

    A single 7B fp16 safetensors file is ~15GB; loading it wholesale to rewrite
    a few rows is wasteful and may hit host RAM limits.  This writes a sharded
    release (index + <=~1.5GB shards) and only holds one source tensor at a
    time.
    """
    source = LedgerManager(source_dir)
    edits: dict[str, tuple[list[int], torch.Tensor]] = {}
    for name, rows in rows_by_layer.items():
        tname = source.tensor_name(name)
        vals = values_by_layer[name].detach().float().cpu()
        if len(rows) != vals.shape[0]:
            raise ValueError(f"{name}: rows != values")
        edits[tname] = ([int(r) for r in rows], vals)

    dest = Path(dest_dir)
    tmp = dest.parent / (dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=False)
    # Non-weight files (config/tokenizer) are linked/copied as-is.
    for src in source.model_dir.iterdir():
        if not src.is_file() or src.suffix == ".safetensors":
            continue
        if src.name == "model.safetensors.index.json":
            continue
        dst = tmp / src.name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    weight_map: dict[str, str] = {}
    shard_names: list[str] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    total_bytes = 0
    expected: dict[str, torch.Tensor] = {}

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        sname = f"model-{len(shard_names):05d}.safetensors"
        save_file(
            {k: v.contiguous() for k, v in current.items()},
            str(tmp / sname),
            metadata={"format": "pt"},
        )
        for k in current:
            weight_map[k] = sname
        shard_names.append(sname)
        current = {}
        current_bytes = 0

    for tname in sorted(source.file_map):
        src_file = source.file_map[tname]
        with safe_open(str(src_file), framework="pt", device="cpu") as f:
            t = f.get_tensor(tname)
        if tname in edits:
            rows, vals = edits[tname]
            base = t.float().cpu()
            if rows and (rows[0] < 0 or rows[-1] >= base.shape[0]):
                raise IndexError(f"{tname}: row out of range")
            if param_mode in {"splice", "splice_fp16"}:
                new_rows = vals.to(base.dtype)
            elif param_mode == "delta":
                new_rows = (base[rows] + vals).to(base.dtype)
            else:
                raise ValueError(f"unknown param_mode: {param_mode}")
            t = t.clone()
            t[rows] = new_rows.to(t.dtype)
            expected[tname] = t[rows].clone()
        nbytes = int(t.numel() * t.element_size())
        if current and current_bytes + nbytes > max_shard_bytes:
            flush()
        current[tname] = t
        current_bytes += nbytes
        total_bytes += nbytes
        if current_bytes >= max_shard_bytes:
            flush()
    flush()

    shard_files = []
    for sname in shard_names:
        p = tmp / sname
        shard_files.append({"name": sname, "size_bytes": int(p.stat().st_size)})
    index = {
        "metadata": {"total_size": int(total_bytes)},
        "weight_map": weight_map,
    }
    write_json(tmp / "model.safetensors.index.json", index)
    verified = True
    for tname, want in expected.items():
        with safe_open(str(tmp / weight_map[tname]), framework="pt", device="cpu") as f:
            got = f.get_tensor(tname)[edits[tname][0]]
        if not torch.equal(got.float(), want.float()):
            verified = False
            break
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return {
        "sharded": True,
        "n_shards": len(shard_names),
        "shards": shard_files,
        "verified": verified,
        "max_shard_bytes": max_shard_bytes,
    }


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------
def release_version(
    store: VersionStore,
    version_id: str,
    cfg: TrainConfig,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    from .train import build_background_batch
    from .verify import background_metrics, evaluate_failure

    cfg.validate()
    t_start = time.perf_counter()
    sc = store.load(version_id)
    if sc.manifest.get("status") != "accepted":
        raise ValueError(f"version {version_id} is not accepted")
    root = Path(out_dir) if out_dir else store.releases_dir
    release_dir = root / version_id
    if release_dir.exists():
        shutil.rmtree(release_dir)

    source = store.ledger
    affected_tensors = [source.tensor_name(n) for n in sc.rows_by_layer]
    max_shard_size = max(source.file_map[t].stat().st_size for t in affected_tensors)
    if max_shard_size > source.max_rewrite_bytes:
        # 7B/13B single-file model: stream into <=1.5GB shards, patching rows.
        write_report = shard_and_patch(
            store.base_dir,
            release_dir,
            sc.rows_by_layer,
            sc.values_by_layer,
            sc.param_mode,
            max_shard_bytes=1_500_000_000,
        )
    else:
        ledger = LedgerManager.prepare(store.base_dir, release_dir)
        base_rows = ledger.read_rows_by_layer(sc.rows_by_layer)
        merged: dict[str, torch.Tensor] = {}
        for name, rows_i in sc.rows_by_layer.items():
            base = base_rows[name]
            vals = sc.values_by_layer[name].to(base.dtype)
            if sc.param_mode in {"splice", "splice_fp16"}:
                merged[name] = vals
            elif sc.param_mode == "delta":
                merged[name] = (base.float() + vals.float()).to(base.dtype)
            else:
                raise ValueError(f"unknown param_mode: {sc.param_mode}")
        write_report = ledger.write_rows(
            merged, sc.rows_by_layer, undo_dir=None, verify=True
        )
    if not write_report.get("verified"):
        raise RuntimeError("release write-back verification failed")

    failure = Failure.from_dict(sc.failure or {})
    regression = store.regression_items(limit=cfg.regression_size)
    regression = [x for x in regression if x["failure"].get("id") != failure.id]

    # Clean base background, then the quantized release.
    base_cfg = TrainConfig.from_dict({**cfg.to_dict(), "model_path": str(store.base_dir)})
    base_rt = ModelRuntime.load(base_cfg)
    try:
        bg_batch, bg_texts = build_background_batch(base_rt, cfg)
        base_bg = (
            base_rt.background_logprobs(bg_batch, use_deltas=False, no_grad=True)
            if bg_texts
            else None
        )
    finally:
        base_rt.close()

    rel_cfg = TrainConfig.from_dict(
        {**cfg.to_dict(), "model_path": str(release_dir), "mode": "merge", "quant": "nf4"}
    )
    rel_rt = ModelRuntime.load(rel_cfg)
    try:
        main_eval = evaluate_failure(
            rel_rt,
            failure,
            use_deltas=False,
            generate=True,
            wrong_spans=sc.base_answers,
        )
        reg_rows = []
        for item in regression:
            f = Failure.from_dict(item["failure"])
            ev = evaluate_failure(
                rel_rt,
                f,
                use_deltas=False,
                generate=True,
                wrong_spans=item.get("base_answers"),
            )
            reg_rows.append(
                {
                    "version_id": item["version_id"],
                    "failure_id": f.id,
                    "fix_ratio": ev.fix_ratio,
                    "previous_fix_ratio": item.get("metrics", {}).get("hybrid_fix_ratio"),
                    "generation": ev.variants[0].get("generation") if ev.variants else None,
                }
            )
        cur_bg = (
            rel_rt.background_logprobs(bg_batch, use_deltas=False, no_grad=True)
            if bg_texts
            else None
        )
        bg = background_metrics(base_bg, cur_bg, bg_texts)
    finally:
        rel_rt.close()

    fix_ok = main_eval.fix_ratio >= float(cfg.verify_fix_ratio)
    reg_ok = all(
        r["fix_ratio"]
        >= max(
            0.0,
            (float(r["previous_fix_ratio"]) - 0.01)
            if r["previous_fix_ratio"] is not None
            else float(cfg.verify_fix_ratio),
        )
        for r in reg_rows
    )
    bg_ok = (
        bg["kl_mean"] <= float(cfg.verify_bg_kl_max)
        and bg["kl_max"] <= float(cfg.verify_bg_kl_max_single)
    )
    published = bool(fix_ok and reg_ok and bg_ok)
    manifest = {
        "release_id": version_id,
        "version_id": version_id,
        "base_sha256": store.base_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "published": published,
        "reasons": [
            *([] if fix_ok else [f"fix_ratio {main_eval.fix_ratio:.3f} < {cfg.verify_fix_ratio}"]),
            *([] if reg_ok else ["regression failed"]),
            *(
                []
                if bg_ok
                else [f"background KL {bg['kl_mean']:.4f}/{bg['kl_max']:.4f} exceeds"]
            ),
        ],
        "failure": main_eval.to_dict(),
        "regression": reg_rows,
        "background": bg,
        "write_back": write_report,
        "release_dir": str(release_dir.resolve()),
        "target_format": "nf4",
        "timings": {"total_s": round(time.perf_counter() - t_start, 3)},
    }
    write_json(release_dir / "manifest.json", manifest)
    return manifest
