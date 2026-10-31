from __future__ import annotations

import argparse
import json
import sys

from .ledger import LedgerManager
from .multiround import run_fix_multi
from .pipeline import run_fix
from .runtime import ModelRuntime
from .types import Failure, TrainConfig
from .versions import VersionStore, release_version


def _load_cfg(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig.load(args.config)
    if getattr(args, "set", None):
        cfg.apply_overrides(args.set)
    return cfg


def cmd_fix(args: argparse.Namespace) -> int:
    failure = Failure.load(args.failure)
    cfg = _load_cfg(args)
    if args.mode:
        cfg.mode = args.mode
    if args.param_mode:
        cfg.param_mode = args.param_mode
    if args.versions_root:
        cfg.versions_root = args.versions_root
    if args.no_write:
        cfg.write_back = False
    if args.no_post_requant:
        cfg.verify_post_requant = False
    report = (
        run_fix_multi(failure, cfg, run_dir=args.out)
        if cfg.mode == "sidecar"
        else run_fix(failure, cfg, run_dir=args.out)
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        stages = report.get("stages", {})
        train = stages.get("train", {})
        verify_stage = stages.get("verify", {})
        post = stages.get("post_requant")
        print(f"call_id      : {report.get('call_id')}")
        print(f"result       : {report.get('result')}")
        if report.get("error"):
            print(f"error        : {report.get('error')}")
        print(f"named rows   : {stages.get('ablate', {}).get('n_named', 0)}")
        print(f"train steps  : {train.get('steps_run', 0)}  ({train.get('stopped_reason', '')})")
        print(f"fix ratio    : {verify_stage.get('failure', {}).get('fix_ratio')}")
        print(f"bg KL mean   : {verify_stage.get('background', {}).get('kl_mean')}")
        if post is not None:
            print(f"post requant : {post.get('passed')}")
        if report.get("version"):
            print(
                f"version      : {report['version'].get('version_id')} "
                f"({report['version'].get('status')})"
            )
        cum = stages.get("cumulative")
        if cum is not None:
            print(f"cum ratio    : {cum.get('max_ratio')} (limit {cum.get('limit')})")
        reg = stages.get("regression")
        if reg is not None:
            print(f"regression   : {reg.get('passed')} (mean {reg.get('mean_fix_ratio')})")
        print(f"report       : {report.get('run_dir')}")
    result = report.get("result")
    return 0 if result in {"written", "version_committed"} else 2


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args).resolved()
    runtime = ModelRuntime.load(cfg)
    try:
        print(
            runtime.generate(
                args.prompt, use_deltas=False, max_new_tokens=args.max_new_tokens
            )
        )
    finally:
        runtime.close()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args).resolved()
    runtime = ModelRuntime.load(cfg)
    try:
        print(json.dumps(runtime.info(), ensure_ascii=False, indent=2))
        print("target modules:")
        for n in runtime.target_names:
            print(f"  {n}  rows={runtime.num_rows(n)}  in={runtime.in_features(n)}")
    finally:
        runtime.close()
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    ledger = LedgerManager(args.ledger)
    report = ledger.rollback(args.undo)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _release_summary(report: dict) -> str:
    lines = [
        f"release      : {report.get('release_id')}",
        f"published    : {report.get('published')}",
        f"failure fix  : {report.get('failure', {}).get('fix_ratio')}",
        f"regression   : {len(report.get('regression', []))} item(s)",
        f"bg KL        : {report.get('background', {}).get('kl_mean')}",
        f"reasons      : {report.get('reasons')}",
        f"dir          : {report.get('release_dir')}",
    ]
    return "\n".join(lines)


def cmd_release(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args).resolved()
    store = VersionStore(cfg.model_path, args.versions_root or cfg.versions_root)
    report = release_version(
        store,
        args.version,
        cfg,
        out_dir=args.out or cfg.release_root,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_release_summary(report))
    return 0 if report.get("published") else 2


def cmd_versions(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    store = VersionStore(cfg.model_path, args.versions_root or cfg.versions_root)
    rows = store.history()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"current: {store.current_id()}")
        for m in rows:
            print(
                f"{m.get('version_id')}  {str(m.get('status')):9s}  "
                f"parent={m.get('parent_version')}  rows={m.get('n_rows')}  "
                f"fix={m.get('metrics', {}).get('hybrid_fix_ratio')}"
            )
    return 0


def cmd_version_rollback(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    store = VersionStore(cfg.model_path, args.versions_root or cfg.versions_root)
    manifest = store.rollback(args.version)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qlamt", description="Q-LAMT: quantized locate-ablate-materialize-train row repair")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default="configs/smoke.json")
        sp.add_argument("--set", action="append", default=[], help="key=value override")
        sp.add_argument("--debug", action="store_true", help="re-raise exceptions")

    fix = sub.add_parser("fix", help="run locate -> ablate -> train -> write-back/sidecar")
    fix.add_argument("--failure", required=True)
    fix.add_argument("--out", default=None)
    fix.add_argument("--mode", choices=["merge", "sidecar"], default=None)
    fix.add_argument("--param-mode", choices=["delta", "splice_fp16"], default=None)
    fix.add_argument("--versions-root", default=None)
    fix.add_argument("--no-write", action="store_true")
    fix.add_argument("--no-post-requant", action="store_true")
    fix.add_argument("--json", action="store_true")
    common(fix)
    fix.set_defaults(func=cmd_fix)

    probe = sub.add_parser("probe", help="generate a base answer (for wrong_span)")
    probe.add_argument("--prompt", required=True)
    probe.add_argument("--max-new-tokens", type=int, default=16)
    common(probe)
    probe.set_defaults(func=cmd_probe)

    info = sub.add_parser("info", help="print runtime and target module info")
    common(info)
    info.set_defaults(func=cmd_info)

    rb = sub.add_parser("rollback", help="restore rows from a single-call undo snapshot")
    rb.add_argument("--ledger", required=True)
    rb.add_argument("--undo", required=True)
    rb.set_defaults(func=cmd_rollback)

    rel = sub.add_parser("release", help="merge a version and quantize to NF4/fp4")
    rel.add_argument("--version", required=True)
    rel.add_argument("--versions-root", default=None)
    rel.add_argument("--out", default=None)
    rel.add_argument("--json", action="store_true")
    common(rel)
    rel.set_defaults(func=cmd_release)

    vs = sub.add_parser("versions", help="list version manifests")
    vs.add_argument("--versions-root", default=None)
    vs.add_argument("--json", action="store_true")
    common(vs)
    vs.set_defaults(func=cmd_versions)

    vrb = sub.add_parser("version-rollback", help="switch current to an accepted version")
    vrb.add_argument("--version", required=True)
    vrb.add_argument("--versions-root", default=None)
    common(vrb)
    vrb.set_defaults(func=cmd_version_rollback)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
